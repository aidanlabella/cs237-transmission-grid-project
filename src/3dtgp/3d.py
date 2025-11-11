#!/usr/bin/env python3
# wecc_poc_with_labels.py
# One-file WECC POC:
#   - builds graph from your CSVs
#   - force-directed layout normalized to WECC outline bbox
#   - interactive 3D render (PyVista) with:
#       * persistent bus labels (toggle with 'l')
#       * click-to-label bus names near the pick
#   - 2D matplotlib fallback if PyVista isn't available
#
# Expected files (adjust DATA_DIR below if needed):
#   bus_info.csv        [bus_number, bus_name, bus_area]
#   branch_info.csv     [from_bus, to_bus, branch_name, branch_type]
#   gen_info.csv        [bus_number, gen_name]
#   load_info.csv       [bus_number, load_name]
#   wecc_outline.geojson (Polygon/MultiPolygon in lon/lat)

import json
import math
import os
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

# Optional: PyVista for 3D; otherwise we’ll fall back to NetworkX/Matplotlib
try:
    import pyvista as pv
    _HAS_PYVISTA = True
except Exception:
    _HAS_PYVISTA = False

# --------- Configuration ----------
# Point to your data directory. Default matches the files you shared.
DATA_DIR = Path("/home/aidan/sandbox/cs237-transmission-grid-project/data/WECC_metadata")

FILES = {
    "bus": DATA_DIR / "bus_info.csv",
    "branch": DATA_DIR / "branch_info.csv",
    "gen": DATA_DIR / "gen_info.csv",
    "load": DATA_DIR / "load_info.csv",
    "outline": DATA_DIR / "wecc_outline.geojson",
}


# --------- IO ---------

def load_tables():
    buses = pd.read_csv(FILES["bus"])
    branches = pd.read_csv(FILES["branch"])
    gens = pd.read_csv(FILES["gen"])
    loads = pd.read_csv(FILES["load"])
    with open(FILES["outline"], "r") as f:
        outline = json.load(f)
    # Schema guards
    for req in ["bus_number", "bus_name", "bus_area"]:
        assert req in buses.columns, f"Missing '{req}' in bus_info.csv"
    for req in ["from_bus", "to_bus"]:
        assert req in branches.columns, f"Missing '{req}' in branch_info.csv"
    for req in ["bus_number"]:
        assert req in gens.columns, f"Missing '{req}' in gen_info.csv"
        assert req in loads.columns, f"Missing '{req}' in load_info.csv"
    return buses, branches, gens, loads, outline


# --------- GeoJSON bbox ---------

def geojson_bounds(geojson_obj):
    def coords_iter(feature):
        geom = feature.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])
        if gtype == "Polygon":
            for ring in coords:
                for x, y in ring:
                    yield x, y
        elif gtype == "MultiPolygon":
            for poly in coords:
                for ring in poly:
                    for x, y in ring:
                        yield x, y

    xs, ys = [], []
    for feat in geojson_obj.get("features", []):
        for x, y in coords_iter(feat):
            xs.append(x)
            ys.append(y)
    if not xs:
        return (0.0, 1.0, 0.0, 1.0)
    return (min(xs), max(xs), min(ys), max(ys))


# --------- Graph build ---------

def build_graph(buses: pd.DataFrame, branches: pd.DataFrame,
                gens: pd.DataFrame, loads: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    # buses
    for _, r in buses.iterrows():
        bn = int(r["bus_number"])
        G.add_node(
            f"bus:{bn}",
            kind="bus",
            bus_number=bn,
            bus_name=str(r["bus_name"]).strip(),
            bus_area=int(r["bus_area"]),
        )
    # generators
    for _, r in gens.iterrows():
        bn = int(r["bus_number"])
        gen_name = str(r.get("gen_name", f"gen-{bn}")).strip()
        if f"bus:{bn}" in G:
            gnode = f"gen:{gen_name}"
            G.add_node(gnode, kind="gen", bus_number=bn, name=gen_name)
            G.add_edge(f"bus:{bn}", gnode, kind="gen_tie")
    # loads
    for _, r in loads.iterrows():
        bn = int(r["bus_number"])
        load_name = str(r.get("load_name", f"load-{bn}")).strip()
        if f"bus:{bn}" in G:
            lnode = f"load:{load_name}"
            G.add_node(lnode, kind="load", bus_number=bn, name=load_name)
            G.add_edge(f"bus:{bn}", lnode, kind="load_tie")
    # branches (bus↔bus)
    for _, r in branches.iterrows():
        a = int(r["from_bus"])
        b = int(r["to_bus"])
        if f"bus:{a}" in G and f"bus:{b}" in G:
            G.add_edge(
                f"bus:{a}",
                f"bus:{b}",
                kind=str(r.get("branch_type", "Line")).strip(),
                name=str(r.get("branch_name", "")).strip(),
            )
    return G


# --------- Layout ---------

def compute_layout_positions(G: nx.Graph, bbox):
    """
    Returns dict: node -> (x, y, z)
    1) spring_layout on bus-only backbone
    2) normalize to [0,1]
    3) scale into WECC bbox (lon/lat extent)
    4) attach gens/loads near their parent bus with a small radial offset
    """
    xmin, xmax, ymin, ymax = bbox
    bus_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "bus"]
    H = G.subgraph(bus_nodes).copy()
    if len(H) == 0:
        raise ValueError("No bus nodes to layout.")

    pos2 = nx.spring_layout(H, dim=2, k=None, iterations=200, seed=42)

    xs = np.array([p[0] for p in pos2.values()])
    ys = np.array([p[1] for p in pos2.values()])
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    xr = max(1e-9, x1 - x0)
    yr = max(1e-9, y1 - y0)

    def norm(p):
        nx_ = (p[0] - x0) / xr
        ny_ = (p[1] - y0) / yr
        return nx_, ny_

    def to_bbox(p01):
        x = xmin + p01[0] * (xmax - xmin)
        y = ymin + p01[1] * (ymax - ymin)
        return x, y

    pos3 = {}
    for n, p in pos2.items():
        p01 = norm(p)
        x, y = to_bbox(p01)
        pos3[n] = (x, y, 0.0)

    for n, d in G.nodes(data=True):
        if d.get("kind") in ("gen", "load"):
            bn = d["bus_number"]
            parent = f"bus:{bn}"
            if parent in pos3:
                px, py, pz = pos3[parent]
                h = abs(hash(n)) % 360
                ang = math.radians(h)
                r = 0.15 * (xmax - xmin) / 50.0
                x = px + r * math.cos(ang)
                y = py + r * math.sin(ang)
                z = 0.02 * (ymax - ymin) / 50.0 if d["kind"] == "gen" else 0.0
                pos3[n] = (x, y, z)
            else:
                pos3[n] = (xmin, ymin, 0.0)
    return pos3


# --------- Rendering (PyVista) ---------

def render_network_pyvista(G: nx.Graph, pos: dict, bbox):
    plotter = pv.Plotter(window_size=(1200, 850))
    xmin, xmax, ymin, ymax = bbox

    # subtle frame slab beneath the graph (visual reference)
    frame_z = -0.005 * (ymax - ymin)
    frame = pv.Cube(center=((xmin + xmax)/2, (ymin + ymax)/2, frame_z),
                    x_length=(xmax - xmin),
                    y_length=(ymax - ymin),
                    z_length=(ymax - ymin) * 0.0005)
    plotter.add_mesh(frame, opacity=0.05, show_edges=True)

    # edges
    def add_edge(u, v, radius):
        axyz = np.array(pos[u])
        bxyz = np.array(pos[v])
        pts = np.vstack([axyz, bxyz])
        line = pv.Spline(pts, 2)
        tube = line.tube(radius=radius)
        plotter.add_mesh(tube, smooth_shading=True)

    for u, v, ed in G.edges(data=True):
        kind = ed.get("kind", "Line")
        radius = 0.002 * (xmax - xmin) if kind == "Line" else 0.001 * (xmax - xmin)
        add_edge(u, v, radius)

    # nodes
    for n, d in G.nodes(data=True):
        x, y, z = pos[n]
        if d["kind"] == "bus":
            # sph = pv.Sphere(radius=0.004 * (xmax - xmin), center=(x, y, z))
            cone = pv.Cone(radius=0.004 * (xmax - xmin), center=(x, y, z), direction=(0,0,1))
            plotter.add_mesh(cone, color="blue")
        elif d["kind"] == "gen":
            # sph = pv.Sphere(radius=0.003 * (xmax - xmin), center=(x, y, z))
            cyl = pv.Cylinder(radius=0.003 * (xmax - xmin), center=(x, y, z), direction=(0,0,1))
            plotter.add_mesh(cyl, color="red")
        else:  # load
            sph = pv.Sphere(radius=0.003 * (xmax - xmin), center=(x, y, z))
            plotter.add_mesh(sph, color="orange")

    # ---------- LABELS ----------
    # 1) Persistent labels for buses (toggle with 'l')
    bus_pts, bus_labels = [], []
    for n, d in G.nodes(data=True):
        if d.get("kind") == "bus":
            bus_pts.append(list(pos[n]))
            label = d.get("bus_name") or f"Bus {d['bus_number']}"
            bus_labels.append(str(label).strip())

    labels_actor = None
    if bus_pts:
        labels_actor = plotter.add_point_labels(
            bus_pts,
            bus_labels,
            point_size=0,
            font_size=12,
            text_color="black",
            shape=None,
            always_visible=False,
            render=False,
        )

        def toggle_labels():
            labels_actor.SetVisibility(not labels_actor.GetVisibility())
            plotter.render()

        plotter.add_key_event("l", toggle_labels)

    # 2) Click-to-show label near picked bus
    hud = {"actor": None}

    def show_pick_label(picked):
        if picked is None:
            return
        xyz = np.array(picked)
        # closest bus
        best_n, best_d2 = None, float("inf")
        for n, d in G.nodes(data=True):
            if d.get("kind") != "bus":
                continue
            p = np.array(pos[n])
            d2 = np.sum((p - xyz) ** 2)
            if d2 < best_d2:
                best_n, best_d2 = n, d2
        if best_n is None:
            return
        label = G.nodes[best_n].get("bus_name") or f"Bus {G.nodes[best_n]['bus_number']}"
        if hud["actor"] is not None:
            plotter.remove_actor(hud["actor"])
        x, y, z = pos[best_n]
        z_offset = 0.003 * (plotter.bounds[3] - plotter.bounds[2])
        hud["actor"] = plotter.add_point_labels(
            [[x, y, z + z_offset]],
            [str(label)],
            point_size=0,
            font_size=14,
            text_color="black",
            shape="rounded_rect",
            always_visible=True,
        )
        plotter.render()

    plotter.enable_point_picking(callback=show_pick_label, show_message=False)

    # Optional: label gens/loads too (comment out if cluttered)
    gen_pts, gen_labels = [], []
    load_pts, load_labels = [], []
    for n, d in G.nodes(data=True):
        if d.get("kind") == "gen":
            gen_pts.append(list(pos[n])); gen_labels.append(d.get("name", "gen"))
        elif d.get("kind") == "load":
            load_pts.append(list(pos[n])); load_labels.append(d.get("name", "load"))

    if gen_pts:
        plotter.add_point_labels(gen_pts, gen_labels, point_size=0, font_size=11, text_color="green", render=False)
    if load_pts:
        plotter.add_point_labels(load_pts, load_labels, point_size=0, font_size=11, text_color="orange", render=False)

    # camera/UI
    plotter.add_axes()
    plotter.show_bounds(grid='front')
    plotter.add_text("Press 'l' to toggle bus labels • Click to show a bus label", font_size=10)
    plotter.show()


# --------- Fallback 2D ---------

def render_network_matplotlib(G: nx.Graph, pos: dict):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 8))
    pos2d = {k: (v[0], v[1]) for k, v in pos.items()}
    nx.draw_networkx_edges(G, pos2d, alpha=0.4, width=0.5)
    buses = [n for n, d in G.nodes(data=True) if d.get("kind") == "bus"]
    gens = [n for n, d in G.nodes(data=True) if d.get("kind") == "gen"]
    loads = [n for n, d in G.nodes(data=True) if d.get("kind") == "load"]
    nx.draw_networkx_nodes(G, pos2d, nodelist=buses, node_size=20, node_color="white", edgecolors="black")
    nx.draw_networkx_nodes(G, pos2d, nodelist=gens, node_size=12, node_color="green")
    nx.draw_networkx_nodes(G, pos2d, nodelist=loads, node_size=12, node_color="orange")
    # lightweight labels for buses (can be dense)
    labels = {n: G.nodes[n].get("bus_name") or f"Bus {G.nodes[n]['bus_number']}" for n in buses}
    nx.draw_networkx_labels(G, pos2d, labels=labels, font_size=6)
    ax.set_title("WECC POC (force-directed over WECC bbox) — 2D fallback")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()


# --------- Main ---------

def main():
    buses, branches, gens, loads, outline = load_tables()
    bbox = geojson_bounds(outline)
    G = build_graph(buses, branches, gens, loads)
    pos = compute_layout_positions(G, bbox)

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    if _HAS_PYVISTA:
        render_network_pyvista(G, pos, bbox)
    else:
        print("PyVista not found; using 2D fallback view.")
        render_network_matplotlib(G, pos)


if __name__ == "__main__":
    main()

