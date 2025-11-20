#!/usr/bin/env python3
# wecc_poc_with_labels_and_basemap.py
#
# One-file WECC POC:
#   - builds graph from your CSVs
#   - force-directed layout normalized to WECC bbox (projected to EPSG:3857)
#   - satellite basemap plane (Esri World Imagery) under the scene
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
#   wecc_outline.geojson (Polygon/MultiPolygon in lon/lat EPSG:4326)

import json
import math
import os
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

# Optional: PyVista for 3D; otherwise fall back to 2D
try:
    import pyvista as pv
    _HAS_PYVISTA = True
except Exception:
    _HAS_PYVISTA = False

# Basemap bits
try:
    import contextily as cx
    from pyproj import Transformer
    _HAS_BASEMAP = True
except Exception:
    _HAS_BASEMAP = False

# --------- Configuration ----------
DATA_DIR = Path("/home/aidan/sandbox/cs237-transmission-grid-project/data/WECC_metadata")
FILES = {
    "bus": DATA_DIR / "bus_info.csv",
    "branch": DATA_DIR / "branch_info.csv",
    "gen": DATA_DIR / "gen_info.csv",
    "load": DATA_DIR / "load_info.csv",
    "outline": DATA_DIR / "wecc_outline.geojson",
}

USE_BASEMAP = True          # set False to disable basemap
BASEMAP_ZOOM = 6            # 4..9 typical; higher = sharper (larger download)
BASEMAP_Z_OFFSET = -1500.0  # meters below graph, tweak if needed

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

# --------- GeoJSON bbox (EPSG:4326 lon/lat) ---------

def geojson_bounds_ll(geojson_obj):
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

def project_bbox_to_3857(bbox_ll):
    xmin, xmax, ymin, ymax = bbox_ll
    to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
    x0, y0 = to_3857(xmin, ymin)
    x1, y1 = to_3857(xmax, ymax)
    # normalize ordering (in case inputs cross antimeridian or similar)
    xmin_m, xmax_m = min(x0, x1), max(x0, x1)
    ymin_m, ymax_m = min(y0, y1), max(y0, y1)
    return (xmin_m, xmax_m, ymin_m, ymax_m)

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
        a = int(r["from_bus"]); b = int(r["to_bus"])
        if f"bus:{a}" in G and f"bus:{b}" in G:
            G.add_edge(
                f"bus:{a}",
                f"bus:{b}",
                kind=str(r.get("branch_type", "Line")).strip(),
                name=str(r.get("branch_name", "")).strip(),
            )
    return G

# --------- Layout (mapped to EPSG:3857 bbox) ---------

def compute_layout_positions(G: nx.Graph, bbox_3857):
    """
    Returns dict: node -> (x, y, z) in EPSG:3857 meters.
    1) spring_layout on bus-only backbone
    2) normalize to [0,1]
    3) scale into EPSG:3857 bbox
    4) attach gens/loads near parent bus with a small radial offset
    """
    xmin, xmax, ymin, ymax = bbox_3857
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
                r = 0.015 * (xmax - xmin)  # small offset relative to width
                x = px + r * math.cos(ang)
                y = py + r * math.sin(ang)
                z = 50.0 if d["kind"] == "gen" else 0.0
                pos3[n] = (x, y, z)
            else:
                pos3[n] = (xmin, ymin, 0.0)
    return pos3

# --------- Basemap (EPSG:3857) ---------

def build_basemap_plane(bbox_3857, zoom=6, z_offset=-1500.0):
    """
    Fetches satellite tiles for the bbox and returns (plane_mesh, texture).
    plane is centered at bbox, dimensions match bbox size, at z=z_offset.
    """
    if not _HAS_BASEMAP:
        raise RuntimeError("contextily/pyproj not available. Install with: conda install -c conda-forge contextily pyproj")

    xmin, xmax, ymin, ymax = bbox_3857
    # Download satellite imagery as one image + its extent
    img, ext = cx.bounds2img(xmin, ymin, xmax, ymax, zoom=zoom, source=cx.providers.Esri.WorldImagery)
    left, right, bottom, top = ext
    width = right - left
    height = top - bottom

    # Build a textured plane with UVs
    plane = pv.Plane(center=((left + right) / 2, (bottom + top) / 2, z_offset),
                     i_size=width, j_size=height,
                     i_resolution=max(img.shape[1] - 1, 1),
                     j_resolution=max(img.shape[0] - 1, 1))
    plane.texture_map_to_plane(inplace=True)
    tex = pv.Texture(img)  # HxWx(3/4) uint8
    return plane, tex

# --------- Rendering (PyVista) ---------

def render_network_pyvista(G: nx.Graph, pos: dict, bbox_3857, use_basemap=True, zoom=6, z_offset=-1500.0):
    plotter = pv.Plotter(window_size=(1200, 850))
    xmin, xmax, ymin, ymax = bbox_3857

    # Basemap (optional)
    if use_basemap and _HAS_BASEMAP:
        try:
            plane, tex = build_basemap_plane(bbox_3857, zoom=zoom, z_offset=z_offset)
            plotter.add_mesh(plane, texture=tex, smooth_shading=False)
        except Exception as e:
            print(f"[basemap] skipped: {e}")

    # Subtle frame slab if no basemap (visual reference)
    if not use_basemap or not _HAS_BASEMAP:
        frame_z = -0.001 * (ymax - ymin)
        frame = pv.Cube(center=((xmin + xmax)/2, (ymin + ymax)/2, frame_z),
                        x_length=(xmax - xmin),
                        y_length=(ymax - ymin),
                        z_length=(ymax - ymin) * 0.0005)
        plotter.add_mesh(frame, opacity=0.05, show_edges=True)

    # Edges
    def add_edge(u, v, radius):
        axyz = np.array(pos[u])
        bxyz = np.array(pos[v])
        pts = np.vstack([axyz, bxyz])
        line = pv.Spline(pts, 2)
        tube = line.tube(radius=radius)
        plotter.add_mesh(tube, smooth_shading=True)

    for u, v, ed in G.edges(data=True):
        kind = ed.get("kind", "Line")
        radius = 0.0015 * (xmax - xmin) if kind == "Line" else 0.0009 * (xmax - xmin)
        add_edge(u, v, radius)

    # Nodes
    for n, d in G.nodes(data=True):
        x, y, z = pos[n]
        if d["kind"] == "bus":
            sph = pv.Sphere(radius=0.003 * (xmax - xmin), center=(x, y, z))
            plotter.add_mesh(sph, color="white")
        elif d["kind"] == "gen":
            sph = pv.Sphere(radius=0.0024 * (xmax - xmin), center=(x, y, z))
            plotter.add_mesh(sph, color="green")
        else:  # load
            sph = pv.Sphere(radius=0.0024 * (xmax - xmin), center=(x, y, z))
            plotter.add_mesh(sph, color="orange")

    # ---------- LABELS ----------
    # Persistent bus labels (toggle with 'l')
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

    # Click-to-show a single bus label
    hud = {"actor": None}
    def show_pick_label(picked):
        if picked is None:
            return
        xyz = np.array(picked)
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
        z_off = 0.01 * (ymax - ymin)
        hud["actor"] = plotter.add_point_labels(
            [[x, y, z + z_off]],
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

    # Camera/UI
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
    labels = {n: G.nodes[n].get("bus_name") or f"Bus {G.nodes[n]['bus_number']}" for n in buses}
    nx.draw_networkx_labels(G, pos2d, labels=labels, font_size=6)
    ax.set_title("WECC POC (EPSG:3857 layout) — 2D fallback")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()

# --------- Main ---------

def main():
    buses, branches, gens, loads, outline = load_tables()
    bbox_ll = geojson_bounds_ll(outline)
    bbox_3857 = project_bbox_to_3857(bbox_ll)

    G = build_graph(buses, branches, gens, loads)
    pos = compute_layout_positions(G, bbox_3857)

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    if _HAS_PYVISTA:
        render_network_pyvista(
            G, pos, bbox_3857,
            use_basemap=(USE_BASEMAP and _HAS_BASEMAP),
            zoom=BASEMAP_ZOOM,
            z_offset=BASEMAP_Z_OFFSET
        )
    else:
        print("PyVista not found; using 2D fallback view.")
        render_network_matplotlib(G, pos)

if __name__ == "__main__":
    main()

