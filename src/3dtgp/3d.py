# Simple WECC proof-of-concept: graph + 3D render over the WECC outline
# Files expected:
#   /mnt/data/bus_info.csv        [bus_number, bus_name, bus_area]
#   /mnt/data/branch_info.csv     [from_bus, to_bus, branch_name, branch_type]
#   /mnt/data/gen_info.csv        [bus_number, gen_name]
#   /mnt/data/load_info.csv       [bus_number, load_name]
#   /mnt/data/wecc_outline.geojson (Polygon or MultiPolygon in lon/lat)
#
# Usage:
#   python wecc_poc.py
#
# Later: replace `compute_layout_positions()` with real projected coordinates.

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

DATA_DIR = Path("/Users/aidan/School/CSCI2370/project/data/WECC_metadata")

FILES = {
    "bus": DATA_DIR / "bus_info.csv",
    "branch": DATA_DIR / "branch_info.csv",
    "gen": DATA_DIR / "gen_info.csv",
    "load": DATA_DIR / "load_info.csv",
    "outline": DATA_DIR / "wecc_outline.geojson",
}

# -----------------------------
# Data loading
# -----------------------------

def load_tables():
    buses = pd.read_csv(FILES["bus"])
    branches = pd.read_csv(FILES["branch"])
    gens = pd.read_csv(FILES["gen"])
    loads = pd.read_csv(FILES["load"])
    with open(FILES["outline"], "r") as f:
        outline = json.load(f)
    # Basic schema guards
    for req in ["bus_number", "bus_name", "bus_area"]:
        assert req in buses.columns, f"Missing '{req}' in bus_info.csv"
    for req in ["from_bus", "to_bus"]:
        assert req in branches.columns, f"Missing '{req}' in branch_info.csv"
    for req in ["bus_number"]:
        assert req in gens.columns, f"Missing '{req}' in gen_info.csv"
        assert req in loads.columns, f"Missing '{req}' in load_info.csv"
    return buses, branches, gens, loads, outline

# -----------------------------
# Geometry helpers for the WECC outline (lon/lat) → just to get a bbox
# -----------------------------

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
        # Default square if something is off
        return (0.0, 1.0, 0.0, 1.0)
    return (min(xs), max(xs), min(ys), max(ys))

# -----------------------------
# Graph construction
# -----------------------------

def build_graph(buses: pd.DataFrame, branches: pd.DataFrame,
                gens: pd.DataFrame, loads: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    # Add bus nodes
    for _, r in buses.iterrows():
        bn = int(r["bus_number"])
        G.add_node(
            f"bus:{bn}",
            kind="bus",
            bus_number=bn,
            bus_name=str(r["bus_name"]),
            bus_area=int(r["bus_area"]),
        )
    # Add generator “leaf” nodes connected to their bus
    for _, r in gens.iterrows():
        bn = int(r["bus_number"])
        gen_name = str(r.get("gen_name", f"gen-{bn}"))
        if f"bus:{bn}" in G:
            gnode = f"gen:{gen_name}"
            G.add_node(gnode, kind="gen", bus_number=bn, name=gen_name)
            G.add_edge(f"bus:{bn}", gnode, kind="gen_tie")
    # Add load “leaf” nodes connected to their bus
    for _, r in loads.iterrows():
        bn = int(r["bus_number"])
        load_name = str(r.get("load_name", f"load-{bn}"))
        if f"bus:{bn}" in G:
            lnode = f"load:{load_name}"
            G.add_node(lnode, kind="load", bus_number=bn, name=load_name)
            G.add_edge(f"bus:{bn}", lnode, kind="load_tie")
    # Add transmission branches (bus-bus)
    for _, r in branches.iterrows():
        a = int(r["from_bus"])
        b = int(r["to_bus"])
        if f"bus:{a}" in G and f"bus:{b}" in G:
            G.add_edge(
                f"bus:{a}",
                f"bus:{b}",
                kind=str(r.get("branch_type", "Line")),
                name=str(r.get("branch_name", "")),
            )
    return G

# -----------------------------
# Layout: force-directed → normalized → mapped to WECC bbox
# -----------------------------

def compute_layout_positions(G: nx.Graph, bbox):
    """
    Returns dict: node -> (x, y, z)
    1) spring_layout on bus-only backbone
    2) normalize to [0,1]
    3) scale into WECC bbox (lon/lat extent)
    4) attach gens/loads near their parent bus with a small radial offset
    """
    xmin, xmax, ymin, ymax = bbox

    # Backbone: only buses for primary layout
    bus_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "bus"]
    H = G.subgraph(bus_nodes).copy()
    if len(H) == 0:
        raise ValueError("No bus nodes to layout.")

    pos2 = nx.spring_layout(H, dim=2, k=None, iterations=200, seed=42)
    # Normalize to 0..1
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

    # Scale into bbox
    def to_bbox(p01):
        x = xmin + p01[0] * (xmax - xmin)
        y = ymin + p01[1] * (ymax - ymin)
        return x, y

    pos3 = {}
    # Place buses on z=0 plane
    for n, p in pos2.items():
        p01 = norm(p)
        x, y = to_bbox(p01)
        pos3[n] = (x, y, 0.0)

    # Place gens/loads near their bus with small offsets
    for n, d in G.nodes(data=True):
        if d.get("kind") in ("gen", "load"):
            bn = d["bus_number"]
            parent = f"bus:{bn}"
            if parent in pos3:
                px, py, pz = pos3[parent]
                # Offset direction based on hash to fan around the bus
                h = abs(hash(n)) % 360
                ang = math.radians(h)
                r = 0.15 * (xmax - xmin) / 50.0  # small relative offset
                x = px + r * math.cos(ang)
                y = py + r * math.sin(ang)
                z = 0.02 * (ymax - ymin) / 50.0 if d["kind"] == "gen" else 0.0
                pos3[n] = (x, y, z)
            else:
                # fallback near origin if bus missing
                pos3[n] = (xmin, ymin, 0.0)

    return pos3

# -----------------------------
# Rendering
# -----------------------------

def render_network_pyvista(G: nx.Graph, pos: dict, bbox):
    plotter = pv.Plotter(window_size=(1100, 800))
    xmin, xmax, ymin, ymax = bbox

    # Draw WECC outline as a thin extruded frame (from bbox—simple frame for now)
    frame_z = -0.005 * (ymax - ymin)
    frame = pv.Cube(center=((xmin + xmax)/2, (ymin + ymax)/2, frame_z),
                    x_length=(xmax - xmin),
                    y_length=(ymax - ymin),
                    z_length=(ymax - ymin) * 0.0005)
    plotter.add_mesh(frame, opacity=0.05, show_edges=True)

    # Edges: lines between positions
    def add_edge(a, b, radius):
        axyz = np.array(pos[a])
        bxyz = np.array(pos[b])
        pts = np.vstack([axyz, bxyz])
        line = pv.Spline(pts, 2)
        tube = line.tube(radius=radius)
        plotter.add_mesh(tube, smooth_shading=True)

    # Choose radii by edge kind
    for u, v, ed in G.edges(data=True):
        kind = ed.get("kind", "Line")
        radius = 0.002 * (xmax - xmin) if kind == "Line" else 0.001 * (xmax - xmin)
        add_edge(u, v, radius)

    # Nodes: buses bigger; gens/loads smaller
    for n, d in G.nodes(data=True):
        x, y, z = pos[n]
        if d["kind"] == "bus":
            sph = pv.Sphere(radius=0.004 * (xmax - xmin), center=(x, y, z))
            plotter.add_mesh(sph, color="white")
        elif d["kind"] == "gen":
            sph = pv.Sphere(radius=0.003 * (xmax - xmin), center=(x, y, z))
            plotter.add_mesh(sph, color="green")
        else:  # load
            sph = pv.Sphere(radius=0.003 * (xmax - xmin), center=(x, y, z))
            plotter.add_mesh(sph, color="orange")

    plotter.add_axes()
    plotter.show_bounds(grid='front')
    plotter.show()

def render_network_matplotlib(G: nx.Graph, pos: dict):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 8))
    # Project to 2D
    pos2d = {k: (v[0], v[1]) for k, v in pos.items()}
    # Draw
    nx.draw_networkx_edges(G, pos2d, alpha=0.4, width=0.5)
    # Categorize nodes
    buses = [n for n, d in G.nodes(data=True) if d.get("kind") == "bus"]
    gens = [n for n, d in G.nodes(data=True) if d.get("kind") == "gen"]
    loads = [n for n, d in G.nodes(data=True) if d.get("kind") == "load"]
    nx.draw_networkx_nodes(G, pos2d, nodelist=buses, node_size=20, node_color="white", edgecolors="black")
    nx.draw_networkx_nodes(G, pos2d, nodelist=gens, node_size=12, node_color="green")
    nx.draw_networkx_nodes(G, pos2d, nodelist=loads, node_size=12, node_color="orange")
    ax.set_title("WECC POC (force-directed over WECC bbox)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()

# -----------------------------
# Main
# -----------------------------

def main():
    buses, branches, gens, loads, outline = load_tables()
    bbox = geojson_bounds(outline)

    G = build_graph(buses, branches, gens, loads)
    pos = compute_layout_positions(G, bbox)

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print("Sample nodes:", list(G.nodes())[:5])

    if _HAS_PYVISTA:
        render_network_pyvista(G, pos, bbox)
    else:
        print("PyVista not found; using a 2D fallback view.")
        render_network_matplotlib(G, pos)

if __name__ == "__main__":
    main()

