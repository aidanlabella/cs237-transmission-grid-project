#!/usr/bin/env python3
# wecc_3d_currents.py
#
# One-file WECC POC:
#   - builds graph from your CSVs
#   - force-directed layout normalized to WECC bbox (projected to EPSG:3857)
#   - optional GeoTIFF basemap plane under the scene (no contextily)
#   - interactive 3D render (PyVista) with:
#       * persistent bus labels (toggle with 'l')
#       * click-to-label bus names near the pick
#       * time-slider animation that colors branches by current
#   - 2D matplotlib fallback if PyVista isn't available
#
# Expected files (adjust DATA_DIR / FILES paths as needed):
#   bus_info.csv          [bus_number, bus_name, bus_area]
#   branch_info.csv       [from_bus, to_bus, branch_name, branch_type]
#   gen_info.csv          [bus_number, gen_name]
#   load_info.csv         [bus_number, load_name]
#   wecc_outline.geojson  (Polygon/MultiPolygon in lon/lat EPSG:4326)
#   wecc_satellite.tif    (optional raster basemap in EPSG:3857)
#   branch_currents.csv   (per-timestep branch currents, header names = branch_name)

import json
import math
import os
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

# 3D rendering
try:
    import pyvista as pv
    _HAS_PYVISTA = True
except Exception:
    _HAS_PYVISTA = False

# Raster + projection
import rasterio
from pyproj import Transformer

# Colormap for animation
from matplotlib import colormaps, colors

# --------- Configuration ----------
METADATA_DIR = Path('/Users/aidan/sandbox/cs237-transmission-grid-project/data/WECC_metadata')
SIMDATA_DIR = Path('/Users/aidan/sandbox/cs237-transmission-grid-project/data/WECC_sim_data/trip_branch_MESA CAL    -2408-MESA CAL    -2438-i_360')

FILES = {
    "bus": METADATA_DIR / "bus_info.csv",
    "branch": METADATA_DIR / "branch_info.csv",
    "gen": METADATA_DIR / "gen_info.csv",
    "load": METADATA_DIR / "load_info.csv",
    "outline": METADATA_DIR / "wecc_outline.geojson",
    "basemap": METADATA_DIR / "wecc_satellite.tif",    # optional, can be missing
    "branch_currents": SIMDATA_DIR / "branch_current_real.csv",  # per-timestep branch currents
    "gen_freq": SIMDATA_DIR / "gen_freqs.csv"
}

USE_BASEMAP = True
BASEMAP_Z_OFFSET = -1500.0  # meters below the graph

# --------- Helpers ----------

def normalize_name(s):
    """Collapse whitespace and strip; used to match generator names."""
    return " ".join(str(s).split())


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


def geojson_bounds_ll(geojson_obj):
    """Return (xmin, xmax, ymin, ymax) in lon/lat from Polygon/MultiPolygon GeoJSON."""
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
    """Project lon/lat bbox to EPSG:3857 and return (xmin, xmax, ymin, ymax) in meters."""
    xmin, xmax, ymin, ymax = bbox_ll
    to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
    x0, y0 = to_3857(xmin, ymin)
    x1, y1 = to_3857(xmax, ymax)
    xmin_m, xmax_m = min(x0, x1), max(x0, x1)
    ymin_m, ymax_m = min(y0, y1), max(y0, y1)
    return (xmin_m, xmax_m, ymin_m, ymax_m)


def build_graph(buses, branches, gens, loads):
    """
    Build a NetworkX graph:
      - bus nodes (kind='bus')
      - generator leaf nodes connected to bus (kind='gen')
      - load leaf nodes connected to bus (kind='load')
      - branches as edges between buses
    """
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
        gen_name_raw = str(r.get("gen_name", f"generator-{bn}")).strip()
        gen_name_norm = normalize_name(gen_name_raw)
        if f"bus:{bn}" in G:
            gnode = f"gen:{gen_name_norm}"
            G.add_node(
                gnode,
                kind="gen",
                bus_number=bn,
                name_raw=gen_name_raw,
                name_norm=gen_name_norm,
            )
            G.add_edge(f"bus:{bn}", gnode, kind="gen_tie")

    # loads
    for _, r in loads.iterrows():
        bn = int(r["bus_number"])
        load_name_raw = str(r.get("load_name", f"load-{bn}")).strip()
        if f"bus:{bn}" in G:
            lnode = f"load:{load_name_raw}"
            G.add_node(lnode, kind="load", bus_number=bn, name_raw=load_name_raw)
            G.add_edge(f"bus:{bn}", lnode, kind="load_tie")

    # branches (bus↔bus), static only here
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


def compute_layout_positions(G, bbox_3857):
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

    pos2 = nx.spring_layout(H, dim=2, iterations=200, seed=42)

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
    # buses on z=0
    for n, p in pos2.items():
        p01 = norm(p)
        x, y = to_bbox(p01)
        pos3[n] = (x, y, 0.0)

    # gens/loads near parent bus
    for n, d in G.nodes(data=True):
        if d.get("kind") in ("gen", "load"):
            bn = d["bus_number"]
            parent = f"bus:{bn}"
            if parent in pos3:
                px, py, pz = pos3[parent]
                h = abs(hash(n)) % 360
                ang = math.radians(h)
                r = 0.015 * (xmax - xmin)
                x = px + r * math.cos(ang)
                y = py + r * math.sin(ang)
                z = 50.0 if d["kind"] == "gen" else 0.0
                pos3[n] = (x, y, z)
            else:
                pos3[n] = (xmin, ymin, 0.0)

    return pos3


def build_basemap_plane_from_geotiff(tif_path, z_offset=-1500.0):
    """
    Reads a GeoTIFF basemap (in EPSG:3857) and returns (plane_mesh, texture, bounds).
    The plane covers the raster's bounds and is placed at z=z_offset.
    """
    if not tif_path.exists():
        raise FileNotFoundError(f"Basemap GeoTIFF not found: {tif_path}")

    with rasterio.open(tif_path) as src:
        left, bottom, right, top = src.bounds
        width = right - left
        height = top - bottom
        data = src.read()
        if data.dtype != np.uint8:
            data = data.astype(np.uint8)
        img = np.transpose(data, (1, 2, 0))

    plane = pv.Plane(
        center=((left + right) / 2, (bottom + top) / 2, z_offset),
        i_size=width,
        j_size=height,
        i_resolution=max(img.shape[1] - 1, 1),
        j_resolution=max(img.shape[0] - 1, 1),
    )
    plane.texture_map_to_plane(inplace=True)
    tex = pv.Texture(img)
    return plane, tex, (left, right, bottom, top)


def load_generator_frequency():
    """
    Load per-timestep generator frequency values from CSV.
    CSV header like you pasted:
        time,generator-3933-CG,generator-8034-H,...
    Returns:
        times: pandas.Series
        freq_df: DataFrame (T x Ngens) with normalized column names
    """
    path = FILES["gen_freq"]
    if not path.exists():
        print(f"[gen_freq] file not found: {path}")
        return None, None

    df = pd.read_csv(path)
    if "time" not in df.columns:
        raise ValueError("Generator frequency CSV must have a 'time' column.")

    times = df["time"]
    df = df.drop(columns=["time"])

    # Normalize columns to match gen_info names
    df = df.rename(columns={c: normalize_name(c) for c in df.columns})
    return times.reset_index(drop=True), df

# --------- Rendering (PyVista) ----------

def render_network_pyvista(G, pos, bbox_3857, use_basemap=True, freq_times=None, freq_df=None):
    """
    Render the WECC graph in 3D with PyVista, with generator frequency animation.
    """
    plotter = pv.Plotter(window_size=(1200, 850))
    xmin, xmax, ymin, ymax = bbox_3857

    # Basemap plane
    if use_basemap:
        try:
            plane, tex, _ = build_basemap_plane_from_geotiff(
                FILES["basemap"], z_offset=BASEMAP_Z_OFFSET
            )
            plotter.add_mesh(plane, texture=tex, smooth_shading=False)
        except Exception as e:
            print(f"[basemap] skipped: {e}")
            frame_z = -0.001 * (ymax - ymin)
            frame = pv.Cube(
                center=((xmin + xmax) / 2, (ymin + ymax) / 2, frame_z),
                x_length=(xmax - xmin),
                y_length=(ymax - ymin),
                z_length=(ymax - ymin) * 0.0005,
            )
            plotter.add_mesh(frame, opacity=0.05, show_edges=True)
    else:
        frame_z = -0.001 * (ymax - ymin)
        frame = pv.Cube(
            center=((xmin + xmax) / 2, (ymin + ymax) / 2, frame_z),
            x_length=(xmax - xmin),
            y_length=(ymax - ymin),
            z_length=(ymax - ymin) * 0.0005,
        )
        plotter.add_mesh(frame, opacity=0.05, show_edges=True)

    # Edges (static)
    for u, v, ed in G.edges(data=True):
        kind = ed.get("kind", "Line")
        radius = 0.0015 * (xmax - xmin) if kind == "Line" else 0.0009 * (xmax - xmin)
        axyz = np.array(pos[u])
        bxyz = np.array(pos[v])
        pts = np.vstack([axyz, bxyz])
        line = pv.Spline(pts, 2)
        tube = line.tube(radius=radius)
        color = "gray" if kind == "Line" else "lightgray"
        plotter.add_mesh(tube, color=color, smooth_shading=True)

    # Nodes and generator actors
    bus_pts, bus_labels = [], []
    gen_actors = {}  # name_norm -> {"sphere": actor, "label": actor}
    load_pts, load_labels = [], []

    base_bus_r = 0.003 * (xmax - xmin)
    base_gen_r = 0.0024 * (xmax - xmin)
    base_load_r = 0.0024 * (xmax - xmin)

    for n, d in G.nodes(data=True):
        x, y, z = pos[n]
        kind = d["kind"]
        if kind == "bus":
            sph = pv.Sphere(radius=base_bus_r, center=(x, y, z))
            plotter.add_mesh(sph, color="white")
            bus_pts.append([x, y, z])
            bus_labels.append(d.get("bus_name") or f"Bus {d['bus_number']}")
        elif kind == "gen":
            sph = pv.Sphere(radius=base_gen_r, center=(x, y, z))
            sph_actor = plotter.add_mesh(sph, color="green")

            name_raw = d.get("name_raw", "gen")
            name_norm = d.get("name_norm", name_raw)
            # label slightly above node
            z_off = 0.01 * (ymax - ymin)
            label_actor = plotter.add_point_labels(
                [[x, y, z + z_off]],
                [name_raw],
                point_size=0,
                font_size=11,
                text_color="green",
                shape=None,
                always_visible=True,
            )
            gen_actors[name_norm] = {"sphere": sph_actor, "label": label_actor}
        elif kind == "load":
            sph = pv.Sphere(radius=base_load_r, center=(x, y, z))
            plotter.add_mesh(sph, color="orange")
            load_pts.append([x, y, z])
            load_labels.append(d.get("name_raw", "load"))

    # ---------- BUS LABEL TOGGLE ----------
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

        plotter.add_key_event("l", lambda: toggle_labels())

    # ---------- CLICK-TO-LABEL BUS ----------
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

    # ---------- GENERATOR FREQUENCY ANIMATION ----------

    has_freq = (
        freq_times is not None
        and freq_df is not None
        and len(freq_times) > 0
        and not freq_df.empty
    )

    if has_freq:
        # Only keep columns that match generator nodes
        common_cols = [c for c in freq_df.columns if c in gen_actors]
        if not common_cols:
            print("[gen_freq] No generator freq columns matched generator nodes.")
            has_freq = False

    if has_freq:
        freq_vals = freq_df[common_cols]
        n_steps = len(freq_times)

        # global min/max
        vmin = float(freq_vals.min().min())
        vmax = float(freq_vals.max().max())
        if vmin == vmax:
            vmin -= 0.1
            vmax += 0.1

        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = colormaps["turbo"]

        state = {"idx": 0, "playing": False}

        # HUD for time and hints
        time_text_actor = plotter.add_text(
            f"time: {freq_times.iloc[0]} (index 0/{n_steps - 1})",
            font_size=10,
            position="upper_left",
        )
        plotter.add_text(
            "Slider + Play/Pause\n'n' to step forward\n'l' toggle bus labels\nClick to show a bus",
            font_size=10,
            position="lower_left",
        )

        def apply_index(idx):
            idx = int(max(0, min(n_steps - 1, idx)))
            state["idx"] = idx
            row = freq_vals.iloc[idx]

            for col in common_cols:
                val = row[col]
                rgba = cmap(norm(val))
                r, g, b = rgba[:3]
                actor_pair = gen_actors[col]
                # sphere color
                actor_pair["sphere"].GetProperty().SetColor(float(r), float(g), float(b))
                # label color (if possible)
                label_actor = actor_pair["label"]
                # In PyVista, label_actor is a vtkActor2D; we can try GetTextProperty()
                tp = getattr(label_actor, "GetTextProperty", None)
                if callable(tp):
                    tp().SetColor(float(r), float(g), float(b))

            time_text_actor.SetText(
                0,
                f"time: {freq_times.iloc[idx]} (index {idx}/{n_steps - 1})",
            )
            plotter.render()

        # init at 0
        apply_index(0)

        # slider
        def slider_cb(val):
            apply_index(int(val))

        plotter.add_slider_widget(
            slider_cb,
            rng=[0, n_steps - 1],
            value=0,
            title="Time index",
            pointa=(0.02, 0.08),
            pointb=(0.4, 0.08),
            style="modern",
        )

        # key: step forward
        def step_forward():
            new_idx = (state["idx"] + 1) % n_steps
            apply_index(new_idx)

        plotter.add_key_event("n", lambda: step_forward())

        # Play/Pause checkbox
        def play_pause_callback(checked):
            # checked=True => playing
            state["playing"] = bool(checked)

        try:
            plotter.add_checkbox_button_widget(
                play_pause_callback,
                value=False,
                position=(10, 70),
                size=25,
            )
            plotter.add_text(
                "Play/Pause",
                position=(40, 70),
                font_size=10,
            )
        except Exception as e:
            print(f"[gen_freq] checkbox not available: {e}")

        # Timed callback for 1 Hz playback (if PyVista supports it)
        if hasattr(plotter, "add_callback"):
            def tick():
                if not state["playing"]:
                    return
                step_forward()

            # interval in milliseconds; 1000 → ~1 Hz
            plotter.add_callback(tick, interval=1000)
        else:
            print(
                "[gen_freq] PyVista version has no timed callback; "
                "use slider or 'n' to step."
            )
    else:
        plotter.add_text(
            "No generator frequency data found.\nStatic view.",
            font_size=10,
            position="upper_left",
        )
        plotter.add_text(
            "Press 'l' to toggle bus labels • Click to show a bus",
            font_size=10,
            position="lower_left",
        )

    # Camera/UI
    plotter.add_axes()
    plotter.show_bounds(grid="front")
    plotter.show()

# --------- Fallback 2D ----------

def render_network_matplotlib(G, pos):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    pos2d = {k: (v[0], v[1]) for k, v in pos.items()}
    nx.draw_networkx_edges(G, pos2d, alpha=0.4, width=0.5)

    buses = [n for n, d in G.nodes(data=True) if d.get("kind") == "bus"]
    gens = [n for n, d in G.nodes(data=True) if d.get("kind") == "gen"]
    loads = [n for n, d in G.nodes(data=True) if d.get("kind") == "load"]

    nx.draw_networkx_nodes(
        G, pos2d, nodelist=buses, node_size=20, node_color="white", edgecolors="black"
    )
    nx.draw_networkx_nodes(G, pos2d, nodelist=gens, node_size=12, node_color="green")
    nx.draw_networkx_nodes(G, pos2d, nodelist=loads, node_size=12, node_color="orange")

    labels = {
        n: G.nodes[n].get("bus_name") or f"Bus {G.nodes[n]['bus_number']}"
        for n in buses
    }
    nx.draw_networkx_labels(G, pos2d, labels=labels, font_size=6)

    ax.set_title("WECC POC (EPSG:3857 layout) — 2D fallback")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()

# --------- Main ----------

def main():
    buses, branches, gens, loads, outline = load_tables()
    bbox_ll = geojson_bounds_ll(outline)
    bbox_3857 = project_bbox_to_3857(bbox_ll)

    G = build_graph(buses, branches, gens, loads)
    pos = compute_layout_positions(G, bbox_3857)

    freq_times, freq_df = load_generator_frequency()

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    if _HAS_PYVISTA:
        render_network_pyvista(
            G,
            pos,
            bbox_3857,
            use_basemap=USE_BASEMAP,
            freq_times=freq_times,
            freq_df=freq_df,
        )
    else:
        print("PyVista not found; using 2D fallback view.")
        render_network_matplotlib(G, pos)


if __name__ == "__main__":
    main()
