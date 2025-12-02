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
    "branch_current": SIMDATA_DIR / "branch_current_real.csv",  # per-timestep branch currents
    "gen_freq": SIMDATA_DIR / "gen_freqs.csv"
}

USE_BASEMAP = True
BASEMAP_Z_OFFSET = -1500.0  # meters below graph


def normalize_name(s: str) -> str:
    return " ".join(str(s).split())


# --------- Data loading / graph building ----------

def load_tables():
    buses = pd.read_csv(FILES["bus"])
    branches = pd.read_csv(FILES["branch"])
    gens = pd.read_csv(FILES["gen"])
    loads = pd.read_csv(FILES["load"])

    with open(FILES["outline"], "r") as f:
        outline = json.load(f)

    for req in ["bus_number", "bus_name", "bus_area"]:
        assert req in buses.columns, f"Missing '{req}' in bus_info.csv"
    for req in ["from_bus", "to_bus"]:
        assert req in branches.columns, f"Missing '{req}' in branch_info.csv"
    for req in ["bus_number"]:
        assert req in gens.columns, f"Missing '{req}' in gen_info.csv"
        assert req in loads.columns, f"Missing '{req}' in load_info.csv"

    return buses, branches, gens, loads, outline


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
    xmin_m, xmax_m = min(x0, x1), max(x0, x1)
    ymin_m, ymax_m = min(y0, y1), max(y0, y1)
    return (xmin_m, xmax_m, ymin_m, ymax_m)


def build_graph(buses, branches, gens, loads):
    G = nx.Graph()

    # Buses
    for _, r in buses.iterrows():
        bn = int(r["bus_number"])
        G.add_node(
            f"bus:{bn}",
            kind="bus",
            bus_number=bn,
            bus_name=str(r["bus_name"]).strip(),
            bus_area=int(r["bus_area"]),
        )

    # Generators (assume gen_name matches "generator-..." used in gen_freq CSV)
    for _, r in gens.iterrows():
        bn = int(r["bus_number"])
        gen_name_raw = str(r.get("gen_name", f"generator-{bn}")).strip()
        if f"bus:{bn}" in G:
            gnode = f"gen:{gen_name_raw}"
            G.add_node(
                gnode,
                kind="gen",
                bus_number=bn,
                gen_key=gen_name_raw,
                name_raw=gen_name_raw,
            )
            G.add_edge(f"bus:{bn}", gnode, kind="gen_tie")

    # Loads
    for _, r in loads.iterrows():
        bn = int(r["bus_number"])
        load_name_raw = str(r.get("load_name", f"load-{bn}")).strip()
        if f"bus:{bn}" in G:
            lnode = f"load:{load_name_raw}"
            G.add_node(
                lnode,
                kind="load",
                bus_number=bn,
                name_raw=load_name_raw,
            )
            G.add_edge(f"bus:{bn}", lnode, kind="load_tie")

    # Branches
    for _, r in branches.iterrows():
        a = int(r["from_bus"])
        b = int(r["to_bus"])
        if f"bus:{a}" not in G or f"bus:{b}" not in G:
            continue

        branch_label_raw = str(
            r.get("branch_name", "")
            or r.get("branch_id", "")
            or ""
        ).strip()
        name_norm = normalize_name(branch_label_raw) if branch_label_raw else ""

        G.add_edge(
            f"bus:{a}",
            f"bus:{b}",
            kind=str(r.get("branch_type", "Line")).strip(),
            name_raw=branch_label_raw,
            name_norm=name_norm,
        )

    return G


def compute_layout_positions(G, bbox_3857):
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
    for n, p in pos2.items():
        p01 = norm(p)
        x, y = to_bbox(p01)
        pos3[n] = (x, y, 0.0)

    # place gens/loads near their parent bus
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


def load_branch_currents():
    path = FILES["branch_current"]
    if not path.exists():
        print(f"[branch_current] file not found: {path}")
        return None, None

    df = pd.read_csv(path)
    if "time" not in df.columns:
        raise ValueError("branch_current.csv must have a 'time' column.")

    times = df["time"]
    df = df.drop(columns=["time"])
    df = df.rename(columns={c: normalize_name(c) for c in df.columns})
    return times.reset_index(drop=True), df


def load_gen_freq():
    path = FILES["gen_freq"]
    if not path.exists():
        print(f"[gen_freq] file not found: {path}")
        return None, None

    df = pd.read_csv(path)
    if "time" not in df.columns:
        raise ValueError("gen_freq.csv must have a 'time' column.")

    times = df["time"]
    df = df.drop(columns=["time"])
    # column names already like "generator-3933-CG"
    return times.reset_index(drop=True), df


def compute_contrasted_norm(df, contrast_zoom=0.4, low_percentile=5, high_percentile=95):
    arr = np.abs(df.to_numpy().ravel())
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return colors.Normalize(vmin=0.0, vmax=1.0)

    low_p = np.percentile(arr, low_percentile)
    high_p = np.percentile(arr, high_percentile)

    vmin = float(low_p)
    vmax = float(high_p)
    center = 0.5 * (vmin + vmax)
    half = 0.5 * (vmax - vmin) * contrast_zoom
    vmin = max(0.0, center - half)
    vmax = center + half
    if vmin >= vmax:
        vmax = vmin + 1.0

    return colors.Normalize(vmin=vmin, vmax=vmax)


# --------- Rendering (PyVista) ----------

def render_network_pyvista(
    G,
    pos,
    bbox_3857,
    use_basemap=True,
    branch_times=None,
    branch_df=None,
    gen_times=None,
    gen_df=None,
):
    plotter = pv.Plotter(window_size=(1200, 850))
    xmin, xmax, ymin, ymax = bbox_3857

    # Basemap / frame
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

    # ---------- Edges (branches) ----------
    branch_actors_by_name = {}

    for u, v, ed in G.edges(data=True):
        kind = ed.get("kind", "Line")
        axyz = np.array(pos[u])
        bxyz = np.array(pos[v])
        pts = np.vstack([axyz, bxyz])
        line = pv.Spline(pts, 2)

        if kind in ("gen_tie", "load_tie"):
            radius = 0.0009 * (xmax - xmin)
            base_color = "lightgray"
        else:
            radius = 0.0015 * (xmax - xmin)
            base_color = "gray"

        tube = line.tube(radius=radius)
        actor = plotter.add_mesh(tube, color=base_color, smooth_shading=True)

        name_norm = ed.get("name_norm", "")
        if name_norm and kind not in ("gen_tie", "load_tie"):
            branch_actors_by_name.setdefault(name_norm, []).append(actor)

    # ---------- Nodes (buses, gens, loads) ----------
    bus_pts, bus_labels = [], []
    load_pts, load_labels = [], []

    gen_actors_by_key = {}  # gen_key -> sphere actor

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
            actor = plotter.add_mesh(sph, color="green")
            gen_key = d.get("gen_key")
            if gen_key:
                gen_actors_by_key[gen_key] = actor

        elif kind == "load":
            sph = pv.Sphere(radius=base_load_r, center=(x, y, z))
            plotter.add_mesh(sph, color="orange")
            load_pts.append([x, y, z])
            load_labels.append(d.get("name_raw", "load"))

    # ---------- Bus label toggle ----------
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

    # ---------- Click-to-label bus ----------
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

    # ---------- Branch + Generator animation data ----------

    has_branch = (
        branch_times is not None
        and branch_df is not None
        and len(branch_times) > 0
        and not branch_df.empty
    )
    has_gen = (
        gen_times is not None
        and gen_df is not None
        and len(gen_times) > 0
        and not gen_df.empty
    )

    # Branch currents
    branch_cols, branch_use_df, branch_norm, branch_cmap = [], None, None, None
    if has_branch:
        available_cols = [c for c in branch_df.columns if c in branch_actors_by_name]
        if not available_cols:
            print("[branch_current] No columns matched branch names; disabling.")
            has_branch = False
        else:
            branch_cols = available_cols
            branch_use_df = branch_df[branch_cols]
            branch_norm = compute_contrasted_norm(branch_use_df, contrast_zoom=0.4)
            branch_cmap = colormaps.get_cmap("viridis")

    # Generator frequencies
    gen_cols, gen_use_df, gen_norm, gen_cmap = [], None, None, None
    if has_gen:
        available_cols = [c for c in gen_df.columns if c in gen_actors_by_key]
        if not available_cols:
            print("[gen_freq] No columns matched generator names; disabling.")
            has_gen = False
        else:
            gen_cols = available_cols
            gen_use_df = gen_df[gen_cols]
            gen_norm = compute_contrasted_norm(gen_use_df, contrast_zoom=0.4)
            gen_cmap = colormaps.get_cmap("plasma")

    if not has_branch and not has_gen:
        plotter.add_text(
            "No branch-current or generator-frequency data matched.\nStatic network view.",
            font_size=10,
            position="upper_left",
        )
        plotter.add_text(
            "Press 'l' to toggle bus labels • Click a bus to show its name",
            font_size=10,
            position="lower_left",
        )
        plotter.add_axes()
        plotter.show_bounds(grid="front")
        plotter.show()
        return

    # Shared timeline
    if has_branch and has_gen:
        n_steps = min(len(branch_times), len(gen_times))
        display_times = branch_times.iloc[:n_steps]
    elif has_branch:
        n_steps = len(branch_times)
        display_times = branch_times
    else:
        n_steps = len(gen_times)
        display_times = gen_times

    if has_branch:
        branch_use_df = branch_use_df.iloc[:n_steps].reset_index(drop=True)
    if has_gen:
        gen_use_df = gen_use_df.iloc[:n_steps].reset_index(drop=True)
    display_times = display_times.reset_index(drop=True)

    BRANCH_GAMMA = 0.6
    GEN_GAMMA = 0.6

    # Keep both time index and a handle to the current time-text actor
    state = {"idx": 0, "time_text_actor": None}

    # Static hint text
    plotter.add_text(
        "Slider: time index • 'n': next timestep • 'l': bus labels • click bus for name",
        font_size=10,
        position="lower_left",
    )

    # --------- Animation update function ----------

    def apply_index(idx: int):
        idx = int(max(0, min(n_steps - 1, idx)))
        state["idx"] = idx

        # Branch colors
        if has_branch:
            row_b = branch_use_df.iloc[idx]
            for col in branch_cols:
                val = row_b[col]
                if not np.isfinite(val):
                    continue
                mag = abs(float(val))
                t = branch_norm(mag)
                t = t ** BRANCH_GAMMA
                rgba = branch_cmap(t)
                r, g, b = rgba[:3]
                for actor in branch_actors_by_name.get(col, []):
                    actor.GetProperty().SetColor(float(r), float(g), float(b))

        # Generator colors
        if has_gen:
            row_g = gen_use_df.iloc[idx]
            for col in gen_cols:
                val = row_g[col]
                if not np.isfinite(val):
                    continue
                mag = abs(float(val))
                t = gen_norm(mag)
                t = t ** GEN_GAMMA
                rgba = gen_cmap(t)
                r, g, b = rgba[:3]

                actor = gen_actors_by_key.get(col)
                if actor is not None:
                    actor.GetProperty().SetColor(float(r), float(g), float(b))

        # time HUD — remove old actor, add new one
        if state["time_text_actor"] is not None:
            plotter.remove_actor(state["time_text_actor"])
        text = f"time: {display_times.iloc[idx]} (index {idx}/{n_steps - 1})"
        state["time_text_actor"] = plotter.add_text(
            text,
            font_size=10,
            position="upper_left",
        )

        plotter.render()

    # init frame
    apply_index(0)

    # --------- Slider + key controls ----------

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

    def step_forward():
        new_idx = (state["idx"] + 1) % n_steps
        apply_index(new_idx)

    plotter.add_key_event("n", lambda: step_forward())

    # --------- Colorbars (using dummy meshes) ----------

    # Branch current colorbar
    if has_branch and branch_norm is not None:
        vmin_b, vmax_b = float(branch_norm.vmin), float(branch_norm.vmax)
        dummy_b = pv.Cube(
            center=(xmin, ymin, BASEMAP_Z_OFFSET - 10000),
            x_length=1.0,
            y_length=1.0,
            z_length=1.0,
        )

        n_pts_b = dummy_b.n_points
        dummy_b["branch_vals"] = np.linspace(vmin_b, vmax_b, n_pts_b)

        plotter.add_mesh(
            dummy_b,
            scalars="branch_vals",
            cmap="viridis",
            clim=[vmin_b, vmax_b],
            opacity=0.0,
            show_scalar_bar=True,
            scalar_bar_args=dict(
                title="Branch |I| (scaled)",
                n_labels=5,
                italic=False,
                bold=True,
                vertical=True,
                position_x=0.86,
                position_y=0.15,
                width=0.12,
                height=0.7,
            ),
        )

    # Generator frequency colorbar
    if has_gen and gen_norm is not None:
        vmin_g, vmax_g = float(gen_norm.vmin), float(gen_norm.vmax)
        dummy_g = pv.Cube(
            center=(xmax, ymin, BASEMAP_Z_OFFSET - 10000),
            x_length=1.0,
            y_length=1.0,
            z_length=1.0,
        )

        n_pts_g = dummy_g.n_points
        dummy_g["gen_vals"] = np.linspace(vmin_g, vmax_g, n_pts_g)

        plotter.add_mesh(
            dummy_g,
            scalars="gen_vals",
            cmap="plasma",
            clim=[vmin_g, vmax_g],
            opacity=0.0,
            show_scalar_bar=True,
            scalar_bar_args=dict(
                title="Generator Frequency (Hz)",
                n_labels=5,
                italic=False,
                bold=True,
                vertical=True,
                position_x=0.02,
                position_y=0.15,
                width=0.12,
                height=0.7,
            ),
        )

    # --------- Show ----------

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

    branch_times, branch_df = load_branch_currents()
    gen_times, gen_df = load_gen_freq()

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    if _HAS_PYVISTA:
        render_network_pyvista(
            G,
            pos,
            bbox_3857,
            use_basemap=USE_BASEMAP,
            branch_times=branch_times,
            branch_df=branch_df,
            gen_times=gen_times,
            gen_df=gen_df,
        )
    else:
        print("PyVista not found; using 2D fallback view.")
        render_network_matplotlib(G, pos)


if __name__ == "__main__":
    main()
