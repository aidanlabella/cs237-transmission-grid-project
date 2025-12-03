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

try:
    import contextily as cx
    _HAS_CTX = True
except Exception:
    _HAS_CTX = False

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
    "gen_freq": SIMDATA_DIR / "gen_freqs_sin.csv"
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

    # Generators
    for _, r in gens.iterrows():
        bn = int(r["bus_number"])
        gen_name_raw = str(r.get("gen_name", f"generator-{bn}")).strip()
        if f"bus:{bn}" in G:
            gnode = f"gen:{gen_name_raw}"
            G.add_node(
                gnode,
                kind="gen",
                bus_number=bn,
                gen_key=gen_name_raw,   # should match "generator-..." columns
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


# --------- Layout from lat/lon ----------

def compute_layout_positions(G, buses):
    """
    Compute node positions from bus lat/lon, projected to EPSG:3857,
    then place generators/loads around their parent bus.

    Returns:
        pos: dict[node] -> (x, y, z)
        bbox_3857: (xmin, xmax, ymin, ymax)
    """

    # Try to auto-detect longitude/latitude columns in bus_info.csv
    def find_col(candidates):
        for c in buses.columns:
            lc = c.lower().replace(" ", "").replace("-", "").replace(".", "")
            for cand in candidates:
                if lc == cand:
                    return c
        return None

    lon_col = find_col(
        [
            "lon",
            "longitude",
            "buslon",
            "buslongitude",
            "x",
            "xcoord",
            "xcoordinate",
        ]
    )
    lat_col = find_col(
        [
            "lat",
            "latitude",
            "buslat",
            "buslatitude",
            "y",
            "ycoord",
            "ycoordinate",
        ]
    )

    def place_children(pos, bbox_3857):
        """Place generators/loads around their parent buses."""
        xmin, xmax, ymin, ymax = bbox_3857
        span = max(xmax - xmin, ymax - ymin, 1.0)
        gen_offset_r = 0.015 * span  # radius for gens
        load_offset_r = 0.015 * span # radius for loads

        for n, d in G.nodes(data=True):
            kind = d.get("kind")
            if kind not in ("gen", "load"):
                continue

            bn = d["bus_number"]
            parent = f"bus:{bn}"
            if parent not in pos:
                pos[n] = (xmin, ymin, 0.0)
                continue

            px, py, pz = pos[parent]
            h = abs(hash(n)) % 360
            ang = math.radians(h)
            r = gen_offset_r if kind == "gen" else load_offset_r
            x = px + r * math.cos(ang)
            y = py + r * math.sin(ang)
            z = 50.0 if kind == "gen" else 0.0
            pos[n] = (x, y, z)

        return pos

    # If lat/lon columns are missing, fall back to a spring layout and
    # normalize it into a synthetic bounding box so the rest of the pipeline
    # (basemap/frame sizing, offsets for gens/loads) still works.
    if lon_col is None or lat_col is None:
        print("[layout] no lat/lon columns found; using spring_layout fallback.")
        bus_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "bus"]
        if not bus_nodes:
            raise ValueError("Graph has no bus nodes to layout.")

        bus_subgraph = G.subgraph(bus_nodes)
        raw_pos = nx.spring_layout(bus_subgraph, dim=2, seed=42)

        xs = [p[0] for p in raw_pos.values()]
        ys = [p[1] for p in raw_pos.values()]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        span = max(xmax - xmin, ymax - ymin, 1e-6)
        scale = 1_000_000.0 / span  # scale to a reasonable "meters" span
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)

        pos = {
            n: ((p[0] - cx) * scale, (p[1] - cy) * scale, 0.0)
            for n, p in raw_pos.items()
        }

        xs_scaled = [p[0] for p in pos.values()]
        ys_scaled = [p[1] for p in pos.values()]
        bbox_3857 = (min(xs_scaled), max(xs_scaled), min(ys_scaled), max(ys_scaled))

        return place_children(pos, bbox_3857), bbox_3857

    # Project WGS84 (EPSG:4326) -> Web Mercator (EPSG:3857)
    to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform

    # First, place buses using their real coordinates
    pos = {}
    xs, ys = [], []

    for _, row in buses.iterrows():
        bn = int(row["bus_number"])
        if f"bus:{bn}" not in G:
            continue

        lon = float(row[lon_col])
        lat = float(row[lat_col])
        x, y = to_3857(lon, lat)
        pos[f"bus:{bn}"] = (x, y, 0.0)
        xs.append(x)
        ys.append(y)

    if not xs:
        raise ValueError("No bus positions were computed from lat/lon.")

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    bbox_3857 = (xmin, xmax, ymin, ymax)

    return place_children(pos, bbox_3857), bbox_3857


# --------- Basemap helpers ----------

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
    tex.flip_y = True  # VTK textures assume origin at lower-left; flip to keep north-up
    return plane, tex, (left, right, bottom, top)


def build_basemap_plane_from_contextily(bbox_3857, z_offset=-1500.0, zoom=6):
    """
    Use contextily.bounds2img to fetch satellite tiles for the WECC bbox
    in EPSG:3857 and map them to a PyVista plane.
    """
    xmin, xmax, ymin, ymax = bbox_3857
    if not _HAS_CTX:
        raise RuntimeError("contextily not available")

    img, ext = cx.bounds2img(
        xmin,
        ymin,
        xmax,
        ymax,
        zoom=zoom,
        source=cx.providers.Esri.WorldImagery,
        ll=False,  # our bounds are already in EPSG:3857
    )
    # ext is (left, bottom, right, top)
    left, bottom, right, top = ext
    width = right - left
    height = top - bottom

    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    plane = pv.Plane(
        center=((left + right) / 2, (bottom + top) / 2, z_offset),
        i_size=width,
        j_size=height,
        i_resolution=max(img.shape[1] - 1, 1),
        j_resolution=max(img.shape[0] - 1, 1),
    )
    plane.texture_map_to_plane(inplace=True)
    tex = pv.Texture(img)
    tex.flip_y = True  # keep map north-up when mapped onto the plane
    return plane, tex, (left, right, bottom, top)


# --------- Time‑varying data ----------

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
    """Percentile-based contrast normalization (used for branches)."""
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


def compute_minmax_norm(df):
    """Plain min/max normalization across all values (used for generator frequency)."""
    arr = df.to_numpy().ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return colors.Normalize(vmin=0.0, vmax=1.0)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmin == vmax:
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
        # Try contextily first (satellite tiles)
        if _HAS_CTX:
            try:
                plane, tex, _ = build_basemap_plane_from_contextily(
                    bbox_3857, z_offset=BASEMAP_Z_OFFSET, zoom=6
                )
                plotter.add_mesh(plane, texture=tex, smooth_shading=False)
                print("[basemap] using contextily (Esri WorldImagery)")
            except Exception as e:
                print(f"[basemap] contextily failed: {e}")
                # Fall back to GeoTIFF if present
                try:
                    plane, tex, _ = build_basemap_plane_from_geotiff(
                        FILES["basemap"], z_offset=BASEMAP_Z_OFFSET
                    )
                    plotter.add_mesh(plane, texture=tex, smooth_shading=False)
                    print("[basemap] using GeoTIFF fallback")
                except Exception as e2:
                    print(f"[basemap] GeoTIFF fallback failed: {e2}")
                    frame_z = -0.001 * (ymax - ymin)
                    frame = pv.Cube(
                        center=((xmin + xmax) / 2, (ymin + ymax) / 2, frame_z),
                        x_length=(xmax - xmin),
                        y_length=(ymax - ymin),
                        z_length=(ymax - ymin) * 0.0005,
                    )
                    plotter.add_mesh(frame, opacity=0.05, show_edges=True)
        else:
            # No contextily: try GeoTIFF, then simple frame
            try:
                plane, tex, _ = build_basemap_plane_from_geotiff(
                    FILES["basemap"], z_offset=BASEMAP_Z_OFFSET
                )
                plotter.add_mesh(plane, texture=tex, smooth_shading=False)
                print("[basemap] using GeoTIFF (no contextily)")
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
    branch_label_positions = []
    branch_label_texts = []

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

            # add branch label at midpoint, slightly above line
            midpoint = (axyz + bxyz) / 2.0
            label_pos = midpoint.copy()
            label_pos[2] += 0.01 * (ymax - ymin)
            branch_label_positions.append(label_pos.tolist())
            branch_label_texts.append(name_norm)

    # Branch labels (black text)
    if branch_label_positions:
        plotter.add_point_labels(
            branch_label_positions,
            branch_label_texts,
            point_size=0,
            font_size=10,
            text_color="black",
            shape=None,
            always_visible=True,
        )

    # ---------- Nodes (buses, gens, loads) ----------
    gen_actors_by_key = {}
    gen_label_positions = []
    gen_label_texts = []

    base_bus_r = 0.003 * (xmax - xmin)
    base_gen_r = 0.0024 * (xmax - xmin)
    base_load_r = 0.0024 * (xmax - xmin)
    gen_cyl_height = 3.0 * base_gen_r

    for n, d in G.nodes(data=True):
        x, y, z = pos[n]
        kind = d["kind"]

        if kind == "bus":
            sph = pv.Sphere(radius=base_bus_r, center=(x, y, z))
            plotter.add_mesh(sph, color="white")

        elif kind == "gen":
            cyl_center = (x, y, z + gen_cyl_height / 2.0)
            cylinder = pv.Cylinder(
                center=cyl_center,
                direction=(0, 0, 1),
                radius=base_gen_r,
                height=gen_cyl_height,
                resolution=24,
            )
            actor = plotter.add_mesh(cylinder, color="green")
            gen_key = d.get("gen_key")
            if gen_key:
                gen_actors_by_key[gen_key] = actor
                # label above the top of the cylinder
                label_z = z + gen_cyl_height * 0.9
                gen_label_positions.append([x, y, label_z])
                gen_label_texts.append(str(gen_key))

        elif kind == "load":
            sph = pv.Sphere(radius=base_load_r, center=(x, y, z))
            plotter.add_mesh(sph, color="orange")

    # Generator labels: green text, no background shape
    if gen_label_positions:
        plotter.add_point_labels(
            gen_label_positions,
            gen_label_texts,
            point_size=0,
            font_size=18,
            text_color="green",
            always_visible=True,
        )

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
            # global min/max across all generators & time
            gen_norm = compute_minmax_norm(gen_df[gen_cols])
            gen_cmap = colormaps.get_cmap("plasma")

    if not has_branch and not has_gen:
        plotter.add_text(
            "No branch-current or generator-frequency data matched.\nStatic network view.",
            font_size=10,
            position="upper_left",
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

    state = {"idx": 0, "time_text_actor": None}

    # Static hint text
    plotter.add_text(
        "Slider: time index • 'n': next timestep",
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
                t = float(np.clip(t, 0.0, 1.0))
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
                mag = float(val)
                t = gen_norm(mag)
                t = float(np.clip(t, 0.0, 1.0))
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
                title="Branch |I|",
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

    # Generator frequency colorbar — true global min/max over all generators/times
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

    ax.set_title("WECC POC (EPSG:3857 layout) — 2D fallback")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()


# --------- Main ----------

def main():
    buses, branches, gens, loads, outline = load_tables()

    G = build_graph(buses, branches, gens, loads)

    # Use real lat/lon positions for buses (projected to EPSG:3857)
    pos, bbox_3857 = compute_layout_positions(G, buses)

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
