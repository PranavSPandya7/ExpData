"""
12_ucm_validate.py
==================
GPS photo bookend report with interactive Folium maps.
For each phase (BikeU, WalkU, BikeG, WalkG, Tram):
  - One combined GPS map with coloured tracks per participant
  - Bookend photos (start/middle/end) per participant-phase

Reads UCM photos from Complete Participantwise data.
Only participants present in key.csv are processed.
"""

import base64, io, re
from datetime import datetime, timedelta
from pathlib import Path

import folium
import numpy as np
import pandas as pd
from PIL import Image

# ── CONFIG ────────────────────────────────────────────────────────────────────
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import KEY_FILE, CPW_ROOT, OUTPUTS
KEY_CSV  = str(KEY_FILE)
OUT_HTML = str(OUTPUTS / '12_ucm_quality_report.html')

PARTICIPANTS = None   # populated from key.csv in main()
PHASES       = ["BikeU", "WalkU", "BikeG", "WalkG", "Tram"]

START_OFFSETS = [0, 10, 20]
END_OFFSETS   = [20, 10, 0]

THUMB_W        = 280
THUMB_QUALITY  = 60
MAP_HEIGHT     = 500
NEAREST_TOL_S  = 90
FALLBACK_TOL_S = 7200   # 2-hour fallback search across all phase folders

# Standard participant colour palette
PARTICIPANT_COLORS = {
    'P2':  '#e63946', 'P3':  '#2a9d8f', 'P4':  '#264653',
    'P5':  '#e76f51', 'P6':  '#457b9d', 'P7':  '#6a4c93',
    'P8':  '#1d3557', 'P9':  '#f4a261', 'P10': '#2d6a4f',
    'P11': '#c1121f', 'P12': '#780000', 'P13': '#3a86ff',
    'P14': '#8338ec', 'P15': '#fb5607', 'P16': '#005f73',
    'P17': '#606c38', 'P18': '#bc6c25',
}

MONTH_MAP = {
    "jan": 1, "fév": 2, "fev": 2, "feb": 2, "mar": 3, "avr": 4, "apr": 4,
    "mai": 5, "may": 5, "juin": 6, "jun": 6, "juil": 7, "jul": 7,
    "août": 8, "aout": 8, "aug": 8, "sep": 9, "oct": 10, "nov": 11,
    "déc": 12, "dec": 12,
}
YEAR   = 2025
CAM_RE = re.compile(r"^cam_(\d{8})_(\d{6})(?:_\d+)?\.jpg$", re.IGNORECASE)
# ─────────────────────────────────────────────────────────────────────────────


# ── LOW-LEVEL HELPERS ─────────────────────────────────────────────────────────

def parse_date(s: str) -> datetime:
    p = str(s).strip().split("-")
    return datetime(YEAR, MONTH_MAP[p[1].lower()], int(p[0]))


def ts_from_cam(name: str):
    m = CAM_RE.match(name)
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S") if m else None


def collect_cam_photos(cam_dir: Path):
    seen = {}
    for f in cam_dir.glob("cam_*.jpg"):
        ts = ts_from_cam(f.name)
        if ts is None:
            continue
        k = ts.strftime("%Y%m%d%H%M%S")
        if k not in seen:
            seen[k] = (ts, f)
    return sorted(seen.values())


def find_all_photos(pdir: Path):
    seen = {}
    for cam_dir in pdir.rglob("CamDown"):
        if cam_dir.is_dir():
            for ts, fp in collect_cam_photos(cam_dir):
                k = ts.strftime("%Y%m%d%H%M%S")
                if k not in seen:
                    seen[k] = (ts, fp)
    return sorted(seen.values())


def nearest_photo(photos, target: datetime, tol: int = NEAREST_TOL_S):
    if not photos:
        return None
    best = min(photos, key=lambda x: abs((x[0] - target).total_seconds()))
    return best if abs((best[0] - target).total_seconds()) <= tol else None


def classify_photo_source(fp: Path, phase: str) -> str:
    """Classify the folder that supplied a fallback photo."""
    parts = {part.lower() for part in fp.parts}
    phase_lower = phase.lower()
    if phase_lower in parts:
        return "same phase folder"
    for other_phase in PHASES:
        if other_phase.lower() in parts:
            return f"other phase folder ({other_phase})"
    if "inputdata" in parts:
        return "shared inputdata folder"
    return "other folder"


def nearest_gps_point(df, target: datetime):
    """Return (lat, lon) of the GPS row closest in time to target, or None."""
    if df is None or len(df) == 0:
        return None
    target_ts = pd.Timestamp(target)
    diffs = (df["GPS_time"] - target_ts).abs()
    idx = diffs.idxmin()
    return (df.loc[idx, "GPS_lat"], df.loc[idx, "GPS_lon"])


def flag_bad_gps_points(df):
    """
    Return a boolean mask (True = bad, should be excluded).
    Exclusion criteria (from 02_ucm_build.py):
    1. HDOP > 5              — poor satellite geometry
    2. IO_flag == 9          — receiver reports no fix
    Note: Speed and jump filters are omitted per user request.
    """
    n = len(df)
    bad = np.zeros(n, dtype=bool)

    # 1. HDOP
    if "GPS_HDOP" in df.columns:
        hdop = df["GPS_HDOP"].values
        bad |= (hdop > 5)

    # 2. IO_flag == 9
    if "IO_flag" in df.columns:
        io_bad = df["IO_flag"].values == 9
        bad |= io_bad

    # Propagate bad one row forward (step INTO bad point is unreliable)
    bad[1:] |= bad[:-1].copy()

    return bad


def kalman_smooth_gps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constant-velocity Kalman filter for GPS lat/lon.
    State: [lat, lon, vlat, vlon]
    Measurement noise R ~ 10 m -> ~9e-5 deg -> variance ~8e-9 deg²
    Process noise tuned for walking/cycling speeds.
    Returns a copy of df with smoothed GPS_lat / GPS_lon columns.
    """
    if len(df) < 4:
        return df

    times = df["GPS_time"].values          # numpy datetime64
    lats  = df["GPS_lat"].values.astype(float)
    lons  = df["GPS_lon"].values.astype(float)
    n     = len(lats)

    # seconds between consecutive rows
    dt_sec = np.diff(times.astype("datetime64[ms]").astype(np.float64)) / 1000.0
    dt_sec = np.clip(dt_sec, 0.05, 60.0)   # guard against gaps / duplicate stamps

    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], dtype=float)
    # GPS ~10 m noise in degrees  (10 / 111_320)² ≈ 8e-9
    R = np.eye(2) * 8e-9
    # acceleration noise – higher = follows raw GPS more closely
    # 0.3 m/s² typical cycling/walking acceleration
    SIGMA_A = 0.3 / 111_320        # convert m/s² → deg/s²

    x = np.array([lats[0], lons[0], 0.0, 0.0], dtype=float)
    P = np.diag([1e-8, 1e-8, 1e-6, 1e-6])

    out_lat = np.empty(n)
    out_lon = np.empty(n)
    out_lat[0] = lats[0]
    out_lon[0] = lons[0]

    for i in range(1, n):
        dt = dt_sec[i - 1]
        F  = np.array([[1, 0, dt,  0],
                       [0, 1,  0, dt],
                       [0, 0,  1,  0],
                       [0, 0,  0,  1]], dtype=float)
        # Piecewise-constant white-noise acceleration model
        sa2 = SIGMA_A ** 2
        Q   = sa2 * np.array([
            [dt**4 / 4,  0,          dt**3 / 2,  0         ],
            [0,          dt**4 / 4,  0,          dt**3 / 2 ],
            [dt**3 / 2,  0,          dt**2,      0         ],
            [0,          dt**3 / 2,  0,          dt**2     ],
        ])
        # ── Predict ────────────────────────────────────────────────────────
        x = F @ x
        P = F @ P @ F.T + Q
        # ── Update ─────────────────────────────────────────────────────────
        z   = np.array([lats[i], lons[i]])
        inn = z - H @ x
        S   = H @ P @ H.T + R
        K   = P @ H.T @ np.linalg.inv(S)
        x   = x + K @ inn
        P   = (np.eye(4) - K @ H) @ P
        out_lat[i] = x[0]
        out_lon[i] = x[1]

    df = df.copy()
    df["GPS_lat"] = out_lat
    df["GPS_lon"] = out_lon
    return df


def read_data_csv(csv_path: Path):
    """Read UCM data.csv using pandas — skip comment lines, find header with GPS_time."""
    if not csv_path or not csv_path.exists():
        return None
    try:
        # Find header line with GPS_time, then also skip the units line
        with open(csv_path, encoding="utf-8", errors="replace") as f:
            header_line = None
            for i, line in enumerate(f):
                if line.startswith("#") and "GPS_time" in line:
                    header_line = line
                    skip_until = i + 2  # skip header + units line
                    break
        if header_line is None:
            return None
        col_names = [c.strip() for c in header_line.lstrip("# ").split(",")]
        df = pd.read_csv(csv_path, skiprows=skip_until, header=None,
                         names=col_names, encoding="utf-8", engine="python",
                         na_values=["nan",""])

        df["GPS_time"] = pd.to_datetime(df["GPS_time"], errors="coerce")
        df["GPS_lat"]  = pd.to_numeric(df["GPS_lat"], errors="coerce")
        df["GPS_lon"]  = pd.to_numeric(df["GPS_lon"], errors="coerce")
        if "GPS_HDOP" in df.columns:
            df["GPS_HDOP"] = pd.to_numeric(df["GPS_HDOP"], errors="coerce")
        if "IO_flag" in df.columns:
            df["IO_flag"] = pd.to_numeric(df["IO_flag"], errors="coerce")
        df = df.dropna(subset=["GPS_time","GPS_lat","GPS_lon"]).sort_values("GPS_time").reset_index(drop=True)
        return df if len(df) > 0 else None
    except Exception:
        return None


def find_gps_for_phase(pdir: Path, phase: str, start_ts: datetime, end_ts: datetime):
    def _resolve_csv(ucm_base):
        dc = ucm_base / "data.csv"
        if dc.exists():
            return dc
        for sd in sorted(ucm_base.iterdir()):
            if sd.is_dir():
                dc2 = sd / "data.csv"
                if dc2.exists():
                    return dc2
        return None

    candidates = []
    for ucm_base in [pdir / phase / "ucm", pdir / "inputdata" / "ucm"]:
        if ucm_base.exists():
            c = _resolve_csv(ucm_base)
            if c:
                candidates.append(c)
    for dc in pdir.rglob("data.csv"):
        if dc not in candidates:
            candidates.append(dc)

    for csv_path in candidates:
        df = read_data_csv(csv_path)
        if df is None:
            continue
        mask = (df["GPS_time"] >= start_ts) & (df["GPS_time"] <= end_ts)
        filtered = df[mask].reset_index(drop=True)

        if len(filtered) > 0:
            bad_mask = flag_bad_gps_points(filtered)
            n_bad = bad_mask.sum()
            if n_bad > 0:
                filtered = filtered[~bad_mask].reset_index(drop=True)

        if len(filtered) >= 2:
            return kalman_smooth_gps(filtered)
    return None


def thumb_b64(fp: Path) -> str:
    img = Image.open(fp)
    w, h = img.size
    if w > THUMB_W:
        h = int(h * THUMB_W / w)
        img = img.resize((THUMB_W, h), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=THUMB_QUALITY)
    return base64.b64encode(buf.getvalue()).decode()


# ── MAP: combined all-participant view with toggle checkboxes ─────────────────

def make_combined_map_html(gps_per_pid: dict, height: int,
                           photo_times_per_pid: dict = None) -> str:
    """
    One folium map with a coloured FeatureGroup per participant.
    A floating checkbox panel lets the user show/hide individual tracks instantly.
    photo_times_per_pid: {pid: [(label, datetime), ...]} — 6 photo target times
      whose nearest GPS points are drawn as numbered circles on the track.
    """
    all_lats, all_lons = [], []
    for df in gps_per_pid.values():
        if df is not None and len(df) > 0:
            all_lats.extend(df["GPS_lat"].tolist())
            all_lons.extend(df["GPS_lon"].tolist())
    if not all_lats:
        return '<p style="color:#aaa;font-size:12px">No GPS data for this phase.</p>'

    center = [sum(all_lats) / len(all_lats), sum(all_lons) / len(all_lons)]
    m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")
    map_name = m.get_name()   # e.g. "map_a1b2c3d4"

    pid_varnames  = {}   # pid  → JS variable name of its FeatureGroup
    pid_colors_used = {}
    hover_tracks_js = []   # per-track JS var declarations (lat/lon/time arrays)
    hover_events_js = []   # (pl_name, pid_label, color) for mousemove binding

    for pid in sorted(gps_per_pid.keys()):
        df = gps_per_pid[pid]
        if df is None or len(df) < 2:
            continue
        color = PARTICIPANT_COLORS.get(f"P{pid}", "#888")
        pid_colors_used[pid] = color
        fg = folium.FeatureGroup(name=f"P{pid}", show=True)
        coords = list(zip(df["GPS_lat"], df["GPS_lon"]))
        pl = folium.PolyLine(coords, color=color, weight=3, opacity=0.85)
        pl.add_to(fg)
        pl_name = pl.get_name()
        # Build subsampled JS array for hover tooltip (max ~500 pts)
        _step = max(1, len(df) // 500)
        _pts = []
        for _, _r in df.iloc[::_step].iterrows():
            _tstr = pd.Timestamp(_r["GPS_time"]).strftime("%H:%M:%S")
            _pts.append(f'[{_r["GPS_lat"]:.6f},{_r["GPS_lon"]:.6f},"{_tstr}"]')
        hover_tracks_js.append(f'var _t_{pl_name}=[{",".join(_pts)}];')
        hover_events_js.append((pl_name, f"P{pid}", color))
        folium.CircleMarker(location=coords[0], radius=5, color=color,
                            fill=True, fill_color=color, fill_opacity=1.0,
                            tooltip=f"P{pid} start").add_to(fg)
        folium.CircleMarker(location=coords[-1], radius=5, color=color,
                            fill=True, fill_color="#fff", fill_opacity=1.0,
                            tooltip=f"P{pid} end").add_to(fg)

        # ── Numbered photo markers ─────────────────────────────────────────
        if photo_times_per_pid and pid in photo_times_per_pid:
            for num, (label, target_ts) in enumerate(photo_times_per_pid[pid], start=1):
                pos = nearest_gps_point(df, target_ts)
                if pos is None:
                    continue
                bg = "#27ae60" if num <= 3 else "#e74c3c"
                folium.Marker(
                    location=pos,
                    icon=folium.DivIcon(
                        html=(
                            f'<div style="background:{bg};color:#fff;'
                            f'border-radius:50%;width:20px;height:20px;'
                            f'display:flex;align-items:center;justify-content:center;'
                            f'font-size:11px;font-weight:700;border:2px solid #fff;'
                            f'box-shadow:0 1px 4px rgba(0,0,0,.55)">{num}</div>'
                        ),
                        icon_size=(20, 20),
                        icon_anchor=(10, 10),
                    ),
                    tooltip=f"P{pid} photo#{num} {label} @ {target_ts.strftime('%H:%M:%S')}",
                ).add_to(fg)

        fg.add_to(m)
        pid_varnames[pid] = fg.get_name()   # folium gives each a unique JS var name

    if not pid_varnames:
        return '<p style="color:#aaa;font-size:12px">No GPS data for this phase.</p>'

    # ── Floating checkbox panel HTML ──────────────────────────────────────────
    cb_rows = ""
    for pid in sorted(pid_varnames.keys()):
        vn    = pid_varnames[pid]
        color = pid_colors_used[pid]
        cb_rows += (
            f'<label style="display:flex;align-items:center;gap:7px;margin:3px 0;'
            f'cursor:pointer;user-select:none">'
            f'<input type="checkbox" class="track-cb" data-vn="{vn}" checked '
            f'style="cursor:pointer;width:13px;height:13px">'
            f'<span style="width:22px;height:4px;background:{color};border-radius:2px;'
            f'flex-shrink:0;display:inline-block"></span>'
            f'<span style="font-size:11px;color:#222">P{pid}</span>'
            f'</label>\n'
        )

    panel_html = f'''
<div id="track-panel" style="position:fixed;top:70px;right:10px;z-index:9999;
     background:rgba(255,255,255,0.96);padding:10px 14px;border-radius:8px;
     box-shadow:0 2px 10px rgba(0,0,0,0.28);min-width:115px;max-height:390px;
     overflow-y:auto;font-family:Arial,sans-serif">
  <b style="font-size:12px;display:block;margin-bottom:6px;color:#111">Participants</b>
  <label style="display:flex;align-items:center;gap:7px;margin-bottom:6px;
                cursor:pointer;user-select:none">
    <input type="checkbox" id="cb-all" checked
           style="cursor:pointer;width:13px;height:13px">
    <span style="font-size:11px;font-weight:700;color:#333">All / None</span>
  </label>
  <hr style="margin:4px 0;border:0;border-top:1px solid #e0e0e0">
  {cb_rows}
</div>'''

    # ── Toggle JS ─────────────────────────────────────────────────────────────
    # This script runs synchronously after all folium layer-creation scripts
    # (because it is add_child'd last), so all feature_group_xxx vars exist.
    # DOMContentLoaded fires after all synchronous scripts → safe to access them.
    reg_lines = "\n    ".join(
        f"tracks['{vn}'] = (typeof {vn} !== 'undefined') ? {vn} : null;"
        for vn in pid_varnames.values()
    )

    toggle_js = f'''<script>
(function () {{
  var mapObj, tracks = {{}};

  function init() {{
    mapObj = {map_name};
    {reg_lines}

    // Per-participant checkboxes
    document.querySelectorAll('.track-cb').forEach(function (cb) {{
      cb.addEventListener('change', function () {{
        var layer = tracks[this.getAttribute('data-vn')];
        if (!layer) return;
        if (this.checked) {{ layer.addTo(mapObj); }}
        else             {{ mapObj.removeLayer(layer); }}
        syncAllCheckbox();
      }});
    }});

    // "All / None" master checkbox
    document.getElementById('cb-all').addEventListener('change', function () {{
      var show = this.checked;
      this.indeterminate = false;
      document.querySelectorAll('.track-cb').forEach(function (cb) {{
        cb.checked = show;
        var layer = tracks[cb.getAttribute('data-vn')];
        if (!layer) return;
        if (show) {{ layer.addTo(mapObj); }}
        else      {{ mapObj.removeLayer(layer); }}
      }});
    }});
  }}

  function syncAllCheckbox() {{
    var cbs  = Array.from(document.querySelectorAll('.track-cb'));
    var nOn  = cbs.filter(function (c) {{ return c.checked; }}).length;
    var cbAll = document.getElementById('cb-all');
    cbAll.indeterminate = nOn > 0 && nOn < cbs.length;
    cbAll.checked       = nOn > 0;
  }}

  // document.readyState is 'loading' while inline scripts run;
  // DOMContentLoaded fires after all scripts have executed.
  if (document.readyState === 'loading') {{
    document.addEventListener('DOMContentLoaded', init);
  }} else {{
    init();
  }}
}})();
</script>'''

    m.get_root().html.add_child(folium.Element(panel_html))
    m.get_root().html.add_child(folium.Element(toggle_js))

    # ── Hover tooltip: participant ID + time follows mouse along each track ────
    if hover_events_js:
        track_decls = "\n  ".join(hover_tracks_js)
        ev_lines = []
        for _pl, _pid_lbl, _col in hover_events_js:
            ev_lines.append(
                f"    if(typeof {_pl}!=='undefined'){{\n"
                f"      {_pl}.on('mousemove',function(e){{\n"
                f"        var pt=_findNear(_t_{_pl},e.latlng);\n"
                f"        if(pt){{\n"
                f"          _hdiv.style.display='block';\n"
                f"          _hdiv.style.left=(e.originalEvent.clientX+14)+'px';\n"
                f"          _hdiv.style.top=(e.originalEvent.clientY-28)+'px';\n"
                f'          _hdiv.innerHTML=\'<b style="color:{_col}">{_pid_lbl}</b>&nbsp;&nbsp;\'+pt[2];\n'
                f"        }}\n"
                f"      }});\n"
                f"      {_pl}.on('mouseout',function(){{_hdiv.style.display='none';}});\n"
                f"    }}"
            )
        events_block = "\n".join(ev_lines)
        hover_js_parts = ["<script>\n(function(){", f"  {track_decls}"]
        hover_js_parts.append(
            "  var _hdiv=document.createElement('div');\n"
            "  _hdiv.style.cssText='position:fixed;background:rgba(20,20,20,0.82);color:#fff;'"
            "+'padding:5px 10px;border-radius:5px;font-size:12px;font-family:monospace;'"
            "+'pointer-events:none;z-index:99999;display:none;white-space:nowrap;'"
            "+'box-shadow:0 2px 6px rgba(0,0,0,.4)';\n"
            "  document.body.appendChild(_hdiv);\n"
            "  function _findNear(tr,ll){\n"
            "    var best=null,bd=Infinity;\n"
            "    for(var i=0;i<tr.length;i++){\n"
            "      var d=(tr[i][0]-ll.lat)*(tr[i][0]-ll.lat)+(tr[i][1]-ll.lng)*(tr[i][1]-ll.lng);\n"
            "      if(d<bd){bd=d;best=tr[i];}\n"
            "    }\n"
            "    return best;\n"
            "  }\n"
            "  function _attachHover(){"
        )
        hover_js_parts.append(events_block)
        hover_js_parts.append(
            "  }\n"
            "  if(document.readyState==='loading'){\n"
            "    document.addEventListener('DOMContentLoaded',_attachHover);\n"
            "  }else{_attachHover();}\n"
            "})();\n</script>"
        )
        m.get_root().html.add_child(folium.Element("\n".join(hover_js_parts)))

    b64 = base64.b64encode(m.get_root().render().encode("utf-8")).decode()
    return (f'<iframe src="data:text/html;base64,{b64}" width="100%" height="{height}px" '
            f'style="border:1px solid #ccc;border-radius:6px;" frameborder="0"></iframe>')


# ── PHOTO ROW: one participant, one phase ─────────────────────────────────────

def render_participant_row(pid: int, phase: str, start_ts: datetime,
                           end_ts: datetime, all_photos: list) -> str:
    color = PARTICIPANT_COLORS.get(f"P{pid}", "#888")

    targets = (
        [(f"S+{o}s", start_ts + timedelta(seconds=o)) for o in START_OFFSETS] +
        [(f"E-{o}s" if o > 0 else "E+0s", end_ts - timedelta(seconds=o)) for o in END_OFFSETS]
    )

    # Narrow time window for speed
    window_photos = [
        (ts, fp) for ts, fp in all_photos
        if (start_ts - timedelta(minutes=5)) <= ts <= (end_ts + timedelta(minutes=5))
    ]

    cards = []
    for i, (label, target) in enumerate(targets):
        group = "start-group" if i < 3 else "end-group"
        res = nearest_photo(window_photos, target)
        fallback_source = ""
        if res is None:
            # Fallback: search ALL photos for this participant (any phase folder)
            res = nearest_photo(all_photos, target, FALLBACK_TOL_S)
            if res is not None:
                fallback_source = classify_photo_source(res[1], phase)
        if res is None:
            cards.append(
                f'<div class="photo-item {group}">'
                f'<div class="photo-label-top">{label}</div>'
                f'<div class="no-photo">No photo<br>found</div>'
                f'<div class="photo-label">target {target.strftime("%H:%M:%S")}</div>'
                f'</div>'
            )
        else:
            ts_actual, fp = res
            diff = int((ts_actual - target).total_seconds())
            diff_str = f"({'+' if diff >= 0 else ''}{diff}s)" if diff != 0 else ""
            b64 = thumb_b64(fp)
            is_fallback = bool(fallback_source)
            if fallback_source == "same phase folder":
                fallback_note = (
                    f'<div style="font-size:9px;color:#c97c00;font-weight:700;margin-top:2px">'
                    f'\u26a0 same phase folder</div>'
                )
                item_style = ' style="border:3px solid #f1c40f;border-radius:4px"'
            elif fallback_source:
                fallback_note = (
                    f'<div style="font-size:9px;color:#e67e22;font-weight:700;margin-top:2px">'
                    f'\u26a0 {fallback_source}</div>'
                )
                item_style = ' style="border:3px solid #e67e22;border-radius:4px"'
            else:
                fallback_note = ""
                item_style = ""
            cards.append(
                f'<div class="photo-item {group}">'
                f'<div class="photo-label-top">{label}</div>'
                f'<img src="data:image/jpeg;base64,{b64}" alt="{label}" loading="lazy"{item_style}>'
                f'<div class="photo-label">{ts_actual.strftime("%H:%M:%S")} {diff_str}</div>'
                + fallback_note +
                f'</div>'
            )

    dur = round((end_ts - start_ts).total_seconds() / 60, 1)
    return (
        f'<div class="p-row">'
        f'<div class="p-label" style="border-left:5px solid {color};background:{color}18">'
        f'  <span class="p-name" style="color:{color}">P{pid}</span>'
        f'  <span class="p-time">{start_ts.strftime("%H:%M")}–{end_ts.strftime("%H:%M")}'
        f'  ({dur}m)</span>'
        f'</div>'
        f'<div class="photo-grid">{"".join(cards)}</div>'
        f'</div>'
    )


# ── KEY CSV ───────────────────────────────────────────────────────────────────

def read_key():
    df = pd.read_csv(KEY_CSV, sep=None, engine="python", encoding="utf-8-sig")
    result = {}
    for _, row in df.iterrows():
        try:
            pid = int(row["Participant_ID"])
        except (ValueError, KeyError):
            continue
        date = parse_date(str(row["Date"]))
        phases = {}
        for ph in PHASES:
            sc, ec = f"{ph}_start", f"{ph}_end"
            if sc not in row or ec not in row:
                continue
            s_raw, e_raw = str(row[sc]).strip(), str(row[ec]).strip()
            if s_raw in ("", "nan") or e_raw in ("", "nan"):
                continue
            try:
                s = datetime.combine(date, datetime.strptime(s_raw, "%H:%M:%S").time())
                e = datetime.combine(date, datetime.strptime(e_raw, "%H:%M:%S").time())
                phases[ph] = (s, e)
            except Exception:
                pass
        result[pid] = phases
    return result


# ── HTML ASSEMBLY ─────────────────────────────────────────────────────────────

def build_html(phase_sections: list) -> str:
    css = """
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', Arial, sans-serif; background: #e8eaed; padding: 20px; }
    h1 { text-align: center; color: #1a1a2e; margin-bottom: 28px; font-size: 22px; }
    .phase-section { background: #fff; border-radius: 10px; padding: 20px 24px;
                     margin: 0 auto 36px; box-shadow: 0 2px 10px rgba(0,0,0,.13);
                     max-width: 1600px; }
    .phase-title { font-size: 20px; font-weight: 700; color: #1a1a2e;
                   border-bottom: 3px solid #3388ff; padding-bottom: 8px;
                   margin-bottom: 16px; }
    .legend-row { font-size: 12px; color: #666; margin-bottom: 10px; }
    .p-row { display: flex; align-items: stretch; gap: 10px;
             border-bottom: 1px solid #f0f0f0; padding: 8px 0; }
    .p-row:last-child { border-bottom: none; }
    .p-label { width: 82px; min-width: 82px; display: flex; flex-direction: column;
               justify-content: center; padding: 4px 8px; border-radius: 4px; }
    .p-name  { font-size: 15px; font-weight: 700; }
    .p-time  { font-size: 10px; color: #666; margin-top: 3px; line-height: 1.3; }
    .photo-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 7px; flex: 1; }
    .photo-item { display: flex; flex-direction: column; align-items: center; }
    .photo-item img { width: 100%; border-radius: 4px; display: block; }
    .start-group img { border: 3px solid #27ae60; border-radius: 4px; }
    .end-group   img { border: 3px solid #e74c3c; border-radius: 4px; }
    .photo-label-top { font-size: 11px; font-weight: 700; color: #fff;
                       border-radius: 3px; padding: 1px 6px; margin-bottom: 3px; }
    .start-group .photo-label-top { background: #27ae60; }
    .end-group   .photo-label-top { background: #e74c3c; }
    .photo-label { text-align: center; font-size: 10px; color: #666; margin-top: 3px; }
    .no-photo { width: 100%; aspect-ratio: 4/3; background: #f7f7f7;
                display: flex; align-items: center; justify-content: center;
                color: #bbb; font-size: 11px; border-radius: 4px;
                border: 2px dashed #ddd; text-align: center; }
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Phase Bookend Report</title>
  <style>{css}</style>
</head>
<body>
  <h1>Phase Bookend Photos — First 3 &amp; Last 3 (10 s apart) · Grouped by Phase</h1>
  {"".join(phase_sections)}
</body>
</html>"""


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    global PARTICIPANTS

    print("Reading key.csv …")
    windows = read_key()
    print(f"  {len(windows)} participants loaded")

    # Derive participant list from key.csv — only these will be processed
    PARTICIPANTS = sorted(windows.keys())
    print(f"  Participants from key.csv: {PARTICIPANTS}")

    # Pre-collect all photos once per participant (expensive disk scan)
    print("\nCollecting UCM photos per participant …")
    all_photos_by_pid = {}
    pdirs = {}
    for pid in PARTICIPANTS:
        pdir = Path(CPW_ROOT) / f"P{pid}"
        pdirs[pid] = pdir
        if not pdir.exists():
            print(f"  P{pid}: folder not found under CPW_ROOT — skipping")
            all_photos_by_pid[pid] = []
            continue
        photos = find_all_photos(pdir)
        all_photos_by_pid[pid] = photos
        print(f"  P{pid}: {len(photos)} photos")

    # Build one section per phase
    phase_sections = []
    for phase in PHASES:
        print(f"\nPhase {phase} …")

        # GPS for combined map
        gps_per_pid = {}
        photo_times_per_pid = {}
        for pid in PARTICIPANTS:
            if pid not in windows or phase not in windows[pid]:
                continue
            start_ts, end_ts = windows[pid][phase]
            pdir = pdirs[pid]
            if not pdir.exists():
                continue
            gps_per_pid[pid] = find_gps_for_phase(pdir, phase, start_ts, end_ts)
            photo_times_per_pid[pid] = (
                [(f"S+{o}s", start_ts + timedelta(seconds=o)) for o in START_OFFSETS] +
                [(f"E-{o}s" if o > 0 else "E+0s", end_ts - timedelta(seconds=o))
                 for o in END_OFFSETS]
            )

        map_html = make_combined_map_html(gps_per_pid, MAP_HEIGHT, photo_times_per_pid)

        # One photo row per participant
        rows_html = []
        for pid in PARTICIPANTS:
            if pid not in windows or phase not in windows[pid]:
                continue
            if not pdirs[pid].exists():
                continue
            start_ts, end_ts = windows[pid][phase]
            row = render_participant_row(pid, phase, start_ts, end_ts,
                                         all_photos_by_pid[pid])
            rows_html.append(row)
            print(f"  P{pid} OK")

        legend = (
            '<p class="legend-row">'
            '<span style="color:#27ae60">■ green border = phase start photos</span>'
            ' &nbsp;&nbsp; '
            '<span style="color:#e74c3c">■ red border = phase end photos</span>'
            ' &nbsp;&nbsp; '
            '<span style="color:#f1c40f">■ amber border = same phase folder, outside local match window</span>'
            ' &nbsp;&nbsp; '
            '<span style="color:#e67e22">■ orange border = fallback from another folder</span>'
            '</p>'
        )
        section = (
            f'<div class="phase-section">'
            f'<h2 class="phase-title">{phase}</h2>'
            + legend + map_html
            + '<div style="margin-top:16px">' + "".join(rows_html) + '</div>'
            + '</div>'
        )
        phase_sections.append(section)

    print("\nBuilding HTML …")
    html = build_html(phase_sections)
    out = Path(OUT_HTML)
    out.write_text(html, encoding="utf-8")
    print(f"\nSaved: {out.resolve()}")
    print(f"Size:  {out.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()

