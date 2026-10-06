"""
12_ucm_validate.py
==================
GPS photo bookend report with interactive Folium maps.
For each requested phase (WalkG, BikeG):
  - One combined GPS map with coloured tracks per participant
  - Bookend photos (start/middle/end) per participant-phase

Participant IDs and phase windows come from key.csv. UCM GPS tracks come from
the built 10-sec UCM output; photo/bookend files come from Final Data UCM.
"""

import base64, html, io, re
from datetime import datetime, timedelta
from pathlib import Path

import folium
import numpy as np
import pandas as pd
from PIL import Image

# -- CONFIG --------------------------------------------------------------------
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import FINAL_DATA, KEY_FILE as KEY_CSV, OUTPUTS, load_key_unique
UCM_10SEC_CSV = OUTPUTS / "02_ucm_10sec.csv"
UCM_SOURCE_ROOTS = [
    ("Final Data UCM", FINAL_DATA / "ucm"),
]
OUT_HTML = str(OUTPUTS / '12_ucm_quality_report.html')

PARTICIPANTS = None   # populated from key.csv and source roots in main()
DEFAULT_UNCHECKED_PIDS = {1, 2, 3, 5, 6, 7, 18}
PHASES       = ["BikeU", "WalkU", "BikeG", "WalkG", "Tram"]
REPORT_PHASES = ["BikeU", "WalkU", "BikeG", "WalkG", "Tram"]
EXCLUDED_NOTE_PHASES = set()
EXCLUDED_NOTE_TEXT = ""

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
# -----------------------------------------------------------------------------


# -- LOW-LEVEL HELPERS ---------------------------------------------------------

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
    ucm_roots = [pdir / phase / "ucm" for phase in PHASES]
    ucm_roots.append(pdir / "inputdata" / "ucm")
    for ucm_root in ucm_roots:
        if not ucm_root.exists():
            continue
        for cam_dir in ucm_root.rglob("CamDown"):
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


def discover_participants(root: Path):
    participants = []
    if not root.exists():
        return participants
    for pdir in sorted(root.iterdir()):
        if not pdir.is_dir():
            continue
        m = re.fullmatch(r"P(\d+)", pdir.name, re.IGNORECASE)
        if m:
            participants.append(int(m.group(1)))
    return participants


def read_built_ucm_gps():
    if not UCM_10SEC_CSV.exists():
        raise FileNotFoundError(f"Built UCM 10-sec file not found: {UCM_10SEC_CSV}")
    df = pd.read_csv(UCM_10SEC_CSV)
    required = {"ParticipantID", "PhaseID", "Datetime", "GPS_lat", "GPS_lon"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{UCM_10SEC_CSV} is missing required columns: {sorted(missing)}")
    df = df.copy()
    df["GPS_time"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df["GPS_lat"] = pd.to_numeric(df["GPS_lat"], errors="coerce")
    df["GPS_lon"] = pd.to_numeric(df["GPS_lon"], errors="coerce")
    df["ParticipantNum"] = pd.to_numeric(
        df["ParticipantID"].astype(str).str.extract(r"(\d+)", expand=False),
        errors="coerce",
    )
    df = df.dropna(subset=["ParticipantNum", "PhaseID", "GPS_time", "GPS_lat", "GPS_lon"])
    df["ParticipantNum"] = df["ParticipantNum"].astype(int)
    df["PhaseID"] = df["PhaseID"].astype(str)
    return df.sort_values(["ParticipantNum", "PhaseID", "GPS_time"]).reset_index(drop=True)


def find_built_gps_for_phase(ucm_df: pd.DataFrame, pid: int, phase: str,
                             start_ts: datetime, end_ts: datetime):
    if ucm_df is None or len(ucm_df) == 0:
        return None
    mask = (ucm_df["ParticipantNum"] == int(pid)) & (ucm_df["PhaseID"] == phase)
    out = ucm_df.loc[mask, ["GPS_time", "GPS_lat", "GPS_lon"]].copy()
    if len(out) == 0:
        return None
    out = out.sort_values("GPS_time").reset_index(drop=True)
    # Display-only retime: built UCM GPS can be on a +2h clock while photos/key
    # are local experiment clock. Keep the cleaned route shape, align markers to key.
    if len(out) >= 2:
        rel = np.linspace(0, 1, len(out))
        span = (pd.Timestamp(end_ts) - pd.Timestamp(start_ts)).total_seconds()
        out["GPS_time"] = pd.Timestamp(start_ts) + pd.to_timedelta(rel * span, unit="s")
    out.attrs["source_csv"] = str(UCM_10SEC_CSV)
    return out if len(out) >= 2 else None


def read_data_csv(csv_path: Path):
    """Read one raw UCM data.csv using its '# GPS_time,...' header."""
    if not csv_path or not csv_path.exists():
        return None
    try:
        col_names = None
        with csv_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                stripped = line.lstrip("# ").strip()
                if stripped.startswith("GPS_time"):
                    col_names = [c.strip() for c in stripped.split(",")]
                    break
        if col_names is None:
            return None
        df = pd.read_csv(csv_path, comment="#", header=None, names=col_names, low_memory=False)
        df["GPS_time"] = pd.to_datetime(df["GPS_time"], errors="coerce")
        df["GPS_lat"] = pd.to_numeric(df.get("GPS_lat"), errors="coerce")
        df["GPS_lon"] = pd.to_numeric(df.get("GPS_lon"), errors="coerce")
        if "GPS_HDOP" in df.columns:
            df["GPS_HDOP"] = pd.to_numeric(df["GPS_HDOP"], errors="coerce")
        if "IO_flag" in df.columns:
            df["IO_flag"] = pd.to_numeric(df["IO_flag"], errors="coerce")
        df = df.dropna(subset=["GPS_time", "GPS_lat", "GPS_lon"])
        return df.sort_values("GPS_time").reset_index(drop=True) if len(df) else None
    except Exception:
        return None


def flag_bad_gps_points(df):
    """Return True for GPS rows that should be excluded from map tracks."""
    n = len(df)
    bad = np.zeros(n, dtype=bool)

    if "GPS_HDOP" in df.columns:
        bad |= pd.to_numeric(df["GPS_HDOP"], errors="coerce").to_numpy() > 5
    if "IO_flag" in df.columns:
        bad |= pd.to_numeric(df["IO_flag"], errors="coerce").to_numpy() == 9

    # Keep the step into a bad point out of the drawn route as well.
    bad[1:] |= bad[:-1].copy()
    return bad


def kalman_smooth_gps(df: pd.DataFrame) -> pd.DataFrame:
    """Constant-velocity Kalman smoothing for displayed GPS map tracks."""
    if len(df) < 4:
        return df

    times = df["GPS_time"].values
    lats = df["GPS_lat"].values.astype(float)
    lons = df["GPS_lon"].values.astype(float)
    n = len(lats)

    dt_sec = np.diff(times.astype("datetime64[ms]").astype(np.float64)) / 1000.0
    dt_sec = np.clip(dt_sec, 0.05, 60.0)

    h = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    r = np.eye(2) * 8e-9
    sigma_a = 0.3 / 111_320

    x = np.array([lats[0], lons[0], 0.0, 0.0], dtype=float)
    p = np.diag([1e-8, 1e-8, 1e-6, 1e-6])
    out_lat = np.empty(n)
    out_lon = np.empty(n)
    out_lat[0] = lats[0]
    out_lon[0] = lons[0]

    for i in range(1, n):
        dt = dt_sec[i - 1]
        f = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        sa2 = sigma_a ** 2
        q = sa2 * np.array([
            [dt**4 / 4, 0, dt**3 / 2, 0],
            [0, dt**4 / 4, 0, dt**3 / 2],
            [dt**3 / 2, 0, dt**2, 0],
            [0, dt**3 / 2, 0, dt**2],
        ])
        x = f @ x
        p = f @ p @ f.T + q
        z = np.array([lats[i], lons[i]])
        inn = z - h @ x
        s = h @ p @ h.T + r
        k = p @ h.T @ np.linalg.inv(s)
        x = x + k @ inn
        p = (np.eye(4) - k @ h) @ p
        out_lat[i] = x[0]
        out_lon[i] = x[1]

    out = df.copy()
    out["GPS_lat"] = out_lat
    out["GPS_lon"] = out_lon
    return out


def find_gps_for_phase(pdir: Path, phase: str, start_ts: datetime, end_ts: datetime):
    """Return raw Final Data UCM GPS cropped to the key.csv phase window."""
    candidates = []
    for base in [pdir / phase / "ucm", pdir / "inputdata" / "ucm"]:
        if not base.exists():
            continue
        direct = base / "data.csv"
        if direct.exists():
            candidates.append(direct)
        candidates.extend(sorted(base.rglob("data.csv")))

    best = None
    best_n = 0
    for csv_path in dict.fromkeys(candidates):
        df = read_data_csv(csv_path)
        if df is None:
            continue
        mask = (df["GPS_time"] >= start_ts) & (df["GPS_time"] <= end_ts)
        filtered = df.loc[mask].copy()
        if filtered.empty:
            continue
        bad_mask = flag_bad_gps_points(filtered.reset_index(drop=True))
        if bad_mask.any():
            filtered = filtered.reset_index(drop=True).loc[~bad_mask].copy()
        filtered = filtered[["GPS_time", "GPS_lat", "GPS_lon"]].dropna()
        if len(filtered) > best_n:
            best = kalman_smooth_gps(filtered.sort_values("GPS_time").reset_index(drop=True))
            best.attrs["source_csv"] = str(csv_path)
            best_n = len(best)
    return best if best_n >= 2 else None


def nearest_gps_point(df, target: datetime):
    """Return (lat, lon) of the GPS row closest in time to target, or None."""
    if df is None or len(df) == 0:
        return None
    target_ts = pd.Timestamp(target)
    diffs = (df["GPS_time"] - target_ts).abs()
    idx = diffs.idxmin()
    return (df.loc[idx, "GPS_lat"], df.loc[idx, "GPS_lon"])



def thumb_b64(fp: Path) -> str:
    try:
        with fp.open('rb') as fh:
            img = Image.open(fh)
            img.load()
        w, h = img.size
        if w > THUMB_W:
            h = int(h * THUMB_W / w)
            img = img.resize((THUMB_W, h), Image.LANCZOS)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=THUMB_QUALITY)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ''


# -- MAP: combined all-participant view with toggle checkboxes -----------------

def make_combined_map_html(gps_per_pid: dict, height: int, phase: str, source_key: str,
                           photo_times_per_pid: dict = None) -> str:
    """
    One folium map with a coloured FeatureGroup per participant.
    A floating checkbox panel lets the user show/hide individual tracks instantly.
    photo_times_per_pid: {pid: [(label, datetime), ...]} - 6 photo target times
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
        show_track = pid not in DEFAULT_UNCHECKED_PIDS
        fg = folium.FeatureGroup(name=f"P{pid}", show=show_track)
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

        # -- Numbered photo markers -----------------------------------------
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

    # -- Floating checkbox panel HTML ------------------------------------------
    cb_rows = ""
    for pid in sorted(pid_varnames.keys()):
        vn    = pid_varnames[pid]
        color = pid_colors_used[pid]
        checked_attr = " checked" if pid not in DEFAULT_UNCHECKED_PIDS else ""
        cb_rows += (
            f'<label style="display:flex;align-items:center;gap:7px;margin:3px 0;'
            f'cursor:pointer;user-select:none">'
            f'<input type="checkbox" class="track-cb" data-vn="{vn}" data-pid="P{pid}"{checked_attr} '
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

    # -- Toggle JS -------------------------------------------------------------
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

  function setPhotoRows(pid, show) {{
    try {{
      parent.document
        .querySelectorAll('.p-row[data-source="{source_key}"][data-phase="{phase}"][data-pid="' + pid + '"]')
        .forEach(function (row) {{
          row.style.display = show ? 'flex' : 'none';
        }});
    }} catch (err) {{}}
  }}

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
        setPhotoRows(this.getAttribute('data-pid'), this.checked);
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
        setPhotoRows(cb.getAttribute('data-pid'), show);
      }});
    }});

    syncAllCheckbox();
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

    # -- Hover tooltip: participant ID + time follows mouse along each track ----
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

    srcdoc = html.escape(m.get_root().render(), quote=True)
    return (f'<iframe srcdoc="{srcdoc}" width="100%" height="{height}px" '
            f'style="border:1px solid #ccc;border-radius:6px;" frameborder="0"></iframe>')


# -- PHOTO ROW: one participant, one phase -------------------------------------

def render_participant_row(pid: int, phase: str, source_key: str, start_ts: datetime,
                           end_ts: datetime, all_photos: list) -> str:
    color = PARTICIPANT_COLORS.get(f"P{pid}", "#888")
    row_style = ' style="display:none"' if pid in DEFAULT_UNCHECKED_PIDS else ''

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
        if res is None:
            # Fallback: search ALL photos for this participant (any phase folder)
            res = nearest_photo(all_photos, target, FALLBACK_TOL_S)
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
            cards.append(
                f'<div class="photo-item {group}">'
                f'<div class="photo-label-top">{label}</div>'
                f'<img src="data:image/jpeg;base64,{b64}" alt="{label}" loading="lazy">'
                f'<div class="photo-label">{ts_actual.strftime("%H:%M:%S")} {diff_str}</div>'
                f'</div>'
            )

    dur = round((end_ts - start_ts).total_seconds() / 60, 1)
    return (
        f'<div class="p-row" data-source="{source_key}" data-phase="{phase}" data-pid="P{pid}"{row_style}>'
        f'<div class="p-label" style="border-left:5px solid {color};background:{color}18">'
        f'  <span class="p-name" style="color:{color}">P{pid}</span>'
        f'  <span class="p-time">{start_ts.strftime("%H:%M")}–{end_ts.strftime("%H:%M")}'
        f'  ({dur}m)</span>'
        f'</div>'
        f'<div class="photo-grid">{"".join(cards)}</div>'
        f'</div>'
    )


# -- KEY CSV -------------------------------------------------------------------

def read_key():
    df = load_key_unique(KEY_CSV)
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


# -- HTML ASSEMBLY -------------------------------------------------------------

def build_html(phase_sections: list) -> str:
    css = """
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', Arial, sans-serif; background: #e8eaed; padding: 20px; }
    h1 { text-align: center; color: #1a1a2e; margin-bottom: 28px; font-size: 22px; }
    .source-section { max-width: 1640px; margin: 0 auto 44px; }
    .source-title { background: #1a1a2e; color: #fff; padding: 14px 18px;
                    border-radius: 8px 8px 0 0; font-size: 19px; font-weight: 700; }
    .source-path { background: #fff; color: #111; border-left: 4px solid #3388ff;
                   padding: 10px 14px; font-size: 12px; overflow-wrap: anywhere;
                   box-shadow: 0 2px 8px rgba(0,0,0,.10); margin-bottom: 18px; }
    .method-note { max-width: 1640px; margin: 0 auto 18px; background: #fff;
                   border-left: 4px solid #27ae60; padding: 12px 16px;
                   font-size: 13px; line-height: 1.45; color: #111;
                   box-shadow: 0 2px 8px rgba(0,0,0,.10); }
    .phase-section { background: #fff; border-radius: 10px; padding: 20px 24px;
                     margin: 0 auto 36px; box-shadow: 0 2px 10px rgba(0,0,0,.13);
                     max-width: 1600px; }
    .phase-title { font-size: 20px; font-weight: 700; color: #1a1a2e;
                   border-bottom: 3px solid #3388ff; padding-bottom: 8px;
                   margin-bottom: 16px; }
    .legend-row { font-size: 12px; color: #666; margin-bottom: 10px; }
    .excluded-note { font-size: 13px; color: #c0392b; font-weight: 700; margin: -4px 0 10px; }
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
  <h1>UCM Phase Bookend Validation - First 3 &amp; Last 3 Photos - Grouped by Source Folder</h1>
  <div class="method-note">
    <b>Method note:</b> GPS tracks are read from the built UCM output
    <code>{html.escape(str(UCM_10SEC_CSV))}</code>. Photo/bookend rows are read from the Final Data UCM source folder printed below,
    using <code>{html.escape(str(KEY_CSV))}</code> phase windows.
  </div>
  {"".join(phase_sections)}
</body>
</html>"""


# -- MAIN ----------------------------------------------------------------------

def main():
    global PARTICIPANTS

    key_data = read_key()

    observed_participants = set()
    for _, source_root in UCM_SOURCE_ROOTS:
        print(f"Scanning participant folders in {source_root} …")
        observed_participants |= set(discover_participants(source_root))

    PARTICIPANTS = sorted(set(key_data.keys()) | observed_participants)
    print(f"  Key participants: {PARTICIPANTS}")

    if not PARTICIPANTS:
        print("  No key participants found; nothing to render.")
        return

    built_ucm_gps = read_built_ucm_gps()
    print(f"  Built UCM GPS rows with coordinates: {len(built_ucm_gps):,}")

    source_sections = []
    for source_i, (source_label, source_root) in enumerate(UCM_SOURCE_ROOTS, start=1):
        source_key = re.sub(r"[^a-zA-Z0-9]+", "_", source_label).strip("_").lower()
        if not source_key:
            source_key = f"source_{source_i}"

        print(f"\nSource {source_i}: {source_label}")
        print(f"  Source data folder: {source_root}")

        observed_here = set(discover_participants(source_root))
        missing_here = [pid for pid in PARTICIPANTS if pid not in observed_here]
        if missing_here:
            print(f"  [WARN] source participant folders missing: {missing_here}")

        all_photos_by_pid = {}
        for pid in PARTICIPANTS:
            pdir = source_root / f"P{pid}"
            all_photos_by_pid[pid] = find_all_photos(pdir) if pdir.exists() else []

        phase_sections = []
        for phase in REPORT_PHASES:
            print(f"\n  Phase {phase} …")

            # GPS for combined map
            gps_per_pid = {}
            photo_times_per_pid = {}
            for pid in PARTICIPANTS:
                phase_window = key_data.get(pid, {}).get(phase)
                if not phase_window:
                    continue
                pdir = source_root / f"P{pid}"
                if not pdir.exists():
                    continue
                start_ts, end_ts = phase_window
                gps = find_built_gps_for_phase(built_ucm_gps, pid, phase, start_ts, end_ts)
                if gps is None or len(gps) == 0:
                    continue
                gps_per_pid[pid] = gps
                photo_times_per_pid[pid] = (
                    [(f"S+{o}s", start_ts + timedelta(seconds=o)) for o in START_OFFSETS] +
                    [(f"E-{o}s" if o > 0 else "E+0s", end_ts - timedelta(seconds=o))
                     for o in END_OFFSETS]
                )
            map_html = make_combined_map_html(
                gps_per_pid,
                MAP_HEIGHT,
                phase,
                source_key,
                photo_times_per_pid,
            )

            # One photo row per participant
            rows_html = []
            for pid in PARTICIPANTS:
                phase_window = key_data.get(pid, {}).get(phase)
                if not phase_window:
                    continue
                start_ts, end_ts = phase_window
                row = render_participant_row(
                    pid,
                    phase,
                    source_key,
                    start_ts,
                    end_ts,
                    all_photos_by_pid.get(pid, []),
                )
                rows_html.append(row)
                gps_note = "" if pid in gps_per_pid else " (no built GPS track)"
                photo_note = "" if all_photos_by_pid.get(pid) else " (no photos found in source)"
                print(f"    P{pid} OK{gps_note}{photo_note}")

            legend = (
                '<p class="legend-row">'
                '<span style="color:#27ae60">■ green border = phase start photos</span>'
                ' &nbsp;&nbsp; '
                '<span style="color:#e74c3c">■ red border = phase end photos</span>'
                '</p>'
            )
            excluded_note = (
                f'<p class="excluded-note">{EXCLUDED_NOTE_TEXT}</p>'
                if phase in EXCLUDED_NOTE_PHASES else ''
            )
            section = (
                f'<div class="phase-section">'
                f'<h2 class="phase-title">{phase}</h2>'
                + excluded_note + legend + map_html
                + '<div style="margin-top:16px">' + "".join(rows_html) + '</div>'
                + '</div>'
            )
            phase_sections.append(section)

        source_sections.append(
            f'<section class="source-section" data-source="{source_key}">'
            f'<div class="source-title">{html.escape(source_label)}</div>'
            f'<div class="source-path"><b>Source data folder:</b> {html.escape(str(source_root))}</div>'
            + ''.join(phase_sections)
            + '</section>'
        )

    print("\nBuilding HTML …")
    html_text = build_html(source_sections)
    out = Path(OUT_HTML)
    out.write_text(html_text, encoding="utf-8")
    print(f"\nSaved: {out.resolve()}")
    print(f"Size:  {out.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()

