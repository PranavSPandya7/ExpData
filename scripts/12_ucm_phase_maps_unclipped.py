from __future__ import annotations

import json
from pathlib import Path
import re

import branca
import folium
import numpy as np
import pandas as pd

import sys; from pathlib import Path as _Path; sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _paths import KEY_FILE as KEY_CSV, OUTPUTS, RAW_UCM_DIR, load_key_unique

UCM_ROOT = RAW_UCM_DIR
OUT_DIR = OUTPUTS / "12_ucm_phase_maps_unclipped"
PHASES = ["BikeU", "WalkU", "BikeG", "WalkG", "Tram"]
# P1 map-only fallback: P1 lacks reliable per-phase UCM GPS after quality
# filtering, so this validation map uses documented phase windows instead.
P1_FALLBACK_WINDOWS = {
    "WalkU": ("2025-07-03 12:24:07", "2025-07-03 12:51:21"),
    "BikeU": ("2025-07-03 12:59:46", "2025-07-03 13:36:01"),
    "BikeG": ("2025-07-03 13:58:52", "2025-07-03 14:10:28"),
    "WalkG": ("2025-07-03 14:34:11", "2025-07-03 15:08:34"),
    "Tram": ("2025-07-03 15:32:30", "2025-07-03 15:40:20"),
}

MONTH_MAP = {
    "jan": 1, "fev": 2, "feb": 2, "mar": 3, "avr": 4, "apr": 4,
    "mai": 5, "may": 5, "juin": 6, "jun": 6, "juil": 7, "jul": 7,
    "aout": 8, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
YEAR = 2025
CAM_RE = re.compile(r"^cam_(\d{8})_(\d{6})(?:_\d+)?\.jpg$", re.IGNORECASE)

PARTICIPANT_COLORS = {
    "P1": "#4e79a7",
    "P2": "#e15759",
    "P3": "#76b7b2",
    "P4": "#f28e2b",
    "P5": "#59a14f",
    "P6": "#edc948",
    "P7": "#b07aa1",
    "P8": "#ff9da7",
    "P9": "#9c755f",
    "P10": "#bab0ab",
    "P11": "#1f77b4",
    "P12": "#d62728",
    "P13": "#2ca02c",
    "P14": "#9467bd",
    "P15": "#8c564b",
    "P16": "#e377c2",
    "P17": "#7f7f7f",
    "P18": "#bcbd22",
}


def discover_participants(root: Path) -> list[tuple[int, Path]]:
    participants = []
    for pdir in sorted(root.iterdir()):
        if not pdir.is_dir():
            continue
        match = re.fullmatch(r"P(\d+)", pdir.name, re.IGNORECASE)
        if match:
            participants.append((int(match.group(1)), pdir))
    return participants


def parse_date(value: str) -> pd.Timestamp:
    parts = str(value).strip().split("-")
    return pd.Timestamp(year=YEAR, month=MONTH_MAP[parts[1].lower()], day=int(parts[0]))


def read_key_windows() -> dict[int, dict[str, tuple[pd.Timestamp, pd.Timestamp]]]:
    df = load_key_unique(KEY_CSV)
    windows: dict[int, dict[str, tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for _, row in df.iterrows():
        try:
            pid = int(row["Participant_ID"])
            date = parse_date(str(row["Date"]))
        except Exception:
            continue
        phase_windows = {}
        for phase in PHASES:
            start_col = f"{phase}_start"
            end_col = f"{phase}_end"
            if start_col not in row or end_col not in row:
                continue
            start_raw = str(row[start_col]).strip()
            end_raw = str(row[end_col]).strip()
            if start_raw in {"", "nan"} or end_raw in {"", "nan"}:
                continue
            try:
                start_time = pd.to_datetime(start_raw, format="%H:%M:%S").time()
                end_time = pd.to_datetime(end_raw, format="%H:%M:%S").time()
            except Exception:
                continue
            phase_windows[phase] = (
                pd.Timestamp.combine(date.date(), start_time),
                pd.Timestamp.combine(date.date(), end_time),
            )
        windows[pid] = phase_windows
    return windows


def camera_timestamp(path: Path) -> pd.Timestamp | None:
    match = CAM_RE.match(path.name)
    if not match:
        return None
    return pd.to_datetime(match.group(1) + match.group(2), format="%Y%m%d%H%M%S", errors="coerce")


def infer_phase_window_from_photos(pdir: Path, phase: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    phase_dir = pdir / phase
    if not phase_dir.exists():
        return None
    times = []
    for photo in phase_dir.rglob("cam_*.jpg"):
        ts = camera_timestamp(photo)
        if ts is not None and not pd.isna(ts):
            times.append(ts)
    if len(times) < 2:
        return None
    return min(times), max(times)


def fallback_phase_window(pid: int, phase: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if pid != 1 or phase not in P1_FALLBACK_WINDOWS:
        return None
    start_raw, end_raw = P1_FALLBACK_WINDOWS[phase]
    return pd.Timestamp(start_raw), pd.Timestamp(end_raw)


def candidate_ucm_csvs(pdir: Path, phase: str) -> list[Path]:
    phase_patterns = [
        f"{phase}/ucm/data.csv",
        f"{phase}/ucm/*/data.csv",
        f"inputdata/{phase}/ucm/data.csv",
        f"inputdata/{phase}/ucm/*/data.csv",
    ]
    shared_patterns = [
        "inputdata/ucm/data.csv",
        "inputdata/ucm/*/data.csv",
    ]
    seen = set()
    phase_candidates = []
    shared_candidates = []
    for pattern in phase_patterns:
        for path in pdir.glob(pattern):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                phase_candidates.append(path)
    for pattern in shared_patterns:
        for path in pdir.glob(pattern):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                shared_candidates.append(path)
    return phase_candidates + shared_candidates


def read_data_csv(csv_path: Path) -> pd.DataFrame | None:
    try:
        with csv_path.open(encoding="utf-8", errors="replace") as handle:
            header_line = None
            for i, line in enumerate(handle):
                if line.startswith("#") and "GPS_time" in line:
                    header_line = line
                    skiprows = i + 2
                    break
        if header_line is None:
            return None

        columns = [col.strip() for col in header_line.lstrip("# ").split(",")]
        df = pd.read_csv(
            csv_path,
            skiprows=skiprows,
            header=None,
            names=columns,
            encoding="utf-8",
            engine="python",
            na_values=["", "nan"],
        )
        df["GPS_time"] = pd.to_datetime(df["GPS_time"], errors="coerce")
        df["GPS_lat"] = pd.to_numeric(df["GPS_lat"], errors="coerce")
        df["GPS_lon"] = pd.to_numeric(df["GPS_lon"], errors="coerce")
        if "GPS_HDOP" in df.columns:
            df["GPS_HDOP"] = pd.to_numeric(df["GPS_HDOP"], errors="coerce")
        if "IO_flag" in df.columns:
            df["IO_flag"] = pd.to_numeric(df["IO_flag"], errors="coerce")
        df = df.dropna(subset=["GPS_time", "GPS_lat", "GPS_lon"]).sort_values("GPS_time")
        return df.reset_index(drop=True) if not df.empty else None
    except Exception:
        return None


def filter_bad_gps(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    bad = np.zeros(len(df), dtype=bool)
    if "GPS_HDOP" in df.columns:
        bad |= df["GPS_HDOP"].to_numpy() > 5
    if "IO_flag" in df.columns:
        bad |= df["IO_flag"].to_numpy() == 9
    bad[1:] |= bad[:-1].copy()

    filtered = df.loc[~bad].copy()
    return filtered.reset_index(drop=True)


def gps_step_speeds_mps(df: pd.DataFrame) -> np.ndarray:
    if len(df) < 2:
        return np.array([], dtype=float)
    lat1 = np.radians(df["GPS_lat"].to_numpy(dtype=float)[:-1])
    lat2 = np.radians(df["GPS_lat"].to_numpy(dtype=float)[1:])
    dlat = lat2 - lat1
    dlon = np.radians(df["GPS_lon"].to_numpy(dtype=float)[1:] - df["GPS_lon"].to_numpy(dtype=float)[:-1])
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    dist_m = 2 * 6371000.0 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dt_s = df["GPS_time"].diff().dt.total_seconds().to_numpy(dtype=float)[1:]
    good = dt_s > 0
    return dist_m[good] / dt_s[good]


def filter_phase_window(df: pd.DataFrame, window: tuple[pd.Timestamp, pd.Timestamp] | None) -> pd.DataFrame:
    if window is None:
        return df
    start_ts, end_ts = window
    filtered = df[(df["GPS_time"] >= start_ts) & (df["GPS_time"] <= end_ts)].copy()
    return filtered.reset_index(drop=True)


def choose_best_track(tracks: list[pd.DataFrame]) -> pd.DataFrame | None:
    viable = []
    for df in tracks:
        if len(df) < 2:
            continue
        speeds = gps_step_speeds_mps(df)
        max_speed = float(np.nanmax(speeds)) if len(speeds) else 0.0
        p95_speed = float(np.nanpercentile(speeds, 95)) if len(speeds) else 0.0
        viable.append((max_speed > 15.0, max_speed, p95_speed, -len(df), df))
    if not viable:
        return None
    viable.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return viable[0][4].reset_index(drop=True)


def load_phase_tracks() -> dict[str, dict[int, pd.DataFrame]]:
    phase_tracks: dict[str, dict[int, pd.DataFrame]] = {phase: {} for phase in PHASES}
    participants = discover_participants(UCM_ROOT)
    key_windows = read_key_windows()
    print(f"Found {len(participants)} participant folders under {UCM_ROOT}")

    for pid, pdir in participants:
        for phase in PHASES:
            phase_tracks_for_pid = []
            shared_tracks_for_pid = []
            phase_fallback_tracks_for_pid = []
            shared_fallback_tracks_for_pid = []
            phase_window = key_windows.get(pid, {}).get(phase)
            if phase_window is None:
                phase_window = infer_phase_window_from_photos(pdir, phase)
            if phase_window is None:
                phase_window = fallback_phase_window(pid, phase)
            for csv_path in candidate_ucm_csvs(pdir, phase):
                df = read_data_csv(csv_path)
                if df is None:
                    continue
                parts_lower = [part.lower() for part in csv_path.parts]
                is_shared_input = "inputdata" in parts_lower and phase.lower() not in parts_lower
                if is_shared_input:
                    if phase_window is None:
                        continue
                    df = filter_phase_window(df, phase_window)
                filtered = filter_bad_gps(df)
                if len(filtered) >= 2:
                    if is_shared_input:
                        shared_tracks_for_pid.append(filtered)
                    else:
                        phase_tracks_for_pid.append(filtered)
                elif len(df) >= 2:
                    if is_shared_input:
                        shared_fallback_tracks_for_pid.append(df)
                    else:
                        phase_fallback_tracks_for_pid.append(df)
            tracks = (
                phase_tracks_for_pid
                or shared_tracks_for_pid
                or phase_fallback_tracks_for_pid
                or shared_fallback_tracks_for_pid
            )
            if not tracks:
                continue

            merged = choose_best_track(tracks)
            if merged is not None and len(merged) >= 2:
                phase_tracks[phase][pid] = merged
                fallback_note = ""
                if not phase_tracks_for_pid and not shared_tracks_for_pid:
                    fallback_note = " (raw GPS fallback; HDOP/IO removed all rows)"
                print(f"  {phase} P{pid}: {len(merged)} points{fallback_note}")

    return phase_tracks


def build_map(phase: str, tracks: dict[int, pd.DataFrame]) -> folium.Map | None:
    if not tracks:
        return None

    all_lats = []
    all_lons = []
    for df in tracks.values():
        all_lats.extend(df["GPS_lat"].tolist())
        all_lons.extend(df["GPS_lon"].tolist())

    center = [sum(all_lats) / len(all_lats), sum(all_lons) / len(all_lons)]
    fmap = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")
    map_name = fmap.get_name()
    pid_varnames = {}
    line_varnames = []
    pid_colors_used = {}
    pid_durations = {}
    hover_track_vars = []
    hover_meta_rows = []

    for pid in sorted(tracks):
        df = tracks[pid]
        color = PARTICIPANT_COLORS.get(f"P{pid}", "#3388ff")
        coords = list(zip(df["GPS_lat"], df["GPS_lon"]))
        tooltip = (
            f"P{pid} | {phase} | {len(coords)} points | "
            f"{df['GPS_time'].iloc[0]} -> {df['GPS_time'].iloc[-1]}"
        )
        fg = folium.FeatureGroup(name=f"P{pid}", show=True)
        line = folium.PolyLine(coords, color=color, weight=3, opacity=0.85, tooltip=tooltip)
        line.add_to(fg)
        line_name = line.get_name()
        line_varnames.append(line_name)
        folium.CircleMarker(coords[0], radius=4, color=color, fill=True, fill_opacity=1).add_to(fg)
        folium.CircleMarker(coords[-1], radius=4, color=color, fill=True, fill_color="#ffffff", fill_opacity=1).add_to(fg)
        fg.add_to(fmap)
        pid_varnames[pid] = fg.get_name()
        pid_colors_used[pid] = color
        duration_minutes = (pd.Timestamp(df["GPS_time"].iloc[-1]) - pd.Timestamp(df["GPS_time"].iloc[0])).total_seconds() / 60.0
        pid_durations[pid] = f"{duration_minutes:.2f}"

        step = max(1, len(df) // 600)
        pts = []
        for _, row in df.iloc[::step].iterrows():
            pts.append([round(float(row["GPS_lat"]), 6), round(float(row["GPS_lon"]), 6), pd.Timestamp(row["GPS_time"]).strftime("%H:%M:%S")])
        track_var = f"_track_{pid}"
        hover_track_vars.append(f"var {track_var} = {json.dumps(pts)};")
        hover_meta_rows.append(
            "{"
            f"pid:'P{pid}',"
            f"pidNum:{pid},"
            f"featureVar:{json.dumps(fg.get_name())},"
            f"lineVar:{json.dumps(line_name)},"
            f"points:{track_var},"
            f"color:{json.dumps(color)},"
            f"duration:{json.dumps(pid_durations[pid])}"
            "}"
        )
    if not pid_varnames:
        return None

    cb_rows = ""
    for pid in sorted(pid_varnames):
        cb_rows += (
            f'<label style="display:flex;align-items:center;gap:7px;margin:3px 0;cursor:pointer;user-select:none;">'
            f'<input type="checkbox" class="track-cb" data-vn="{pid_varnames[pid]}" checked '
            f'style="cursor:pointer;width:13px;height:13px;">'
            f'<span style="width:22px;height:4px;background:{pid_colors_used[pid]};border-radius:2px;display:inline-block;flex-shrink:0;"></span>'
            f'<span style="font-size:11px;color:#222;">P{pid} - {pid_durations[pid]}</span>'
            f'</label>'
        )

    panel_html = f"""
<div id="track-panel" style="position:fixed;top:70px;right:10px;z-index:9999;background:rgba(255,255,255,0.96);padding:10px 14px;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,0.28);min-width:115px;max-height:390px;overflow-y:auto;font-family:Arial,sans-serif;">
  <b style="font-size:12px;display:block;margin-bottom:6px;color:#111;">Participants</b>
  <label style="display:flex;align-items:center;gap:7px;margin-bottom:6px;cursor:pointer;user-select:none;">
    <input type="checkbox" id="cb-all" checked style="cursor:pointer;width:13px;height:13px;">
    <span style="font-size:11px;font-weight:700;color:#333;">All / None</span>
  </label>
  <hr style="margin:4px 0;border:0;border-top:1px solid #e0e0e0;">
  {cb_rows}
</div>
"""

    reg_lines = "\n".join([f'tracks["{vn}"] = window["{vn}"];' for vn in pid_varnames.values()])
    toggle_js = f"""
<script>
(function() {{
  var mapObj, tracks = {{}};
  function syncAllCheckbox() {{
    var boxes = Array.from(document.querySelectorAll('.track-cb'));
    var checked = boxes.filter(function(cb) {{ return cb.checked; }}).length;
    var all = document.getElementById('cb-all');
    all.checked = checked === boxes.length;
    all.indeterminate = checked > 0 && checked < boxes.length;
  }}
  function resolveRuntimeObjects() {{
    mapObj = window[{json.dumps(map_name)}];
    if (!mapObj) return false;
    tracks = {{}};
    {reg_lines}
    return Object.values(tracks).every(function(layer) {{ return !!layer; }});
  }}
  function bindControls() {{
    if (!resolveRuntimeObjects()) return false;
    document.querySelectorAll('.track-cb').forEach(function(cb) {{
      cb.addEventListener('change', function() {{
        var layer = tracks[this.getAttribute('data-vn')];
        if (!layer) return;
        if (this.checked) layer.addTo(mapObj);
        else mapObj.removeLayer(layer);
        syncAllCheckbox();
      }});
    }});
    document.getElementById('cb-all').addEventListener('change', function() {{
      var show = this.checked;
      this.indeterminate = false;
      document.querySelectorAll('.track-cb').forEach(function(cb) {{
        cb.checked = show;
        var layer = tracks[cb.getAttribute('data-vn')];
        if (!layer) return;
        if (show) layer.addTo(mapObj);
        else mapObj.removeLayer(layer);
      }});
      var layer = tracks[cb.getAttribute('data-vn')];
      cb.checked = !!(layer && mapObj.hasLayer(layer));
    }});
    syncAllCheckbox();
    return true;
  }}
  function init() {{
    if (bindControls()) return;
    var tries = 0;
    var timer = setInterval(function() {{
      tries += 1;
      if (bindControls() || tries >= 60) clearInterval(timer);
    }}, 250);
  }}
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
}})();
</script>
"""

    hover_js = f"""
<script>
(function() {{
  {" ".join(hover_track_vars)}
  var rawTrackMeta = [{",".join(hover_meta_rows)}];
  var mapVarName = {json.dumps(map_name)};
  var hoverBox = document.createElement('div');
  hoverBox.style.cssText = 'position:fixed;background:rgba(20,20,20,0.82);color:#fff;padding:5px 10px;border-radius:5px;font-size:12px;font-family:monospace;pointer-events:none;z-index:99999;display:none;white-space:nowrap;box-shadow:0 2px 6px rgba(0,0,0,.4)';
  document.body.appendChild(hoverBox);
  var mapObj = null;
  var trackMeta = [];
  var hoverMarkers = {{}};
  function resolveRuntimeObjects() {{
    mapObj = window[mapVarName];
    if (!mapObj) return false;
    trackMeta = rawTrackMeta.map(function(meta) {{
      return {{
        pid: meta.pid,
        pidNum: meta.pidNum,
        feature: window[meta.featureVar],
        line: window[meta.lineVar],
        points: meta.points,
        color: meta.color,
        duration: meta.duration
      }};
    }}).filter(function(meta) {{
      return meta.feature && meta.line && meta.points;
    }});
    return trackMeta.length > 0;
  }}
  function getHoverMarker(pid, color) {{
    if (!hoverMarkers[pid]) {{
      hoverMarkers[pid] = L.circleMarker([0, 0], {{
        radius: 6,
        color: color,
        weight: 2,
        fillColor: '#ffffff',
        fillOpacity: 1,
        opacity: 1
      }});
    }}
    return hoverMarkers[pid];
  }}
  function hideAllHoverMarkers() {{
    Object.keys(hoverMarkers).forEach(function(pid) {{
      if (mapObj && mapObj.hasLayer(hoverMarkers[pid])) {{
        mapObj.removeLayer(hoverMarkers[pid]);
      }}
    }});
  }}
  function findNearest(track, latlng) {{
    var best = null, bestDist = Infinity;
    for (var i = 0; i < track.length; i++) {{
      var d = mapObj.distance(L.latLng(track[i][0], track[i][1]), latlng);
      if (d < bestDist) {{
        bestDist = d;
        best = track[i];
      }}
    }}
    return best ? {{point: best, dist: bestDist}} : null;
  }}
  function nearbyParticipants(latlng) {{
    var rows = [];
    trackMeta.forEach(function(meta) {{
      if (!mapObj.hasLayer(meta.feature)) return;
      var nearest = findNearest(meta.points, latlng);
      if (!nearest || nearest.dist > 50) return;
      rows.push({{
        meta: meta,
        point: nearest.point,
        dist: nearest.dist
      }});
    }});
    rows.sort(function(a, b) {{ return a.meta.pidNum - b.meta.pidNum; }});
    return rows;
  }}
  function updateHoverState(latlng, clientX, clientY) {{
    var nearby = nearbyParticipants(latlng);
    if (!nearby.length) {{
      hoverBox.style.display = 'none';
      hideAllHoverMarkers();
      return;
    }}
    hideAllHoverMarkers();
    nearby.forEach(function(item) {{
      var marker = getHoverMarker(item.meta.pid, item.meta.color);
      marker.setLatLng([item.point[0], item.point[1]]);
      if (!mapObj.hasLayer(marker)) marker.addTo(mapObj);
    }});
    hoverBox.style.display = 'block';
    hoverBox.style.left = (clientX + 14) + 'px';
    hoverBox.style.top = (clientY - 28) + 'px';
    hoverBox.innerHTML =
      '<div style="font-weight:700;margin-bottom:4px">Nearby tracks (50m) - ' + nearby.length + '</div>' +
      nearby.map(function(item) {{
        return '<div><b style="color:' + item.meta.color + '">' + item.meta.pid + '</b> - ' +
          item.meta.duration + ' min - ' + item.point[2] + '</div>';
      }}).join('');
  }}
  function attachHover() {{
    if (!resolveRuntimeObjects()) return false;
    mapObj.on('mousemove', function(e) {{
      var source = e.originalEvent || e;
      updateHoverState(e.latlng, source.clientX || 0, source.clientY || 0);
    }});
    mapObj.on('mouseout', function() {{
      hoverBox.style.display = 'none';
      hideAllHoverMarkers();
    }});
    mapObj.on('zoomstart', function() {{
      hoverBox.style.display = 'none';
      hideAllHoverMarkers();
    }});
    mapObj.on('movestart', function() {{
      hoverBox.style.display = 'none';
      hideAllHoverMarkers();
    }});
    return true;
  }}
  function initHover() {{
    if (attachHover()) return;
    var tries = 0;
    var timer = setInterval(function() {{
      tries += 1;
      if (attachHover() || tries >= 60) clearInterval(timer);
    }}, 250);
  }}
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initHover);
  else initHover();
}})();
</script>
"""

    fmap.get_root().html.add_child(branca.element.Element(panel_html))
    fmap.get_root().html.add_child(branca.element.Element(toggle_js))
    fmap.get_root().html.add_child(branca.element.Element(hover_js))

    return fmap


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase_tracks = load_phase_tracks()

    for phase in PHASES:
        out_path = OUT_DIR / f"{phase.lower()}_all_participants_unclipped_map.html"
        fmap = build_map(phase, phase_tracks[phase])
        if fmap is None:
            print(f"No usable GPS tracks found for {phase}")
            continue
        fmap.save(out_path)
        print(f"Saved {phase} map: {out_path}")


if __name__ == "__main__":
    main()
