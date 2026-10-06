"""Atmo/LYS quality report from the corrected 10-sec aligned output."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import KEY_FILE, OUTPUTS, load_key_unique


DATA_CSV = OUTPUTS / "03_atmo_lys_merged.csv"
REPORT_HTML = OUTPUTS / "13_atmo_lys_quality_report.html"
MAX_CONNECTED_GAP_SECONDS = 15
DEFAULT_VISIBLE_PIDS = {f"P{p}" for p in [4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]}

PHASES = ["BikeU", "WalkU", "BikeG", "WalkG", "Tram", "reststop"]
PHASE_COLORS = {
    "BikeU": "#d45500",
    "WalkU": "#b8860b",
    "BikeG": "#1a6b1a",
    "WalkG": "#52b852",
    "Tram": "#7f8c8d",
    "reststop": "#b0bec5",
}

PLOT_SIGNALS = [
    ("atmotube_left__atmotube_pm1", "PM1 left", "ug/m3", "#1565C0"),
    ("atmotube_right__atmotube_pm1", "PM1 right", "ug/m3", "#E53935"),
    ("atmotube_left__atmotube_pm2.5", "PM2.5 left", "ug/m3", "#1565C0"),
    ("atmotube_right__atmotube_pm2.5", "PM2.5 right", "ug/m3", "#E53935"),
    ("atmotube_left__atmotube_pm10", "PM10 left", "ug/m3", "#1565C0"),
    ("atmotube_right__atmotube_pm10", "PM10 right", "ug/m3", "#E53935"),
    ("atmotube_left__atmotube_temperature", "Temp left", "C", "#1565C0"),
    ("atmotube_right__atmotube_temperature", "Temp right", "C", "#E53935"),
    ("atmotube_left__atmotube_humidity", "Humidity left", "%", "#1565C0"),
    ("atmotube_right__atmotube_humidity", "Humidity right", "%", "#E53935"),
    ("LYS1__lys_lux", "LYS1 Lux", "lux", "#FB8C00"),
    ("LYS2__lys_lux", "LYS2 Lux", "lux", "#A65E00"),
    ("LYS1__lys_kelvin", "LYS1 Kelvin", "K", "#FB8C00"),
    ("LYS2__lys_kelvin", "LYS2 Kelvin", "K", "#A65E00"),
    ("LYS1__lys_medi", "LYS1 MEDI", "", "#FB8C00"),
    ("LYS2__lys_medi", "LYS2 MEDI", "", "#A65E00"),
    ("LYS1__lys_movement", "LYS1 Movement", "", "#FB8C00"),
    ("LYS2__lys_movement", "LYS2 Movement", "", "#A65E00"),
    ("LYS1__lys_r'", "LYS1 r'", "", "#c62828"),
    ("LYS2__lys_r'", "LYS2 r'", "", "#8e0000"),
    ("LYS1__lys_g'", "LYS1 g'", "", "#2e7d32"),
    ("LYS2__lys_g'", "LYS2 g'", "", "#005005"),
    ("LYS1__lys_b'", "LYS1 b'", "", "#1565C0"),
    ("LYS2__lys_b'", "LYS2 b'", "", "#003c8f"),
    ("LYS1__lys_rgbr", "LYS1 RGB R", "", "#c62828"),
    ("LYS2__lys_rgbr", "LYS2 RGB R", "", "#8e0000"),
    ("LYS1__lys_rgbg", "LYS1 RGB G", "", "#2e7d32"),
    ("LYS2__lys_rgbg", "LYS2 RGB G", "", "#005005"),
    ("LYS1__lys_rgbb", "LYS1 RGB B", "", "#1565C0"),
    ("LYS2__lys_rgbb", "LYS2 RGB B", "", "#003c8f"),
    ("LYS1__lys_rgbir", "LYS1 RGB IR", "", "#6A1B9A"),
    ("LYS2__lys_rgbir", "LYS2 RGB IR", "", "#4A148C"),
]

DISPLAY_SIGNAL_COLS = {col for col, *_ in PLOT_SIGNALS}

ATMO_PAIRS = [
    ("PM1", "ug/m3", "atmotube_left__atmotube_pm1", "Atmo left", "atmotube_right__atmotube_pm1", "Atmo right"),
    ("PM2.5", "ug/m3", "atmotube_left__atmotube_pm2.5", "Atmo left", "atmotube_right__atmotube_pm2.5", "Atmo right"),
    ("PM10", "ug/m3", "atmotube_left__atmotube_pm10", "Atmo left", "atmotube_right__atmotube_pm10", "Atmo right"),
    ("Temperature", "C", "atmotube_left__atmotube_temperature", "Atmo left", "atmotube_right__atmotube_temperature", "Atmo right"),
    ("Humidity", "%", "atmotube_left__atmotube_humidity", "Atmo left", "atmotube_right__atmotube_humidity", "Atmo right"),
]

LYS_PAIRS = [
    ("Lux", "lux", "LYS1__lys_lux", "LYS1", "LYS2__lys_lux", "LYS2"),
    ("Kelvin", "K", "LYS1__lys_kelvin", "LYS1", "LYS2__lys_kelvin", "LYS2"),
    ("MEDI", "", "LYS1__lys_medi", "LYS1", "LYS2__lys_medi", "LYS2"),
    ("Movement", "", "LYS1__lys_movement", "LYS1", "LYS2__lys_movement", "LYS2"),
    ("r'", "", "LYS1__lys_r'", "LYS1", "LYS2__lys_r'", "LYS2"),
    ("g'", "", "LYS1__lys_g'", "LYS1", "LYS2__lys_g'", "LYS2"),
    ("b'", "", "LYS1__lys_b'", "LYS1", "LYS2__lys_b'", "LYS2"),
    ("RGB R", "", "LYS1__lys_rgbr", "LYS1", "LYS2__lys_rgbr", "LYS2"),
    ("RGB G", "", "LYS1__lys_rgbg", "LYS1", "LYS2__lys_rgbg", "LYS2"),
    ("RGB B", "", "LYS1__lys_rgbb", "LYS1", "LYS2__lys_rgbb", "LYS2"),
    ("RGB IR", "", "LYS1__lys_rgbir", "LYS1", "LYS2__lys_rgbir", "LYS2"),
]

LYS_LUX_BANDS = [
    ("Shaded", 0, 1000, "#d6f5d6", "#2e7d32", "<1,000 lux"),
    ("Daylight", 1000, 10000, "#fff2b3", "#8a6d00", "1,000-10,000 lux"),
    ("Bright daylight", 10000, None, "#ffd6a5", "#b85c00", ">10,000 lux"),
]

def load_data() -> pd.DataFrame:
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Missing input: {DATA_CSV}")
    df = pd.read_csv(DATA_CSV, parse_dates=["Datetime"], low_memory=False)
    for flag_col in [col for col in df.columns if col.startswith("QC__")]:
        if flag_col in df.columns:
            df[flag_col] = df[flag_col].astype(str).str.lower().isin(["true", "1", "yes"])
    df["PhaseID"] = (
        df["PhaseID"]
        .where(df["PhaseID"].notna(), "reststop")
        .astype(str)
        .replace({"": "reststop", "nan": "reststop", "None": "reststop"})
    )
    return df.sort_values(["ParticipantID", "Datetime"]).reset_index(drop=True)


def signal_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"ParticipantID", "PhaseID", "Datetime", "Date", "LYS1__lux_cat", "LYS2__lux_cat"}
    return [c for c in df.columns if c not in exclude and not c.startswith("QC__")]


def minute_constancy_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    work = df.copy()
    work["minute"] = work["Datetime"].dt.floor("min")
    for pid, part in work.groupby("ParticipantID", sort=True):
        bad_minutes = 0
        checked_minutes = 0
        for _, minute_df in part.groupby("minute"):
            any_signal = False
            for col in cols:
                vals = pd.to_numeric(minute_df[col], errors="coerce").dropna().round(12).unique()
                if len(vals):
                    any_signal = True
                if len(vals) > 1:
                    bad_minutes += 1
                    break
            if any_signal:
                checked_minutes += 1
        rows.append(
            {
                "ParticipantID": pid,
                "minutes_checked": checked_minutes,
                "minutes_with_within_minute_change": bad_minutes,
            }
        )
    return pd.DataFrame(rows)


def coverage_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["any_signal"] = out[cols].notna().any(axis=1)
    rows = []
    for (pid, phase), part in out.groupby(["ParticipantID", "PhaseID"], sort=True):
        rows.append(
            {
                "ParticipantID": pid,
                "PhaseID": phase,
                "rows": len(part),
                "rows_with_signal": int(part["any_signal"].sum()),
                "coverage_pct": round(100 * part["any_signal"].mean(), 2),
            }
        )
    return pd.DataFrame(rows)


def plot_valid_segments(ax, times, values, color, max_gap_seconds=MAX_CONNECTED_GAP_SECONDS, **kwargs) -> None:
    values = pd.to_numeric(values, errors="coerce")
    valid = values.notna()
    if not valid.any():
        return
    t = times.loc[valid].reset_index(drop=True)
    v = values.loc[valid].reset_index(drop=True)
    gaps = t.diff().dt.total_seconds()
    gaps = gaps.where(gaps.notna(), 0)
    groups = (gaps > max_gap_seconds).cumsum()
    group_sizes = groups.map(groups.value_counts())
    singletons = group_sizes.eq(1)
    if singletons.any():
        ax.plot(
            t.loc[singletons],
            v.loc[singletons],
            linestyle="None",
            marker=kwargs.get("marker", "o"),
            markersize=kwargs.get("markersize", 2.6),
            color=color,
            alpha=kwargs.get("alpha", 0.9),
            zorder=kwargs.get("zorder", 4),
        )
    for _, idx in groups.groupby(groups).groups.items():
        idx = list(idx)
        if len(idx) > 1:
            ax.plot(t.iloc[idx], v.iloc[idx], color=color, **kwargs)


def add_phase_shading(ax, sub: pd.DataFrame) -> None:
    for phase in PHASES:
        ph_df = sub[sub["PhaseID"] == phase]
        if ph_df.empty:
            continue
        start, end = ph_df["Datetime"].min(), ph_df["Datetime"].max()
        ax.axvspan(start, end, color=PHASE_COLORS.get(phase, "#888"), alpha=0.10, zorder=0)
        ax.text(
            start + (end - start) / 2,
            0.97,
            phase,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=11,
            color=PHASE_COLORS.get(phase, "#555"),
            fontweight="bold",
        )


def fig_to_html(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=85, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" />'


def parameter_ylim(df: pd.DataFrame, col: str) -> tuple[float, float] | None:
    series = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)
    vals = series.dropna().to_numpy(dtype=float)
    if not len(vals):
        return None
    lo, hi = np.nanpercentile(vals, 2), np.nanpercentile(vals, 98)
    margin = max((hi - lo) * 0.25, 0.01)
    if "lys_lux" in col:
        return 0, max(12000, hi * 1.1)
    return max(0, lo - margin), hi + margin


def global_y_limits(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    limits = {}
    for col in DISPLAY_SIGNAL_COLS:
        ylim = parameter_ylim(df, col)
        if ylim is not None:
            limits[col] = ylim
    return limits


def participant_y_limits(sub: pd.DataFrame) -> dict[str, tuple[float, float]]:
    limits = {}
    for col in DISPLAY_SIGNAL_COLS:
        ylim = parameter_ylim(sub, col)
        if ylim is not None:
            limits[col] = ylim
    return limits


def participant_timeline_figure(df: pd.DataFrame, pid: str, y_limits: dict[str, tuple[float, float]]):
    sub = df[df["ParticipantID"].astype(str) == pid].copy().sort_values("Datetime")
    local_y_limits = participant_y_limits(sub)
    available = [
        (c, label, unit, color)
        for c, label, unit, color in PLOT_SIGNALS
        if c in DISPLAY_SIGNAL_COLS and c in sub.columns and pd.to_numeric(sub[c], errors="coerce").notna().any()
    ]
    if not available:
        return None

    fig, axes = plt.subplots(len(available), 1, figsize=(18, 1.8 * len(available) + 1.2), sharex=True)
    if len(available) == 1:
        axes = [axes]
    fig.suptitle(f"{pid} Atmo/LYS one-minute values on 10-sec index", fontsize=18, fontweight="bold", y=0.995)

    for ax, (col, label, unit, color) in zip(axes, available):
        add_phase_shading(ax, sub)
        series = pd.to_numeric(sub[col], errors="coerce")
        plot_valid_segments(ax, sub["Datetime"], series, color, linewidth=1.35, marker="o", markersize=2.6, alpha=0.9)
        if col in local_y_limits:
            ax.set_ylim(*local_y_limits[col])
        ylabel = f"{label}\n({unit})" if unit else label
        if "lys_lux" in col:
            ylabel = f"{label}\n(lux; fixed y-axis)\nShaded <1,000\nDaylight 1,000-10,000\nBright >10,000"
        ax.set_ylabel(ylabel, fontsize=13)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, alpha=0.28, linewidth=0.7)
        ax.text(
            0.995,
            0.94,
            f"{series.notna().mean() * 100:.1f}% occupied",
            ha="right",
            va="top",
            fontsize=10,
            color="#555",
            transform=ax.transAxes,
        )

    handles = [mpatches.Patch(color=PHASE_COLORS[p], alpha=0.35, label=p) for p in PHASES]
    axes[-1].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.30), fontsize=11, ncol=6, frameon=False)
    axes[-1].set_xlabel("Datetime", fontsize=13)
    plt.tight_layout(rect=[0, 0.035, 1, 0.987])
    return fig


def paired_sensor_figure(df: pd.DataFrame, pid: str, pairs: list[tuple], title: str, colors: tuple[str, str]):
    sub = df[df["ParticipantID"].astype(str) == pid].copy().sort_values("Datetime")
    if sub.empty:
        return None
    fig, axes = plt.subplots(len(pairs), 2, figsize=(22, 2.25 * len(pairs) + 1.4), sharex=True)
    fig.suptitle(f"{pid} {title}: paired sensor parameters", fontsize=18, fontweight="bold", y=0.995)
    for row, (param, unit, left_col, left_label, right_col, right_label) in enumerate(pairs):
        left = pd.to_numeric(sub[left_col], errors="coerce") if left_col in sub.columns else pd.Series(np.nan, index=sub.index)
        right = pd.to_numeric(sub[right_col], errors="coerce") if right_col in sub.columns else pd.Series(np.nan, index=sub.index)
        both = pd.concat([left, right]).dropna()
        if both.empty:
            ymin, ymax = 0, 1
        elif "lux" in left_col:
            ymin, ymax = 0, max(12000, float(both.max()) * 1.12)
        elif "kelvin" in left_col:
            ymin, ymax = 1000, max(12000, float(both.max()) * 1.05)
        elif "medi" in left_col:
            ymin, ymax = 0, 8750
        else:
            lo, hi = np.nanpercentile(both, 2), np.nanpercentile(both, 98)
            margin = max((hi - lo) * 0.25, 0.01)
            ymin, ymax = max(0, lo - margin), hi + margin

        for ax, series, side_label, color in [
            (axes[row, 0], left, left_label, colors[0]),
            (axes[row, 1], right, right_label, colors[1]),
        ]:
            add_phase_shading(ax, sub)
            plot_valid_segments(ax, sub["Datetime"], series, color, linewidth=1.15, marker="o", markersize=2.3, alpha=0.9, zorder=4)
            if series.notna().sum() == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=12, color="#777")
            ax.set_ylim(ymin, ymax)
            ax.set_title(f"{side_label} - {param}", fontsize=11, fontweight="bold")
            ax.set_ylabel(f"{param}\n({unit})" if unit else param, fontsize=10)
            ax.grid(True, alpha=0.25, linewidth=0.7)
            ax.tick_params(axis="both", labelsize=9)

    axes[-1, 0].set_xlabel("Datetime", fontsize=12)
    axes[-1, 1].set_xlabel("Datetime", fontsize=12)
    handles = [mpatches.Patch(color=PHASE_COLORS[p], alpha=0.35, label=p) for p in PHASES]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=10)
    plt.tight_layout(rect=[0, 0.035, 1, 0.985])
    return fig


def lys_threshold_figure(df: pd.DataFrame, pid: str, y_limits: dict[str, tuple[float, float]]):
    sub = df[df["ParticipantID"].astype(str) == pid].copy().sort_values("Datetime")
    sensors = [
        (col, label)
        for col, label in [("LYS1__lys_lux", "LYS1 Lux"), ("LYS2__lys_lux", "LYS2 Lux")]
        if col in sub.columns and pd.to_numeric(sub[col], errors="coerce").notna().any()
    ]
    if not sensors:
        return None
    fig, axes = plt.subplots(len(sensors), 1, figsize=(20, 3.2 * len(sensors) + 1.2), sharex=True)
    if len(sensors) == 1:
        axes = [axes]
    fig.suptitle(f"{pid} LYS lux with Shaded / Daylight / Bright daylight bands", fontsize=18, fontweight="bold")
    for ax, (col, label) in zip(axes, sensors):
        series = pd.to_numeric(sub[col], errors="coerce")
        ymin, ymax = 0, max(12000, float(series.max(skipna=True)) * 1.12)
        for _, low, high, fill, _, _ in LYS_LUX_BANDS:
            span_high = ymax if high is None else min(high, ymax)
            if ymax > low and span_high > low:
                ax.axhspan(low, span_high, color=fill, alpha=0.62, zorder=0)
        ax.axhline(1000, color="#4CAF50", linewidth=0.9, alpha=0.9)
        ax.axhline(10000, color="#E67E22", linewidth=0.9, alpha=0.9)
        add_phase_shading(ax, sub)
        plot_valid_segments(ax, sub["Datetime"], series, "#1f2d3d", linewidth=1.1, marker="o", markersize=2.6, alpha=0.9, zorder=4)
        band_positions = [
            ("SHADED\n<1,000 lux", min(500, ymax * 0.18), "#2e7d32"),
            ("DAYLIGHT\n1,000-10,000 lux", min(5500, ymax * 0.46), "#8a6d00"),
            ("BRIGHT DAYLIGHT\n>10,000 lux", max(10500, ymax * 0.76), "#b85c00"),
        ]
        for text, y, color in band_positions:
            if ymin <= y <= ymax:
                ax.text(0.012, y, text, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=12, color=color, fontweight="bold")
        ax.set_ylabel(f"{label}\n(lux; fixed y-axis)", fontsize=13)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.28, linewidth=0.7)
        ax.tick_params(axis="both", labelsize=11)
    axes[-1].set_xlabel("Datetime", fontsize=13)
    plt.tight_layout()
    return fig


def write_report(df: pd.DataFrame) -> None:
    expected_pids = [f"P{pid}" for pid in load_key_unique(KEY_FILE)["Participant_ID"].astype(int).tolist()]
    observed_pids = set(df["ParticipantID"].dropna().astype(str))
    y_limits = global_y_limits(df)
    participant_checks = "".join(
        f"<label><input type='checkbox' class='participant-cb' data-pid='{pid}'{' checked' if pid in DEFAULT_VISIBLE_PIDS else ''}> {pid}</label>"
        for pid in expected_pids
    )

    html = [
        "<html><head><meta charset='utf-8'><title>Atmo/LYS Quality Report</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;color:#222}"
        "img{max-width:100%;height:auto;border:1px solid #ddd;margin:14px 0 34px}"
        "h2{margin-top:36px}"
        ".controls{background:#fff;border:1px solid #ddd;border-radius:6px;padding:12px;margin:18px 0}"
        ".controls label{display:inline-flex;align-items:center;gap:6px;margin:4px 12px 4px 0;font-weight:600}"
        ".participant-section{border-top:1px solid #ddd;margin-top:18px}</style>",
        "</head><body>",
        "<h1>Atmo/LYS Quality Report</h1>",
        f"<p>Input: {DATA_CSV}</p>",
        "<p>Y-axes are participant-specific robust scales for readability. "
        "LYS lux bands: Shaded &lt;1,000 lux; Daylight 1,000-10,000 lux; Bright daylight &gt;10,000 lux.</p>",
        "<p>Atmo/LYS readings are approximately one-minute observations aligned to the 10-second index. "
        "Each observed minute value is repeated at the six 10-second slots inside that minute. "
        "Missing source minutes remain blank. Plots connect occupied adjacent 10-second slots and break when "
        f"the gap is more than {MAX_CONNECTED_GAP_SECONDS} seconds.</p>",
        f"<div class='controls'><strong>Visible participants:</strong> {participant_checks}</div>",
    ]

    participants = expected_pids
    for pid in participants:
        visible = pid in DEFAULT_VISIBLE_PIDS
        section_style = "" if visible else " style='display:none'"
        html.append(f"<div class='participant-section' data-pid='{pid}'{section_style}>")
        html.append(f"<h2>{pid}</h2>")
        if pid not in observed_pids:
            html.append("<p>No Atmo/LYS rows found for this key participant.</p>")
            html.append("</div>")
            continue
        html.append("<div>")
        html.append("<h3>Atmotube paired parameters: left vs right</h3>")
        fig = paired_sensor_figure(df, pid, ATMO_PAIRS, "Atmotube", ("#1565C0", "#E53935"))
        if fig is not None:
            html.append(fig_to_html(fig))
        html.append("<h3>LYS paired parameters: LYS1 vs LYS2</h3>")
        fig = paired_sensor_figure(df, pid, LYS_PAIRS, "LYS", ("#FB8C00", "#A65E00"))
        if fig is not None:
            html.append(fig_to_html(fig))
        html.append("<h3>LYS lux exposure bands</h3>")
        fig = lys_threshold_figure(df, pid, y_limits)
        if fig is not None:
            html.append(fig_to_html(fig))
        html.append("</div></div>")

    html.append("""<script>
document.querySelectorAll('.participant-cb').forEach(function(cb){
  cb.addEventListener('change', function(){
    document.querySelectorAll('.participant-section[data-pid="'+cb.dataset.pid+'"]').forEach(function(el){ el.style.display = cb.checked ? '' : 'none'; });
  });
});
</script></body></html>""")
    REPORT_HTML.write_text("\n".join(html), encoding="utf-8")
    print(f"Saved report: {REPORT_HTML}")


def main() -> None:
    df = load_data()
    write_report(df)


if __name__ == "__main__":
    main()
