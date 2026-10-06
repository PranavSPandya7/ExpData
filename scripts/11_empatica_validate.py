"""
11_empatica_validate.py
=======================
Generates a quality HTML report for Empatica data with full-day
phase-shaded time-series plots and per-phase breakdown tables.

Reads:  Paper3_Github/output/01_empatica_corrected_10sec.csv
        Paper3_Github/output/key.csv
Output: Paper3_Github/output/11_empatica_quality_report.html
"""
import base64, io
from datetime import datetime
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os
from _paths import OUTPUTS, KEY_FILE, load_key_unique

default_csv = OUTPUTS / '01_empatica_corrected_10sec.csv'
pending_csv = OUTPUTS / '01_empatica_corrected_10sec_PENDING_CLOSE_OPEN_FILE.csv'
if pending_csv.exists():
    default_csv = pending_csv
CSV_IN   = Path(os.environ.get('EMPATICA_VALIDATE_CSV', str(default_csv)))
csv_name = CSV_IN.stem
if csv_name in ('01_empatica_corrected_10sec', '01_empatica_corrected_10sec_PENDING_CLOSE_OPEN_FILE'):
    HTML_OUT = OUTPUTS / '11_empatica_quality_report.html'
else:
    HTML_OUT = OUTPUTS / f'11_empatica_quality_report_{csv_name}.html'
HTML_OUT = Path(os.environ.get('EMPATICA_VALIDATE_HTML_OUT', str(HTML_OUT)))
ONLY_PARTICIPANTS = {
    p.strip()
    for p in os.environ.get('EMPATICA_VALIDATE_PARTICIPANTS', '').split(',')
    if p.strip()
}
DEFAULT_VISIBLE_PIDS = {f'P{p}' for p in [4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]}

PHASES = ['BikeU','WalkU','BikeG','WalkG','Tram']
PHASE_ID = {'BikeU':'BikeU','WalkU':'WalkU','BikeG':'BikeG','WalkG':'WalkG','Tram':'Tram'}
PHASE_COLORS = {'BikeU':'#d45500','WalkU':'#b8860b','BikeG':'#1a6b1a','WalkG':'#52b852','Tram':'#7f8c8d'}
STATUS_COLOR = {'GOOD':'#27ae60','FAIR':'#e67e22','POOR':'#c0392b','NO DATA':'#7f8c8d'}

SIGNAL_PLOTS = [
    ('EDA (\u00b5S)', 'eda', '#1565C0'),
    ('EDA Tonic', 'eda_tonic', '#64B5F6'),
    ('EDA Phasic', 'eda_phasic', '#0277BD'),
    ('HR (bpm)', 'heart_rate', '#E53935'),
    ('PPG RMSSD (ms)', 'hrv_td_rmssd', '#8E44AD'),
    ('PPG SDNN (ms)', 'hrv_td_sdnn', '#5E35B1'),
    ('PPG LF Power', 'hrv_fd_lf', '#6A1B9A'),
    ('PPG HF Power', 'hrv_fd_hf', '#AB47BC'),
    ('PPG LF/HF Ratio', 'hrv_fd_lf_hf_ratio', '#8E24AA'),
    ('Temp (\u00b0C)', 'temperature', '#FB8C00'),
    ('Acc |g|', 'vector_magnitude', '#43A047'),
]
if os.environ.get('EMPATICA_VALIDATE_SKIP_FD_PLOTS') == '1':
    SIGNAL_PLOTS = [
        spec for spec in SIGNAL_PLOTS
        if spec[1] not in {'hrv_fd_lf', 'hrv_fd_hf', 'hrv_fd_lf_hf_ratio'}
    ]

KEY_TO_DATA_OFFSET_HOURS = 2


def parse_key_date(d):
    return datetime.strptime(f'{d}-2025','%d-%b-%Y').strftime('%Y-%m-%d')


def build_phase_windows(key):
    windows = {}
    for _,r in key.iterrows():
        if pd.isna(r.get('Participant_ID')): continue
        pid = int(r['Participant_ID']); date = parse_key_date(str(r['Date']))
        for ph in PHASES:
            sc,ec = f'{ph}_start',f'{ph}_end'
            if sc not in r.index or pd.isna(r.get(sc)) or pd.isna(r.get(ec)): continue
            try:
                s = pd.Timestamp(f"{date} {r[sc]}") + pd.Timedelta(hours=KEY_TO_DATA_OFFSET_HOURS)
                e = pd.Timestamp(f"{date} {r[ec]}") + pd.Timedelta(hours=KEY_TO_DATA_OFFSET_HOURS)
                if e < s: e += pd.Timedelta(days=1)
                windows[(pid,ph)] = (s,e)
            except: pass
    return windows


def compute_quality_metrics(df_day, windows, pid, date_str):
    m = {'pid':pid,'date':date_str,'total_rows':len(df_day),'eda_coverage':0,'eda_zero_pct':0,
         'eda_mean':np.nan,'eda_min':np.nan,'eda_max':np.nan,'hr_coverage':0,'hr_mean':np.nan,
         'hr_min':np.nan,'hr_max':np.nan,'hrv_mean':np.nan,'sdnn_mean':np.nan,'lf_mean':np.nan,
         'hf_mean':np.nan,'lfhf_mean':np.nan,'rri_valid_mean':np.nan,'artifact_mean':np.nan,
         'temp_coverage':0,'temp_mean':np.nan,'acc_coverage':0,'phases_covered':0,'phase_detail':{}}
    if df_day.empty: return m
    n = len(df_day)
    if 'eda' in df_day.columns:
        eda_v = df_day['eda'].dropna(); eda_nz = eda_v[eda_v > 0.01]
        m['eda_coverage'] = round(len(eda_v)/n*100,1)
        m['eda_zero_pct'] = round((len(eda_v)-len(eda_nz))/max(len(eda_v),1)*100,1)
        if len(eda_nz) > 0:
            m['eda_mean'] = round(eda_nz.mean(),3); m['eda_min'] = round(eda_nz.min(),3); m['eda_max'] = round(eda_nz.max(),3)
    if 'heart_rate' in df_day.columns:
        hr_v = df_day['heart_rate'].dropna(); hr_ph = hr_v[(hr_v>=30)&(hr_v<=220)]
        m['hr_coverage'] = round(len(hr_v)/n*100,1)
        if len(hr_ph) > 0: m['hr_mean'] = round(hr_ph.mean(),1); m['hr_min'] = round(hr_ph.min(),1); m['hr_max'] = round(hr_ph.max(),1)
    hrv_col = 'hrv_td_rmssd' if 'hrv_td_rmssd' in df_day.columns else ('hrv_rmssd' if 'hrv_rmssd' in df_day.columns else None)
    if hrv_col:
        hrv_v = df_day[hrv_col].dropna()
        if len(hrv_v) > 0: m['hrv_mean'] = round(hrv_v.mean(),1)
    if 'hrv_td_sdnn' in df_day.columns:
        v = df_day['hrv_td_sdnn'].dropna()
        if len(v) > 0: m['sdnn_mean'] = round(v.mean(),1)
    if 'hrv_fd_lf' in df_day.columns:
        v = df_day['hrv_fd_lf'].dropna()
        if len(v) > 0: m['lf_mean'] = round(v.mean(),1)
    if 'hrv_fd_hf' in df_day.columns:
        v = df_day['hrv_fd_hf'].dropna()
        if len(v) > 0: m['hf_mean'] = round(v.mean(),1)
    if 'hrv_fd_lf_hf_ratio' in df_day.columns:
        v = df_day['hrv_fd_lf_hf_ratio'].dropna()
        if len(v) > 0: m['lfhf_mean'] = round(v.mean(),2)
    if 'rri_count_valid' in df_day.columns:
        v = df_day['rri_count_valid'].dropna()
        if len(v) > 0: m['rri_valid_mean'] = round(v.mean(),1)
    if 'rri_artifact_pct' in df_day.columns:
        v = df_day['rri_artifact_pct'].dropna()
        if len(v) > 0: m['artifact_mean'] = round(v.mean(),1)
    if 'temperature' in df_day.columns:
        t_v = df_day['temperature'].dropna(); t_ph = t_v[(t_v>25)&(t_v<42)]
        m['temp_coverage'] = round(len(t_v)/n*100,1)
        if len(t_ph) > 0: m['temp_mean'] = round(t_ph.mean(),1)
    if 'vector_magnitude' in df_day.columns:
        m['acc_coverage'] = round(df_day['vector_magnitude'].notna().mean()*100,1)
    ts = df_day['timestamp']
    for ph in PHASES:
        if (pid,ph) not in windows:
            m['phase_detail'][ph] = {'status':'no_window','rows':0,'eda_cov':0,'hr_cov':0}; continue
        s,e = windows[(pid,ph)]
        ph_df = df_day[(ts>=s)&(ts<=e)]
        if len(ph_df)==0:
            m['phase_detail'][ph] = {'status':'no_data','rows':0,'eda_cov':0,'hr_cov':0}; continue
        m['phases_covered'] += 1
        eda_c = round(ph_df['eda'].notna().mean()*100,1) if 'eda' in ph_df else 0
        hr_c = round(ph_df['heart_rate'].notna().mean()*100,1) if 'heart_rate' in ph_df else 0
        m['phase_detail'][ph] = {'status':'ok','rows':len(ph_df),'eda_cov':eda_c,'hr_cov':hr_c,'start':s,'end':e}
    return m


def participant_status(m):
    if m['phases_covered']==0: return 'NO DATA'
    if m['phases_covered']<3 or (m.get('eda_coverage',0) or 0)<20: return 'POOR'
    if m['phases_covered']<5 or (m.get('eda_coverage',0) or 0)<50: return 'FAIR'
    return 'GOOD'


def make_participant_plot(df_day, windows, pid):
    """Full-day time series with phase-shaded backgrounds."""
    plot_df = df_day.sort_values('timestamp').copy()
    full_index = pd.date_range(
        plot_df['timestamp'].min(),
        plot_df['timestamp'].max(),
        freq='10s'
    )
    plot_df = (
        plot_df.set_index('timestamp')
        .reindex(full_index)
        .rename_axis('timestamp')
        .reset_index()
    )
    plot_df['ParticipantID'] = f'P{pid}'
    pairs = [(l,c,col) for l,c,col in SIGNAL_PLOTS if c in plot_df.columns and plot_df[c].notna().any()]
    if not pairs: return ''
    fig, axes = plt.subplots(len(pairs), 1, figsize=(14, 2.2*len(pairs) + 0.8), sharex=True)
    if len(pairs) == 1: axes = [axes]
    fig.suptitle(f'P{pid} \u2014 Empatica Embrace Plus: Full Day (10-sec bins)', fontsize=12, fontweight='bold')
    ts = plot_df['timestamp']
    for ax, (label, col, color) in zip(axes, pairs):
        series = pd.to_numeric(plot_df[col], errors='coerce')
        series_plot = series.to_numpy()
        ax.plot(ts, series_plot, color=color, linewidth=0.7, alpha=0.85)
        ax.set_ylabel(label, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        for ph in PHASES:
            if (pid, ph) in windows:
                s, e = windows[(pid, ph)]
                ax.axvspan(s, e, alpha=0.15, color=PHASE_COLORS.get(PHASE_ID[ph], '#888'))
    handles = [mpatches.Patch(facecolor=PHASE_COLORS[PHASE_ID[ph]], alpha=0.5, label=PHASE_ID[ph]) for ph in PHASES]
    axes[-1].legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.32), fontsize=7, ncol=5, frameon=False)
    axes[-1].set_xlabel('Time (Brussels)', fontsize=8)
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=90, bbox_inches='tight')
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')


def _cell_bg(value, good_t, warn_t, higher_better=True):
    if value is None or (isinstance(value, float) and np.isnan(value)): return ''
    if higher_better:
        if value >= good_t: return 'style="background:#d4edda"'
        if value >= warn_t: return 'style="background:#fff3cd"'
        return 'style="background:#f8d7da"'
    else:
        if value <= good_t: return 'style="background:#d4edda"'
        if value <= warn_t: return 'style="background:#fff3cd"'
        return 'style="background:#f8d7da"'


def fmt(v, d=1, sfx=''):
    if v is None or (isinstance(v, float) and np.isnan(v)): return '\u2014'
    return f'{round(v,d)}{sfx}'


def generate_html(all_metrics, all_plots, generated_at):
    summary_rows = []
    for m in all_metrics:
        pid=m['pid']; st=m['_status']; sc=STATUS_COLOR.get(st,'#888')
        if st=='NO DATA':
            summary_rows.append(f'<tr><td><strong>P{pid}</strong></td><td><span style="background:{sc};color:#fff;padding:2px 8px;border-radius:4px">{st}</span></td><td colspan="12" style="color:#999">\u2014</td></tr>')
        else:
            summary_rows.append(f'''<tr><td><a href="#p{pid}" style="font-weight:bold;color:#2c3e50">P{pid}</a></td><td><span style="background:{sc};color:#fff;padding:2px 8px;border-radius:4px">{st}</span></td><td>{m["total_rows"]}</td><td {_cell_bg(m["eda_coverage"],70,40)}>{fmt(m["eda_coverage"])}%</td><td>{fmt(m["eda_zero_pct"])}%</td><td>{fmt(m["eda_mean"],3)} \u00b5S</td><td {_cell_bg(m["hr_coverage"],60,30)}>{fmt(m["hr_coverage"])}%</td><td>{fmt(m["hr_mean"])} bpm</td><td>{fmt(m["hrv_mean"])} ms</td><td>{fmt(m["sdnn_mean"])} ms</td><td>{fmt(m["lfhf_mean"],2)}</td><td>{fmt(m["temp_mean"])} \u00b0C</td><td {_cell_bg(m["phases_covered"],4,2)}>{m["phases_covered"]}/5</td></tr>''')
    toc_items = []
    for m in all_metrics:
        pid=m['pid']; st=m['_status']; sc=STATUS_COLOR.get(st,'#888')
        if st=='NO DATA':
            toc_items.append(f'<span style="background:{sc};color:#fff;padding:5px 12px;border-radius:20px;font-size:0.85em">P{pid} N/A</span>')
        else:
            toc_items.append(f'<a href="#p{pid}" style="background:{sc};color:#fff;padding:5px 12px;border-radius:20px;font-size:0.85em;text-decoration:none;font-weight:bold">P{pid} {st}</a>')
    participant_checks = []
    for m in all_metrics:
        pid_label = f'P{m["pid"]}'
        checked = ' checked' if pid_label in DEFAULT_VISIBLE_PIDS else ''
        participant_checks.append(f'<label><input type="checkbox" class="participant-cb" data-pid="{pid_label}"{checked}> {pid_label}</label>')
    participant_checks = ''.join(participant_checks)
    participant_sections = []
    for m in all_metrics:
        pid=m['pid']; st=m['_status']; sc=STATUS_COLOR.get(st,'#888')
        pid_label = f'P{pid}'
        section_style = '' if pid_label in DEFAULT_VISIBLE_PIDS else ' style="display:none"'
        plot_checked = ' checked' if pid_label in DEFAULT_VISIBLE_PIDS else ''
        plot_style = '' if pid_label in DEFAULT_VISIBLE_PIDS else ' style="display:none"'
        plot_b64 = all_plots.get(pid,'')
        if st=='NO DATA':
            participant_sections.append(f'''<div id="p{pid}" class="participant-card participant-section" data-pid="{pid_label}"{section_style}><div class="p-header" style="background:{sc}22;border-left:6px solid {sc}"><h2>P{pid} <span class="status-badge" style="background:{sc}">NO DATA</span></h2></div><p style="padding:20px;color:#999">No data in Empatica CSV.</p></div>''')
            continue
        phase_rows = []
        for ph in PHASES:
            ph_id=PHASE_ID[ph]; ph_col=PHASE_COLORS.get(ph_id,'#888')
            pd_info=m['phase_detail'].get(ph,{})
            if pd_info.get('status')=='no_window':
                phase_rows.append(f'<tr><td><span style="background:{ph_col};color:#fff;padding:2px 6px;border-radius:3px;font-size:0.85em">{ph_id}</span></td><td colspan="5" style="color:#aaa">No window in key.csv</td></tr>')
            elif pd_info.get('status')=='no_data':
                phase_rows.append(f'<tr><td><span style="background:{ph_col};color:#fff;padding:2px 6px;border-radius:3px;font-size:0.85em">{ph_id}</span></td><td colspan="5" style="background:#f8d7da;color:#c0392b;font-weight:bold">NO DATA (gap)</td></tr>')
            else:
                s_t=pd_info.get('start',''); e_t=pd_info.get('end','')
                win_str=f'{s_t.strftime("%H:%M") if hasattr(s_t,"strftime") else "?"} \u2013 {e_t.strftime("%H:%M") if hasattr(e_t,"strftime") else "?"}'
                dur_min=round((e_t-s_t).total_seconds()/60,1) if hasattr(s_t,'strftime') else '?'
                eda_c=pd_info.get('eda_cov',0); hr_c=pd_info.get('hr_cov',0)
                phase_rows.append(f'<tr><td><span style="background:{ph_col};color:#fff;padding:2px 6px;border-radius:3px;font-size:0.85em">{ph_id}</span></td><td style="font-size:0.85em;color:#555">{win_str}</td><td>{dur_min} min</td><td>{pd_info.get("rows",0)}</td><td {_cell_bg(eda_c,70,40)}>{eda_c}% EDA valid</td><td {_cell_bg(hr_c,60,30)}>{hr_c}% HR valid</td></tr>')
        plot_html = ''
        if plot_b64:
            plot_html = f'''<div class="card full-width"><h3>Full-Day Signal Time-Series</h3><p class="footnote" style="margin-bottom:8px">Shaded = experiment phases. Lines show available 10-sec values only; gaps are not interpolated in this report.</p><img src="data:image/png;base64,{plot_b64}" style="max-width:100%;border-radius:4px" /></div>'''
        participant_sections.append(f'''<div id="p{pid}" class="participant-card participant-section" data-pid="{pid_label}"{section_style}><div class="p-header" style="background:{sc}22;border-left:6px solid {sc}"><h2>P{pid}<span class="status-badge" style="background:{sc}">{st}</span><span class="meta">{m["total_rows"]} 10-sec rows &nbsp;|&nbsp; {m["date"]}</span></h2></div><div class="grid2"><div class="card"><h3>EDA</h3><table class="metrics"><tr><th>Coverage</th><td {_cell_bg(m["eda_coverage"],70,40)}>{fmt(m["eda_coverage"])}%</td></tr><tr><th>Zero readings</th><td {_cell_bg(m["eda_zero_pct"],10,30,False)}>{fmt(m["eda_zero_pct"])}%</td></tr><tr><th>Mean / Min / Max</th><td>{fmt(m["eda_mean"],3)} / {fmt(m["eda_min"],3)} / {fmt(m["eda_max"],3)} \u00b5S</td></tr></table></div><div class="card"><h3>Heart Rate &amp; HRV</h3><table class="metrics"><tr><th>HR coverage</th><td {_cell_bg(m["hr_coverage"],60,30)}>{fmt(m["hr_coverage"])}%</td></tr><tr><th>HR mean / min / max</th><td>{fmt(m["hr_mean"],1)} / {fmt(m["hr_min"],1)} / {fmt(m["hr_max"],1)} bpm</td></tr><tr><th>RMSSD</th><td>{fmt(m["hrv_mean"],1)} ms</td></tr><tr><th>SDNN</th><td>{fmt(m["sdnn_mean"],1)} ms</td></tr><tr><th>LF / HF / LF:HF</th><td>{fmt(m["lf_mean"],1)} / {fmt(m["hf_mean"],1)} / {fmt(m["lfhf_mean"],2)}</td></tr></table></div><div class="card"><h3>RRI Quality</h3><table class="metrics"><tr><th>Mean valid RRI count</th><td>{fmt(m["rri_valid_mean"],1)}</td></tr><tr><th>Mean artifact %</th><td>{fmt(m["artifact_mean"],1)}%</td></tr><tr><th>Temp coverage</th><td {_cell_bg(m["temp_coverage"],70,40)}>{fmt(m["temp_coverage"])}%</td></tr><tr><th>Skin temp mean</th><td>{fmt(m["temp_mean"],1)} \u00b0C</td></tr><tr><th>Acc coverage</th><td {_cell_bg(m["acc_coverage"],70,40)}>{fmt(m["acc_coverage"])}%</td></tr></table></div></div><div class="card full-width"><h3>Per-Phase Breakdown</h3><table class="phase-table"><thead><tr><th>Phase</th><th>Window (local)</th><th>Duration</th><th>10-sec rows</th><th>EDA coverage</th><th>HR coverage</th></tr></thead><tbody>{"".join(phase_rows)}</tbody></table></div>{plot_html}</div>''')
    html = f'''<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Empatica Quality Report</title><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:"Segoe UI",Arial,sans-serif;font-size:14px;background:#f0f2f5;color:#2c3e50}}
h1{{font-size:1.6em;color:#fff}} h2{{font-size:1.25em;margin-bottom:8px}} h3{{font-size:1em;color:#34495e;margin-bottom:8px;border-bottom:2px solid #e0e0e0;padding-bottom:4px}}
.header-bar{{background:#2c3e50;padding:20px 30px;color:#fff;margin-bottom:20px}}
.header-bar p{{color:#bdc3c7;margin-top:4px;font-size:0.9em}}
.container{{max-width:1400px;margin:0 auto;padding:0 20px 40px}}
.summary-card{{background:#fff;border-radius:8px;padding:20px;box-shadow:0 2px 6px rgba(0,0,0,0.1);margin-bottom:30px}}
table{{width:100%;border-collapse:collapse}}
thead th{{background:#2c3e50;color:#fff;padding:8px 10px;text-align:left;font-size:0.85em;white-space:nowrap}}
tbody tr:nth-child(even){{background:#f8f9fa}}
tbody td,tbody th{{padding:6px 10px;border-bottom:1px solid #e0e0e0;vertical-align:top}}
.participant-card{{background:#fff;border-radius:8px;margin-bottom:30px;box-shadow:0 2px 8px rgba(0,0,0,0.12);overflow:hidden}}
.p-header{{padding:14px 20px}}
.p-header h2{{display:flex;align-items:center;gap:12px;flex-wrap:wrap}}
.status-badge{{font-size:0.65em;padding:3px 10px;border-radius:12px;color:#fff}}
.meta{{font-size:0.65em;font-weight:normal;color:#555}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:16px;padding:16px}}
.card{{background:#fafafa;border:1px solid #e8e8e8;border-radius:6px;padding:14px}}
.full-width{{margin:0 16px 16px}}
table.metrics{{font-size:0.88em}}
table.metrics th{{background:none;color:#555;font-weight:600;padding:5px 8px;width:45%;font-size:0.88em}}
table.metrics td{{padding:5px 8px}}
.footnote{{font-size:0.78em;color:#888;font-style:italic;margin-top:8px;border-top:1px solid #eee;padding-top:6px}}
.toc{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:20px}}
.controls{{background:#fff;border-radius:8px;padding:14px 16px;box-shadow:0 2px 6px rgba(0,0,0,0.1);margin-bottom:20px}}
.controls label{{display:inline-flex;align-items:center;gap:6px;margin:4px 10px 4px 0;font-weight:600}}
.legend{{display:flex;gap:20px;margin-top:12px;font-size:0.82em}}
.legend-item{{display:flex;align-items:center;gap:6px}}
.legend-box{{width:16px;height:16px;border-radius:3px;display:inline-block}}
</style></head><body>
<div class="header-bar"><h1>Empatica Embrace Plus \u2014 Quality Report</h1><p>Generated: {generated_at}</p></div>
<div class="container">
<div class="summary-card"><h2>Summary \u2014 All Participants</h2><table><thead><tr><th>PID</th><th>Status</th><th>10-sec rows</th><th>EDA cov.</th><th>EDA zero%</th><th>EDA mean</th><th>HR cov.</th><th>HR mean</th><th>RMSSD</th><th>SDNN</th><th>LF/HF</th><th>Temp mean</th><th>Phases</th></tr></thead><tbody>{"".join(summary_rows)}</tbody></table><div class="legend"><div class="legend-item"><div class="legend-box" style="background:#d4edda"></div>Within range</div><div class="legend-item"><div class="legend-box" style="background:#fff3cd"></div>Mild concern</div><div class="legend-item"><div class="legend-box" style="background:#f8d7da"></div>Flag</div></div></div>
<div class="controls"><strong>Visible participants:</strong> {participant_checks}</div>
<div class="toc">{"".join(toc_items)}</div>
{"".join(participant_sections)}
<script>
document.querySelectorAll('.participant-cb').forEach(function(cb){{
  cb.addEventListener('change', function(){{
    document.querySelectorAll('.participant-section[data-pid="'+cb.dataset.pid+'"]').forEach(function(el){{ el.style.display = cb.checked ? '' : 'none'; }});
  }});
}});
</script>
</div></body></html>'''
    return html


def main():
    print('  [11_empatica_validate] ...')
    if not CSV_IN.exists(): print(f'ERROR: {CSV_IN} not found'); return
    if not KEY_FILE.exists(): print(f'ERROR: {KEY_FILE} not found'); return
    date_col = 'timestamp'
    header = pd.read_csv(CSV_IN, nrows=0)
    if 'timestamp' not in header.columns and 'Datetime' in header.columns:
        date_col = 'Datetime'
    csv_df = pd.read_csv(CSV_IN, parse_dates=[date_col])
    if 'timestamp' not in csv_df.columns:
        csv_df['timestamp'] = pd.to_datetime(csv_df[date_col])
    if 'eda' not in csv_df.columns and 'empatica__eda_scl_usiemens' in csv_df.columns:
        csv_df['eda'] = pd.to_numeric(csv_df['empatica__eda_scl_usiemens'], errors='coerce')
    if 'heart_rate' not in csv_df.columns:
        print('  [WARN] heart_rate column is missing; HR panels will be blank.')
    for optional_col in ['temperature', 'vector_magnitude']:
        if optional_col not in csv_df.columns:
            csv_df[optional_col] = np.nan
    key = load_key_unique(KEY_FILE)
    windows = build_phase_windows(key)
    pid_to_date = {}
    for _,r in key.iterrows():
        if pd.isna(r.get('Participant_ID')): continue
        pid_to_date[int(r['Participant_ID'])] = parse_key_date(str(r['Date']))
    expected_pids = [f"P{int(pid)}" for pid in key["Participant_ID"].dropna().astype(int).tolist()]
    if ONLY_PARTICIPANTS:
        expected_pids = [pid for pid in expected_pids if pid in ONLY_PARTICIPANTS]
    pids_in_csv = sorted(set(csv_df['ParticipantID'].dropna().astype(str)), key=lambda x: int(x.replace('P','')))
    unexpected = sorted(set(pids_in_csv) - set(expected_pids), key=lambda x: int(x.replace('P','')))
    if unexpected:
        print(f'  [WARN] unexpected participants in source CSV: {unexpected}')
    all_metrics = []; all_plots = {}
    for ps in expected_pids:
        pn = int(ps.replace('P',''))
        date_str = pid_to_date.get(pn, '')
        pw = [(s,e) for (p,ph),(s,e) in windows.items() if p==pn]
        if not pw:
            all_metrics.append({'pid':pn,'date':date_str,'total_rows':0,'eda_coverage':0,'eda_zero_pct':0,
                'eda_mean':np.nan,'eda_min':np.nan,'eda_max':np.nan,'hr_coverage':0,'hr_mean':np.nan,
                'hr_min':np.nan,'hr_max':np.nan,'hrv_mean':np.nan,'sdnn_mean':np.nan,'lf_mean':np.nan,
                'hf_mean':np.nan,'lfhf_mean':np.nan,'rri_valid_mean':np.nan,'artifact_mean':np.nan,
                'temp_coverage':0,'temp_mean':np.nan,'acc_coverage':0,'phases_covered':0,'phase_detail':{},'_status':'NO DATA'})
            continue
        earliest = min(s for s,e in pw); latest = max(e for s,e in pw)
        pdf = csv_df[csv_df['ParticipantID']==ps].copy()
        pdf['timestamp'] = pd.to_datetime(pdf['timestamp']).dt.floor('10s')
        # Duplicates from flooring: keep first.
        pdf = pdf.groupby('timestamp', as_index=False).first()
        day_df = pdf[(pdf['timestamp'] >= earliest.floor('10s')) & (pdf['timestamp'] <= latest.ceil('10s'))].copy()
        day_df['ParticipantID'] = ps
        if day_df.empty:
            all_metrics.append({'pid':pn,'date':date_str,'total_rows':0,'eda_coverage':0,'eda_zero_pct':0,
                'eda_mean':np.nan,'eda_min':np.nan,'eda_max':np.nan,'hr_coverage':0,'hr_mean':np.nan,
                'hr_min':np.nan,'hr_max':np.nan,'hrv_mean':np.nan,'sdnn_mean':np.nan,'lf_mean':np.nan,
                'hf_mean':np.nan,'lfhf_mean':np.nan,'rri_valid_mean':np.nan,'artifact_mean':np.nan,
                'temp_coverage':0,'temp_mean':np.nan,'acc_coverage':0,'phases_covered':0,'phase_detail':{},'_status':'NO DATA'})
            all_plots[pn] = ''
            continue
        m = compute_quality_metrics(day_df, windows, pn, date_str)
        m['_status'] = participant_status(m)
        all_metrics.append(m)
        try:
            all_plots[pn] = make_participant_plot(day_df, windows, pn)
        except Exception as ex:
            print(f'  [WARN] Plot failed for P{pn}: {ex}')
            all_plots[pn] = ''
    all_metrics.sort(key=lambda x: x['pid'])
    html = generate_html(all_metrics, all_plots, datetime.now().strftime('%d %b %Y %H:%M'))
    HTML_OUT.parent.mkdir(parents=True, exist_ok=True)
    HTML_OUT.write_text(html, encoding='utf-8')
    print(f'Saved: {HTML_OUT}')
    print(f'  {len(all_metrics)} participants, {len(csv_df):,} rows in source CSV')

if __name__ == '__main__':
    main()
