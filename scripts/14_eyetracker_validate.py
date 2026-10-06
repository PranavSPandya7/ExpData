"""
14_eyetracker_validate.py
=========================
Generates quality HTML report for Pupil Labs Neon eyetracker data.

Reads:  Paper3_Github/output/04_eyetracker_10sec.csv (via _paths.py)
Output: Paper3_Github/output/14_eyetracker_quality_report.html
"""
import warnings; warnings.filterwarnings('ignore')
import base64, io
from datetime import datetime
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from _paths import KEY_FILE, OUTPUTS, load_key_unique

CSV_IN   = OUTPUTS / '04_eyetracker_10sec.csv'
HTML_OUT = OUTPUTS / '14_eyetracker_quality_report.html'
DEFAULT_VISIBLE_PIDS = {f'P{p}' for p in [4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]}

PHASES = ['BikeU','WalkU','BikeG','WalkG','Tram']
PHASE_COLORS = {'BikeU':'#d45500','WalkU':'#b8860b','BikeG':'#1a6b1a','WalkG':'#52b852','Tram':'#7f8c8d'}

PLOT_SIGNALS = [
    ('pupil_diameter_avg',  'Pupil diameter',  'mm'),
    ('stress_composite',    'Stress composite', ''),
    ('saccade_amplitude',   'Saccade ampl.',   'deg'),
    ('gaze_dispersion',     'Gaze dispersion',  ''),
    ('fixation_duration_s', 'Fixation dur.',   's'),
]

def status_badge(pct_nan):
    if pct_nan>=80: return f'<span class="badge" style="background:#c0392b">{pct_nan:.0f}% NaN</span>'
    if pct_nan>=50: return f'<span class="badge" style="background:#e67e22">{pct_nan:.0f}% NaN</span>'
    return f'<span class="badge" style="background:#27ae60">{pct_nan:.0f}% NaN</span>'

def make_phase_plot(pid_str, df_pid):
    n_sig,n_ph = len(PLOT_SIGNALS),len(PHASES)
    fig = plt.figure(figsize=(18,1.8*n_sig+0.8))
    fig.suptitle(f'{pid_str} - Phase-Aligned Eyetracker',fontsize=11,fontweight='bold',y=0.98)
    gs = gridspec.GridSpec(n_sig,n_ph,figure=fig,hspace=0.10,wspace=0.12,left=0.07,right=0.99,top=0.93,bottom=0.06)
    for col,ph in enumerate(PHASES):
        pc = PHASE_COLORS.get(ph,'#888')
        ax = fig.add_axes([gs[0,col].get_position(fig).x0,0.945,gs[0,col].get_position(fig).width,0.025])
        ax.set_xlim(0,1);ax.set_ylim(0,1);ax.add_patch(plt.Rectangle((0,0),1,1,color=pc,transform=ax.transAxes))
        ax.text(0.5,0.5,ph,ha='center',va='center',color='white',fontsize=8,fontweight='bold',transform=ax.transAxes);ax.axis('off')
    for row,(col_name,label,unit) in enumerate(PLOT_SIGNALS):
        for col,ph in enumerate(PHASES):
            ax = fig.add_subplot(gs[row,col])
            yl = f'{label}\n({unit})' if unit else label
            if col==0: ax.set_ylabel(yl,fontsize=6.5,labelpad=3)
            else: ax.set_yticklabels([])
            if row==n_sig-1: ax.set_xlabel('min',fontsize=6)
            else: ax.set_xticklabels([])
            ax.tick_params(labelsize=5.5);ax.grid(True,alpha=0.25,linewidth=0.5)
            pdf = df_pid[df_pid['PhaseID']==ph].copy().sort_values('Datetime')
            if pdf.empty or col_name not in pdf.columns:
                ax.set_facecolor('#f0f0f0');ax.text(0.5,0.5,'no data',ha='center',va='center',fontsize=7,color='#aaa',transform=ax.transAxes);continue
            t0=pdf['Datetime'].iloc[0];xm=(pdf['Datetime']-t0).dt.total_seconds()/60.0;vals=pdf[col_name].values.astype(float);nm=np.isnan(vals)
            clr='#8E24AA'
            ax.plot(xm[~nm],vals[~nm],color=clr,linewidth=0.9,alpha=0.85)
            valid=vals[~nm]
            if len(valid)>0:
                lo,hi=np.nanpercentile(valid,2),np.nanpercentile(valid,98)
                mx=max((hi-lo)*0.2,0.01);ax.set_ylim(lo-mx,hi+mx)
            if nm.any(): ax.scatter(xm[nm],np.full(nm.sum(),ax.get_ylim()[0]),color='#e74c3c',marker='|',s=12,linewidths=0.7,zorder=5,alpha=0.7)
            ax.set_facecolor(PHASE_COLORS.get(ph,'#888')+'0a')
            cov=(1-nm.mean())*100;ax.text(0.99,0.97,f'{cov:.0f}%',ha='right',va='top',fontsize=5,color='#555',transform=ax.transAxes)
    buf=io.BytesIO();fig.savefig(buf,format='png',dpi=100,bbox_inches='tight');plt.close(fig);buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')

def generate_html():
    print('  [14_eyetracker_validate] ...')
    df=pd.read_csv(CSV_IN,parse_dates=['Datetime'])
    key=load_key_unique(KEY_FILE).copy()
    key["Participant_ID"]=key["Participant_ID"].astype(int)
    pids=[f"P{pid}" for pid in key["Participant_ID"].tolist()]
    observed_pids=set(df['ParticipantID'].dropna().astype(str))
    unexpected=sorted(observed_pids-set(pids),key=lambda x:int(x.replace('P','')))
    if unexpected:
        print(f'  [WARN] unexpected participants in eyetracker CSV: {unexpected}')
    print(f'  {len(pids)} expected participants, {len(observed_pids)} observed, {len(df):,} rows')
    participant_checks=''.join(
        f'<label><input type="checkbox" class="participant-cb" data-pid="{pid}"{" checked" if pid in DEFAULT_VISIBLE_PIDS else ""}> {pid}</label>'
        for pid in pids
    )
    html=[f'''<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Eyetracker Quality Report</title>
<style>
  body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:20px;background:#f5f5f5}}
  .header{{background:#2c3e50;color:#fff;padding:20px;border-radius:5px;margin-bottom:20px}}
  .header h1{{margin:0;font-size:1.6em}}.header p{{margin:5px 0 0;opacity:0.8}}
  table{{border-collapse:collapse;width:100%;margin:15px 0;font-size:0.85em}}
  th{{background:#2c3e50;color:#fff;padding:6px 8px;text-align:center;white-space:nowrap}}
  td{{padding:5px 8px;border-bottom:1px solid #ddd;text-align:center}}tr:hover{{background:#f0f0f0}}
  .badge{{display:inline-block;padding:2px 8px;border-radius:4px;color:#fff;font-weight:bold;font-size:0.85em}}
  .section{{background:#fff;padding:15px;margin-bottom:20px;border-radius:5px;box-shadow:0 1px 3px rgba(0,0,0,0.1)}}
  .section h2{{color:#2c3e50;border-bottom:2px solid #3498db;padding-bottom:8px;margin-top:0}}
  .controls{{background:#fff;padding:12px 15px;margin-bottom:20px;border-radius:5px;box-shadow:0 1px 3px rgba(0,0,0,0.1)}}
  .controls label{{display:inline-flex;align-items:center;gap:6px;margin:4px 12px 4px 0;font-weight:600}}
  .plot-img{{width:100%;border:1px solid #ccc;border-radius:4px;margin-top:10px}}.na{{color:#999;text-align:center}}
  .footer{{text-align:center;color:#999;margin:30px 0}}
</style></head><body>
<div class="header"><h1>Pupil Labs Neon - Eyetracker Quality Report</h1><p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p></div>
<div class="controls"><strong>Visible participants:</strong> {participant_checks}</div>''']
    for pid in pids:
        visible = pid in DEFAULT_VISIBLE_PIDS
        section_style = '' if visible else ' style="display:none"'
        pdf_all=df[df['ParticipantID']==pid]
        pdf=pdf_all[pdf_all['PhaseID'].isin(PHASES)].copy()
        html.append(f'<div class="section participant-section" data-pid="{pid}"{section_style}><h2>{pid}</h2>')
        if pdf.empty:
            html.append('<p class="na">No eyetracker rows found for this key participant.</p>')
            html.append('</div>')
            continue
        html.append('<table><tr><th>Signal</th><th>Count</th><th>NaN%</th><th>Mean</th><th>Min</th><th>Max</th><th>Q25</th><th>Q75</th></tr>')
        for col_name,label,_ in PLOT_SIGNALS:
            if col_name not in pdf.columns: continue
            cd=pdf[col_name];val=cd.dropna();n_tot=len(cd);n_val=len(val)
            pct_nan=100*(1-n_val/n_tot) if n_tot else 100
            mn=val.mean();mi=val.min();mx=val.max();q25=val.quantile(0.25);q75=val.quantile(0.75)
            html.append(f'<tr><td>{label}</td><td>{n_val}/{n_tot}</td><td>{status_badge(pct_nan)}</td><td>{mn:.3f}</td><td>{mi:.3f}</td><td>{mx:.3f}</td><td>{q25:.3f}</td><td>{q75:.3f}</td></tr>')
        html.append('</table>')
        html.append(f'<img class="plot-img" src="data:image/png;base64,{make_phase_plot(pid,pdf)}">')
        html.append('</div>')
    html.append(f'''<div class="footer">Report end - {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
<script>
document.querySelectorAll('.participant-cb').forEach(function(cb){{
  cb.addEventListener('change', function(){{
    document.querySelectorAll('.participant-section[data-pid="'+cb.dataset.pid+'"]').forEach(function(el){{ el.style.display = cb.checked ? '' : 'none'; }});
  }});
}});
</script></body></html>''')
    HTML_OUT.parent.mkdir(parents=True,exist_ok=True)
    with open(HTML_OUT,'w',encoding='utf-8') as f: f.write('\n'.join(html))
    print(f'Saved: {HTML_OUT}')

if __name__=='__main__':
    generate_html()
