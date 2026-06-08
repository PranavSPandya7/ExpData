"""
13_atmo_lys_validate.py
=======================
Generates quality HTML report for Atmotube + LYS data with auto-scaled y-axis.

Reads:  outputs/03_atmo_lys_merged.csv (via _paths.py)
Output: outputs/03_atmo_lys_quality_report_v2.html
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
from _paths import OUTPUTS, KEY_FILE

CSV_IN   = OUTPUTS / '03_atmo_lys_merged.csv'
HTML_OUT = OUTPUTS / '13_atmo_lys_quality_report.html'

PHASES = ['BikeU','WalkU','BikeG','WalkG','Tram']
PHASE_COLORS = {'BikeU':'#d45500','WalkU':'#b8860b','BikeG':'#1a6b1a','WalkG':'#52b852','Tram':'#7f8c8d'}

# (sensor_prefix, column_suffix, label, unit) - y ranges computed dynamically
PLOT_SIGNALS = [
    ('atmotube_left','atmotube_pm1','PM1 left','ug/m3'),
    ('atmotube_right','atmotube_pm1','PM1 right','ug/m3'),
    ('atmotube_left','atmotube_pm2.5','PM2.5 left','ug/m3'),
    ('atmotube_right','atmotube_pm2.5','PM2.5 right','ug/m3'),
    ('atmotube_left','atmotube_temperature','Temp left','C'),
    ('atmotube_right','atmotube_temperature','Temp right','C'),
    ('atmotube_left','atmotube_humidity','Humidity left','%'),
    ('atmotube_right','atmotube_humidity','Humidity right','%'),
    ('LYS1','lys_lux','LYS1 Lux','lux'),
    ('LYS2','lys_lux','LYS2 Lux','lux'),
]

def status_badge(pct_nan):
    if pct_nan>=50: return f'<span class="badge" style="background:#c0392b">{pct_nan:.0f}% NaN</span>'
    if pct_nan>=20: return f'<span class="badge" style="background:#e67e22">{pct_nan:.0f}% NaN</span>'
    return f'<span class="badge" style="background:#27ae60">{pct_nan:.0f}% NaN</span>'

def make_phase_plot(pid_str, df_pid):
    """Phase-aligned signal grid with auto-scaled y-axis based on actual data range."""
    n_sig,n_ph = len(PLOT_SIGNALS),len(PHASES)
    fig = plt.figure(figsize=(18,1.8*n_sig+0.8))
    fig.suptitle(f'{pid_str} - Phase-Aligned Atmotube & LYS',fontsize=11,fontweight='bold',y=0.98)
    gs = gridspec.GridSpec(n_sig,n_ph,figure=fig,hspace=0.10,wspace=0.12,left=0.07,right=0.99,top=0.93,bottom=0.06)
    for col,ph in enumerate(PHASES):
        pc = PHASE_COLORS.get(ph,'#888')
        ax = fig.add_axes([gs[0,col].get_position(fig).x0,0.945,gs[0,col].get_position(fig).width,0.025])
        ax.set_xlim(0,1);ax.set_ylim(0,1);ax.add_patch(plt.Rectangle((0,0),1,1,color=pc,transform=ax.transAxes))
        ax.text(0.5,0.5,ph,ha='center',va='center',color='white',fontsize=8,fontweight='bold',transform=ax.transAxes);ax.axis('off')
    for row,(pre,csfx,label,unit) in enumerate(PLOT_SIGNALS):
        col_name = f'{pre}__{csfx}'
        for col,ph in enumerate(PHASES):
            ax = fig.add_subplot(gs[row,col])
            if col==0: ax.set_ylabel(f'{label}\n({unit})',fontsize=6,labelpad=3)
            else: ax.set_yticklabels([])
            if row==n_sig-1: ax.set_xlabel('min',fontsize=6)
            else: ax.set_xticklabels([])
            ax.tick_params(labelsize=5.5);ax.grid(True,alpha=0.25,linewidth=0.5)
            ph_df = df_pid[df_pid['PhaseID']==ph].copy().sort_values('Datetime')
            if ph_df.empty or col_name not in ph_df.columns:
                ax.set_facecolor('#f0f0f0');ax.text(0.5,0.5,'no data',ha='center',va='center',fontsize=7,color='#aaa',transform=ax.transAxes);continue
            t0 = ph_df['Datetime'].iloc[0]
            xm = (ph_df['Datetime']-t0).dt.total_seconds()/60.0
            vals = ph_df[col_name].values.astype(float); nm = np.isnan(vals)
            clr = '#1565C0' if 'left' in pre else '#E53935'
            if 'LYS' in pre: clr = '#FB8C00'
            ax.plot(xm[~nm],vals[~nm],color=clr,linewidth=0.9,alpha=0.85)
            # Auto-scale y-axis: use 2nd-98th percentile of valid data
            valid = vals[~nm]
            if len(valid)>0:
                lo,hi = np.nanpercentile(valid,2),np.nanpercentile(valid,98)
                margin = max((hi-lo)*0.2,0.01)
                # Add extra margin for LYS lux which can have spikes
                if 'LYS' in pre: margin = max(margin, hi*0.3)
                ax.set_ylim(max(0,lo-margin), hi+margin)
            if nm.any():
                ax.scatter(xm[nm],np.full(nm.sum(),ax.get_ylim()[0]),color='#e74c3c',marker='|',s=12,linewidths=0.7,zorder=5,alpha=0.7)
            ax.set_facecolor(PHASE_COLORS.get(ph,'#888')+'0a')
            cov = (1-nm.mean())*100
            ax.text(0.99,0.97,f'{cov:.0f}%',ha='right',va='top',fontsize=5,color='#555',transform=ax.transAxes)
    buf=io.BytesIO();fig.savefig(buf,format='png',dpi=100,bbox_inches='tight');plt.close(fig);buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')

def generate_html():
    print('  [13_atmo_lys_validate] ...')
    df = pd.read_csv(CSV_IN, parse_dates=['Datetime'])
    pids = sorted(df['ParticipantID'].dropna().unique(), key=lambda x: int(x.replace('P','')))
    print(f'  {len(pids)} participants, {len(df):,} rows')
    html = [f'''<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Atmotube & LYS Quality Report</title>
<style>
  body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:20px;background:#f5f5f5}}
  .header{{background:#2c3e50;color:#fff;padding:20px;border-radius:5px;margin-bottom:20px}}
  .header h1{{margin:0;font-size:1.6em}}.header p{{margin:5px 0 0;opacity:0.8}}
  table{{border-collapse:collapse;width:100%;margin:15px 0;font-size:0.85em}}
  th{{background:#2c3e50;color:#fff;padding:6px 8px;text-align:center;white-space:nowrap}}
  td{{padding:5px 8px;border-bottom:1px solid #ddd;text-align:center}}
  tr:hover{{background:#f0f0f0}}
  .badge{{display:inline-block;padding:2px 8px;border-radius:4px;color:#fff;font-weight:bold;font-size:0.85em}}
  .section{{background:#fff;padding:15px;margin-bottom:20px;border-radius:5px;box-shadow:0 1px 3px rgba(0,0,0,0.1)}}
  .section h2{{color:#2c3e50;border-bottom:2px solid #3498db;padding-bottom:8px;margin-top:0}}
  .plot-img{{width:100%;border:1px solid #ccc;border-radius:4px;margin-top:10px}}
  .na{{color:#999;text-align:center}}.footer{{text-align:center;color:#999;margin:30px 0}}
</style></head><body>
<div class="header"><h1>Atmotube &amp; LYS - Quality Report</h1><p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p></div>''']
    for pid in pids:
        pdf_all = df[df['ParticipantID']==pid]
        # Only count rows within the 5 experiment phases (exclude between-phase gaps)
        pdf = pdf_all[pdf_all['PhaseID'].isin(PHASES)].copy()
        html.append(f'<div class="section"><h2>{pid}</h2>')
        html.append('<table><tr><th>Signal</th><th>Count</th><th>NaN%</th><th>Mean</th><th>Min</th><th>Max</th><th>Q25</th><th>Q75</th></tr>')
        for pre,csfx,label,_ in PLOT_SIGNALS:
            cn = f'{pre}__{csfx}'
            if cn not in pdf.columns: continue
            cd = pdf[cn]; val = cd.dropna()
            n_tot = len(cd); n_val = len(val)
            pct_nan = 100*(1-n_val/n_tot) if n_tot else 100
            mn = val.mean(); mi = val.min(); mx = val.max(); q25 = val.quantile(0.25); q75 = val.quantile(0.75)
            html.append(f'<tr><td>{label}</td><td>{n_val}/{n_tot}</td><td>{status_badge(pct_nan)}</td><td>{mn:.2f}</td><td>{mi:.2f}</td><td>{mx:.2f}</td><td>{q25:.2f}</td><td>{q75:.2f}</td></tr>')
        html.append('</table>')
        html.append(f'<img class="plot-img" src="data:image/png;base64,{make_phase_plot(pid,pdf)}">')
        html.append('</div>')
    html.append(f'<div class="footer">Report end - {datetime.now().strftime("%Y-%m-%d %H:%M")}</div></body></html>')
    HTML_OUT.parent.mkdir(parents=True,exist_ok=True)
    with open(HTML_OUT,'w',encoding='utf-8') as f: f.write('\n'.join(html))
    print(f'Saved: {HTML_OUT}')

if __name__ == '__main__':
    generate_html()
