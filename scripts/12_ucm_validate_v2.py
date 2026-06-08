"""
12_ucm_validate_v2.py
=====================
Generates quality HTML report for UCM backpack environmental data.
Uses ONE combined figure per participant (all signals) for speed.

Reads:  outputs/02_ucm_10sec.csv (via _paths.py)
Output: outputs/12_ucm_quality_report_v2.html
"""
import warnings; warnings.filterwarnings('ignore')
import base64, io
from datetime import datetime
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _paths import OUTPUTS

CSV_IN   = OUTPUTS / '02_ucm_10sec.csv'
HTML_OUT = OUTPUTS / '12_ucm_quality_report_v2.html'

PHASES = ['BikeU','WalkU','BikeG','WalkG','Tram']
PHASE_COLORS = {'BikeU':'#d45500','WalkU':'#b8860b','BikeG':'#1a6b1a','WalkG':'#52b852','Tram':'#7f8c8d'}

# All UCM environmental signals: (column, label, unit)
ALL_SIGNALS = [
    # Temperature & Humidity
    ('AIR_temp',    'Air temperature',    '°C'),
    ('AIR_T_bot',   'Temp - bottom',      '°C'),
    ('AIR_T_mid',   'Temp - middle',      '°C'),
    ('AIR_T_top',   'Temp - top',         '°C'),
    ('AIR_RH',      'Relative humidity',  '%'),
    ('AH',          'Absolute humidity',  'g/m³'),
    # Radiation & IR
    ('SUN_Gh',      'Solar radiation',    'W/m²'),
    ('SUN_alt',     'Sun altitude',       '°'),
    ('SUN_az',      'Sun azimuth',        '°'),
    ('MRT',         'Mean radiant temp',  '°C'),
    ('MRT_S',       'MRT (simplified)',   '°C'),
    ('OPT',         'Operative temp',     '°C'),
    ('IR_up',       'IR upward',          'W/m²'),
    ('IR_down',     'IR downward',        'W/m²'),
    ('IR_front',    'IR front',           'W/m²'),
    ('IR_back',     'IR back',            'W/m²'),
    ('IR_left',     'IR left',            'W/m²'),
    ('IR_right',    'IR right',           'W/m²'),
    ('IR_spot_gnd', 'IR spot - ground',   'W/m²'),
    ('IR_spot_sky', 'IR spot - sky',      'W/m²'),
    ('IR_spot_left','IR spot - left',     'W/m²'),
    ('IR_spot_right','IR spot - right',   'W/m²'),
    # Sound
    ('SND_dBA',     'Sound level',        'dBA'),
    # Air Quality - Particulates
    ('AQ_pm010',    'PM1.0',              'µg/m³'),
    ('AQ_pm025',    'PM2.5',              'µg/m³'),
    ('AQ_pm100',    'PM10',               'µg/m³'),
    # Air Quality - Gases
    ('AQ_CO2',      'CO₂',                'ppm'),
    ('AQ_NO2_m1',   'NO₂',                'ppb'),
    ('AQ_O3_m1',    'O₃',                 'ppb'),
    ('AQ_SO2_m4',   'SO₂',                'ppb'),
    ('AQ_CO_m1',    'CO',                 'ppb'),
    # Air Quality - Indices
    ('AQ1',         'AQI 1',              ''),
    ('AQ2',         'AQI 2',              ''),
    ('AQ3',         'AQI 3',              ''),
    # Wind
    ('WIND_AWS',    'Wind speed (avg)',   'm/s'),
    ('WIND_TWS',    'Wind speed (true)',  'm/s'),
    ('WIND_AWA',    'Wind direction (avg)','°'),
    ('WIND_TWD',    'Wind direction (true)','°'),
    ('MAG_hdg',     'Magnetic heading',   '°'),
    # Thermal Comfort
    ('UTCI',        'UTCI',               '°C'),
    ('UTCI_SR',     'UTCI (solar rad)',   '°C'),
    ('PET',         'PET',                '°C'),
    ('PET_SR',      'PET (solar rad)',    '°C'),
    ('HUMIDEX',     'Humidex',            ''),
    ('HUMIDEX_SR',  'Humidex (solar rad)',''),
]

def status_badge(pct_nan):
    if pct_nan>=80: return f'<span class="badge" style="background:#c0392b">{pct_nan:.0f}% NaN</span>'
    if pct_nan>=50: return f'<span class="badge" style="background:#e67e22">{pct_nan:.0f}% NaN</span>'
    return f'<span class="badge" style="background:#27ae60">{pct_nan:.0f}% NaN</span>'

def make_combined_plot(pid_str, df_pid):
    """One tall figure per participant — all signals, all phases (fast)."""
    n_sig = len(ALL_SIGNALS)
    fig, axes = plt.subplots(n_sig, 5, figsize=(12, 0.7 * n_sig + 0.5),
                             squeeze=False)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.97, bottom=0.02,
                        hspace=0.06, wspace=0.08)

    for row, (col_name, label, unit) in enumerate(ALL_SIGNALS):
        for col, ph in enumerate(PHASES):
            ax = axes[row, col]
            if col == 0:
                u = f' [{unit}]' if unit else ''
                ax.set_ylabel(f'{label}{u}', fontsize=4.5)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=3.5, length=1.5, pad=0.5)

            pdf = df_pid[df_pid['PhaseID'] == ph].sort_values('Datetime')
            if pdf.empty or col_name not in pdf.columns:
                ax.set_facecolor('#ececec')
                continue

            t0 = pdf['Datetime'].iloc[0]
            xm = (pdf['Datetime'] - t0).dt.total_seconds() / 60.0
            vals = pdf[col_name].values.astype(float)
            nm = np.isnan(vals)

            ax.plot(xm[~nm], vals[~nm], color='#2E86AB', linewidth=0.5)
            valid = vals[~nm]
            if len(valid) > 0:
                lo, hi = np.nanpercentile(valid, 2), np.nanpercentile(valid, 98)
                margin = max((hi - lo) * 0.1, 0.01)
                ax.set_ylim(lo - margin, hi + margin)
            if nm.any():
                ax.scatter(xm[nm], np.full(nm.sum(), ax.get_ylim()[0]),
                           color='#e74c3c', marker='|', s=4, linewidths=0.3,
                           alpha=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=60)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')

def generate_html():
    print('  [12_ucm_validate_v2] ...')
    df = pd.read_csv(CSV_IN, parse_dates=['Datetime'])
    pids = sorted(df['ParticipantID'].dropna().unique(),
                  key=lambda x: int(x.replace('P', '')))
    print(f'  {len(pids)} participants, {len(df):,} rows')

    html = [f'''<!DOCTYPE html><html><head><meta charset="UTF-8">
    <title>UCM Environmental Quality Report</title>
    <style>
    body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:20px;background:#f5f5f5}}
    .header{{background:#2c3e50;color:#fff;padding:20px;border-radius:5px;margin-bottom:20px}}
    .header h1{{margin:0;font-size:1.6em}}.header p{{margin:5px 0 0;opacity:0.8}}
    table{{border-collapse:collapse;width:100%;margin:15px 0;font-size:0.8em}}
    th{{background:#2c3e50;color:#fff;padding:4px 6px;text-align:center;white-space:nowrap}}
    td{{padding:3px 6px;border-bottom:1px solid #ddd;text-align:center}}tr:hover{{background:#f0f0f0}}
    .badge{{display:inline-block;padding:2px 6px;border-radius:4px;color:#fff;font-weight:bold;font-size:0.8em}}
    .section{{background:#fff;padding:15px;margin-bottom:20px;border-radius:5px;box-shadow:0 1px 3px rgba(0,0,0,0.1)}}
    .section h2{{color:#2c3e50;border-bottom:2px solid #3498db;padding-bottom:8px;margin-top:0}}
    .plot-img{{width:100%;border:1px solid #ccc;border-radius:4px;margin-top:10px}}
    .footer{{text-align:center;color:#999;margin:30px 0}}
    </style></head><body>
    <div class="header"><h1>UCM Backpack — Environmental Quality Report</h1>
    <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")} | {len(pids)} participants, {len(df):,} rows | {len(ALL_SIGNALS)} signals</p>
    </div>''']

    for pid in pids:
        pdf_all = df[df['ParticipantID'] == pid]
        # Only count rows within the 5 experiment phases (exclude between-phase gaps)
        pdf = pdf_all[pdf_all['PhaseID'].isin(PHASES)].copy()
        html.append(f'<div class="section"><h2>{pid}</h2>')
        # Stats table
        html.append('<table><tr><th>Signal</th><th>Count</th><th>NaN%</th>'
                    '<th>Mean</th><th>Min</th><th>Max</th><th>Q25</th><th>Q75</th></tr>')
        for col_name, label, _ in ALL_SIGNALS:
            if col_name not in pdf.columns:
                continue
            cd = pdf[col_name]
            val = cd.dropna()
            n_tot = len(cd)
            n_val = len(val)
            pct_nan = 100 * (1 - n_val / n_tot) if n_tot else 100
            mn = val.mean()
            mi = val.min()
            mx = val.max()
            q25 = val.quantile(0.25)
            q75 = val.quantile(0.75)
            html.append(f'<tr><td>{label}</td><td>{n_val}/{n_tot}</td>'
                        f'<td>{status_badge(pct_nan)}</td>'
                        f'<td>{mn:.2f}</td><td>{mi:.2f}</td><td>{mx:.2f}</td>'
                        f'<td>{q25:.2f}</td><td>{q75:.2f}</td></tr>')
        html.append('</table>')
        # One combined plot
        b64 = make_combined_plot(pid, pdf)
        html.append(f'<img class="plot-img" src="data:image/png;base64,{b64}">')
        html.append('</div>')

    html.append(f'<div class="footer">Report end — {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>'
                '</body></html>')
    HTML_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(HTML_OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))
    print(f'Saved: {HTML_OUT}')

if __name__ == '__main__':
    generate_html()
