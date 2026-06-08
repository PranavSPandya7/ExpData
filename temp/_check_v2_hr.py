import pandas as pd
old = pd.read_csv('C:/Users/pandya/OneDrive - UCL/Pranav PhD/Paper 3/Data Processing_05May/outputs/empatica_corrected_10sec_v2.csv', low_memory=False)
for pid in ['P13','P15','P4','P9']:
    sub = old[old['ParticipantID']==pid]
    print(f'=== {pid} === total rows: {len(sub)}')
    for ph in sorted(sub['PhaseID'].unique()):
        ph_df = sub[sub['PhaseID']==ph]
        hr = ph_df['heart_rate'].dropna()
        print(f'  Phase=\"{ph}\": {len(hr)}/{len(ph_df)} = {len(hr)/len(ph_df)*100:.1f}%')
    print()
