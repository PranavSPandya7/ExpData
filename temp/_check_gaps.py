import pandas as pd
df = pd.read_csv('C:/Users/pandya/Documents/Github/docker/ExpData/outputs/01_empatica_10sec.csv', low_memory=False)
df['PhaseID'] = df['PhaseID'].fillna('').astype(str)
for pid in ['P13','P15','P4','P9']:
    sub = df[df['ParticipantID']==pid]
    print(f'=== {pid} === total rows: {len(sub)}')
    for ph in sorted(sub['PhaseID'].unique()):
        ph_df = sub[sub['PhaseID']==ph]
        hr = ph_df['heart_rate'].dropna()
        print(f'  Phase="{ph}": {len(hr)}/{len(ph_df)} = {len(hr)/len(ph_df)*100:.1f}%')
    print()
