import pandas as pd
df = pd.read_csv('C:/Users/pandya/Documents/Github/docker/ExpData/outputs/01_empatica_10sec.csv', low_memory=False)
df['PhaseID'] = df['PhaseID'].fillna('').astype(str)
for pid in sorted(df['ParticipantID'].unique()):
    sub = df[df['ParticipantID']==pid]
    hr_ok = sub['heart_rate'].notna().sum()
    print(f'{pid}: {len(sub)} rows, HR={hr_ok}/{len(sub)} = {hr_ok/len(sub)*100:.1f}%')
# Also check per-phase for problematic ones
for pid in ['P13','P15','P4','P9']:
    sub = df[df['ParticipantID']==pid]
    print(f'\n{pid} per-phase:')
    for ph in sorted(sub['PhaseID'].unique()):
        phd = sub[sub['PhaseID']==ph]
        hr_ok = phd['heart_rate'].notna().sum()
        print(f'  {ph}: {hr_ok}/{len(phd)} = {hr_ok/len(phd)*100:.1f}%')

