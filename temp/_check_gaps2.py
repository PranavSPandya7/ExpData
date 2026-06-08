import pandas as pd
# Check if old v2 used key_full.csv
import os
print('key_full.csv exists:', os.path.exists('C:/Users/pandya/OneDrive - UCL/Pranav PhD/Paper 3/Data Processing_05May/key_full.csv'))
print()

# Our key.csv participants
key = pd.read_csv('C:/Users/pandya/Documents/Github/docker/ExpData/metadata/key.csv')
print('Our key.csv participants:', sorted(key['Participant_ID'].astype(int).tolist()))
print()

# Check a specific case: P13 BikeU - why 0%?
df = pd.read_csv('C:/Users/pandya/Documents/Github/docker/ExpData/outputs/01_empatica_10sec.csv', low_memory=False)
p13 = df[df['ParticipantID']=='P13']
p13_bikeu = p13[p13['PhaseID']=='BikeU']
print('P13 BikeU in our output:')
print(f'  {len(p13_bikeu)} rows')
print(f'  eda non-null: {p13_bikeu["eda"].notna().sum()}')
print(f'  bvp non-null: {p13_bikeu["bvp"].notna().sum()}')
print(f'  hr non-null: {p13_bikeu["heart_rate"].notna().sum()}')
print()

# Old v2: P13 BikeU
old = pd.read_csv('C:/Users/pandya/OneDrive - UCL/Pranav PhD/Paper 3/Data Processing_05May/outputs/empatica_corrected_10sec_v2.csv', low_memory=False)
o13 = old[old['ParticipantID']=='P13']
o13_bikeu = o13[o13['PhaseID']=='BikeU']
print('P13 BikeU in old v2:')
print(f'  {len(o13_bikeu)} rows')
print(f'  eda non-null: {o13_bikeu["eda"].notna().sum()}')
print(f'  bvp non-null: {o13_bikeu["bvp"].notna().sum()}')
print(f'  hr non-null: {o13_bikeu["heart_rate"].notna().sum()}')
