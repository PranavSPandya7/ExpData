import pandas as pd
f = r'C:\Users\pandya\OneDrive - UCL\Field experiment raw data\Complete Participantwise data\P4\BikeU\ucm\20250825_124032\data.csv'
with open(f) as fh:
    for i,line in enumerate(fh):
        if line.startswith('#') and 'GPS_time' in line:
            skip=i+2
            cols=[c.strip() for c in line.lstrip('# ').split(',')]
            print(f'Header line {i}, skip {skip}, {len(cols)} cols')
            print(f'First cols: {cols[:5]}')
            break
df=pd.read_csv(f,skiprows=skip,header=None,names=cols,encoding='utf-8',engine='python',na_values=['nan',''])
gt=df['GPS_time']
print(f'Read {len(df)} rows')
print(f'GPS_time: {gt.iloc[0]} to {gt.iloc[-1]}')
print(f'Null GPS_time: {gt.isna().sum()}')
