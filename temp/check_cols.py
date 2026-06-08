import pandas as pd
df = pd.read_csv(r"temp\ExperimentData_20Apr_Jun.csv", nrows=1)
targets = ['atmotube_left__atmotube_pm1','atmotube_left__atmotube_pm2.5',
           'atmotube_left__atmotube_pm10','atmotube_left__atmotube_temperature',
           'atmotube_left__atmotube_humidity','LYS1__lys_lux','LYS2__lys_lux',
           'atmotube_right__atmotube_pm1','atmotube_right__atmotube_pm2.5',
           'atmotube_right__atmotube_pm10','atmotube_right__atmotube_temperature',
           'atmotube_right__atmotube_humidity']
for t in targets:
    found = t in df.columns
    print(f'{t}: {"YES" if found else "NO"}')
