#!/usr/bin/python3
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path
from dateutil.parser import parse
from datetime import datetime,  timedelta
from state_convert import abbrev_us_state, us_state_abbrev

df = pd.read_csv('data/us_state_vaccinations.csv')
df = df[df['date'].apply(lambda x: parse(x)) > datetime(2021, 1, 10)]
cols = ["total_vaccinations","total_vaccinations_per_hundred", "people_fully_vaccinated","people_fully_vaccinated_per_hundred", "total_distributed","distributed_per_hundred"]
pd.options.mode.chained_assignment = None
path = Path('covid-vaccine-tracker-data/data/')
data = pd.read_csv(path / 'historical-usa-doses-administered.csv')
data.fillna(0, inplace=True)
diff = parse(data['date'][len(data) - 1]) - parse(data['date'][0])
days = diff.days
data = data.set_index('date')
data = data.pivot(columns='id', values='value')
data.index = pd.Series([datetime(2020, 12, 21) + timedelta(days=k) for k in range(len(data))])
pops = pd.read_csv('data/state_pops.csv')

def get_data(col, state):
    if state == None:
        data = df[['date','location',col]].set_index('date').pivot(columns='location', values=col).fillna(method='ffill').fillna(0)
    else:
        data = df[['date','location',col]].set_index('date').pivot(columns='location', values=col).fillna(method='ffill').fillna(0)[state].fillna(0)

    return data


xs, ys, zs = np.zeros((data.shape[1]+1, len(cols))), np.zeros((data.shape[1]+1, len(cols))), np.zeros((data.shape[1]+1, len(cols)))
data.fillna(0, inplace=True)

def func(x, a, b, c):
    return a + b * x + c * x ** 2

states = ['US'] + data.columns.to_list()

for idy, data_col in enumerate(cols):
    for idx, col in enumerate(states):
        if abbrev_us_state[col] not in df.location.to_list():
            continue
        ma = get_data(data_col, abbrev_us_state[col]).rolling(window=10).mean().fillna(method='ffill').fillna(0)
        try:
            coefs = curve_fit(func, list(range(len(ma))), ma, bounds=((-np.inf,-np.inf,0), (np.inf,np.inf,np.inf)))[0]
            xs[idx, idy], ys[idx, idy], zs[idx, idy] = coefs
        except ValueError:
            print(ma)

np.save('data/xs.npy', xs)
np.save('data/ys.npy', ys)
np.save('data/zs.npy', zs)
