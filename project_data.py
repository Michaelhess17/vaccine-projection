#!/usr/bin/python3
import pandas as pd
import numpy as np
import collections
from pathlib import Path
import plotly.express as px
import plotly.graph_objs as go
from state_convert import us_state_abbrev, abbrev_us_state
from dateutil.parser import parse
from datetime import date, timedelta, datetime
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None
path = Path('covid-vaccine-tracker-data/data/')
data = pd.read_csv(path/'historical-usa-doses-administered.csv')
data.fillna(0, inplace=True)
diff = parse(data['date'][len(data)-1]) - parse(data['date'][0])
days = diff.days
data = data.set_index('date')
data = data.pivot(columns='id', values='value')
data.index = pd.Series([datetime(2020, 12, 21) + timedelta(days=k) for k in range(len(data))])
pops = pd.read_csv('data/state_pops.csv')
num_to_index = {i: datetime(2020, 12, 21) + timedelta(days=k) for i, k in enumerate(range(days+100))}
index_to_num = dict(map(reversed, num_to_index.items()))
xs,ys,zs  = np.zeros(data.shape[1]), np.zeros(data.shape[1]), np.zeros(data.shape[1])
data.fillna(0, inplace=True)
for idx, col in enumerate(data.columns):
    data[col+'ma'] = data[col].rolling(window=10).mean().fillna(0)
    coefs = np.polyfit(range(10,len(data)), data[col+'ma'][10:], 2)
    xs[idx],ys[idx],zs[idx] = coefs
    
most_current = index_to_num[data.index[-1]]
ds = list(range(most_current, most_current+50))
ds_days = [num_to_index[x] for x in ds]
all_days = np.arange(num_to_index[0], num_to_index[140-[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[],timedelta(1))


# Plot the data
def update_graph(state):
    idx = np.argmax(data.columns == state)
    extrap = np.poly1d((xs[idx],ys[idx],zs[idx]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index[:days], y=data[state][:days],mode='lines', name='Data to Date'))
    fig.add_trace(go.Scatter(x=ds_days, y=extrap(ds),mode='lines', name='Projected Data'))
    pop = pops[pops['State']==abbrev_us_state[state]]['Pop'].to_numpy()[0]

    time_to_min_imm = (extrap - (pop*0.75)).roots[0]
    time_to_max_imm = (extrap - (pop*0.85)).roots[0]
    t_min = datetime(2020, 12, 21) + timedelta(days=round(time_to_min_imm))
    t_max = datetime(2020, 12, 21) + timedelta(days=round(time_to_max_imm))
    fig.add_trace(go.Scatter(x=all_days, y=[pop*0.75]*len(all_days), mode='lines', name='75% of Pop.'))
    fig.add_trace(go.Scatter(x=all_days, y=[pop*0.85]*len(all_days), mode='lines', name='85% of Pop.'))
    fig.add_trace(go.Scatter(x=[t_min]*30, y=np.linspace(0, extrap(time_to_min_imm), 30),mode='lines', name='Time to 75% immunity'))
    fig.add_trace(go.Scatter(x=[t_max]*30, y=np.linspace(0, extrap(time_to_max_imm), 30),mode='lines', name='Time to 85% immunity'))
    fig.show()
update_graph('AL')