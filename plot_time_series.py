#!/usr/bin/python3
import pandas as pd
import numpy as np
import collections
from pathlib import Path
import plotly.express as px
from datetime import date, timedelta
from dateutil import parser
import plotly.graph_objs as go
from state_convert import us_state_abbrev, abbrev_us_state
from sklearn.linear_model import LinearRegression


path = Path('covid-vaccine-tracker-data/data/')
data = pd.read_csv(path/'historical-usa-doses-administered.csv')
data.fillna(0, inplace=True)
for k in range(len(data)):
	data['date'][k] = parser.parse(data['date'][k])
data.set_index('date')
data.pivot(columns='id', values='value')
index_to_num = {i: date(2020, 12, 21) + timedelta(k) for i, k in enumerate(range(len(data)))}
pops = pd.read_csv('data/state_pops.csv')
# fig = px.scatter(data, x=data.index, y='AK')
fig = go.Figure()
for col in data.columns:
	data[col+'ma'] = data[col].rolling(window=7).mean()
	m = np.polyfit(data.index, data[col+'ma'], 1)
	fig.add_scatter(x=data.index, y=data[col+'ma'], mode='lines', name=col)
fig.show()


# #print(data.head())
# for _, x in data.iterrows():
#     name = abbrev_us_state[x[0]]
#     if len(pops[pops['State']==name]):
#         if x[2] > counter[name]:
#             counter[name] = x[2]
#     #print(counter)
# for state in counter.keys():
#     if len(pops[pops['State']==state]):
#         pop_pcts = (pops[pops['State']==state]['Pop']).item()
#     else:
#         pop_pcts = 1
#     counter[state] /= pop_pcts
# fig = px.choropleth(locations=[us_state_abbrev[loc] for loc in np.unique(pops['State'])], locationmode="USA-states", color=counter, scope="usa")
# fig.show()
