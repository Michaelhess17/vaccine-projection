#!/usr/bin/python3
import pandas as pd
import numpy as np
import collections
from pathlib import Path
import plotly.express as px
import plotly
from state_convert import us_state_abbrev, abbrev_us_state

counter = collections.defaultdict(int)
path = Path('covid-vaccine-tracker-data/data/')
data = pd.read_csv(path/'historical-usa-doses-administered.csv')
pops = pd.read_csv('data/state_pops.csv')
#print(data.head())
for _, x in data.iterrows():
    name = abbrev_us_state[x[0]]
    if len(pops[pops['State']==name]):
        if x[2] > counter[name]:
            counter[name] = x[2]
    #print(counter)
for state in counter.keys():
    if len(pops[pops['State']==state]):
        pop_pcts = (pops[pops['State']==state]['Pop']).item()
    else:
        pop_pcts = 1
    counter[state] /= pop_pcts
fig = px.choropleth(locations=[us_state_abbrev[loc] for loc in np.unique(pops['State'])], locationmode="USA-states", color=counter, scope="usa")
fig.show()
