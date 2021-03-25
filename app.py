import os
import pathlib
import re

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output, State
import cufflinks as cf
from scipy.optimize import curve_fit
import collections
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objs as go
from state_convert import abbrev_us_state, us_state_abbrev
from dateutil.parser import parse
from datetime import timedelta, datetime
from state_convert import abbrev_us_state, us_state_abbrev

# Initialize app

app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
server = app.server

# Load data
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
num_to_index = {i: datetime(2020, 12, 21) + timedelta(days=k) for i, k in enumerate(range(days + 100))}
index_to_num = dict(map(reversed, num_to_index.items()))
cols = ["total_vaccinations","total_vaccinations_per_hundred", "people_fully_vaccinated","people_fully_vaccinated_per_hundred", "total_distributed","distributed_per_hundred"]


def get_data(col, state):
    if state == None:
        data = df[['date','location',col]].set_index('date').pivot(columns='location', values=col).fillna(method='ffill').fillna(0)
    else:
        data = df[['date','location',col]].set_index('date').pivot(columns='location', values=col).fillna(method='ffill').fillna(0)[state].fillna(0)

    return data


xs, ys, zs = np.zeros((data.shape[1], len(cols))), np.zeros((data.shape[1], len(cols))), np.zeros((data.shape[1], len(cols)))
data.fillna(0, inplace=True)

df = pd.read_csv('data/us_state_vaccinations.csv')

def func(x, a, b, c):
    return a + b * x + c * x ** 2

for idy, data_col in enumerate(cols):
    for idx, col in enumerate(data.columns):
        if abbrev_us_state[col] not in df.location.to_list():
            continue
        ma = get_data(data_col, abbrev_us_state[col]).rolling(window=10).mean().fillna(method='ffill').fillna(0)
        try:
            coefs = curve_fit(func, list(range(len(ma))), ma, bounds=((-np.inf,-np.inf,0), (np.inf,np.inf,np.inf)))[0]
            xs[idx, idy], ys[idx, idy], zs[idx, idy] = coefs
        except ValueError:
            print(ma)

most_current = index_to_num[data.index[-1]]
ds = list(range(most_current, most_current + 50))
ds_days = [num_to_index[x] for x in ds]
all_days = np.arange(num_to_index[0], num_to_index[140], timedelta(1))


state = 'CA'
year = 'total_vaccinations'
idx = np.argmax(data.columns == state)
idy = cols.index(year)
extrap = np.poly1d((zs[idx, idy], ys[idx, idy], xs[idx, idy]))
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index[:days], y=data[state][:days], mode='lines', name='Data to Date'))
fig.add_trace(go.Scatter(x=ds_days, y=extrap(ds), mode='lines', name='Projected Data'))
pop = pops[pops['State'] == abbrev_us_state[state]]['Pop'].to_numpy()[0]

time_to_min_imm = np.max((extrap - (pop * 0.75)).roots)
time_to_max_imm = np.max((extrap - (pop * 0.85)).roots)
t_min = datetime(2020, 12, 21) + timedelta(days=round(time_to_min_imm))
t_max = datetime(2020, 12, 21) + timedelta(days=round(time_to_max_imm))
fig.add_trace(go.Scatter(x=all_days, y=[pop * 0.75] * len(all_days), mode='lines', name='75% of Pop.'))
fig.add_trace(go.Scatter(x=all_days, y=[pop * 0.85] * len(all_days), mode='lines', name='85% of Pop.'))
fig.add_trace(go.Scatter(x=[t_min] * 30, y=np.linspace(0, extrap(time_to_min_imm), 30), mode='lines',
                         name='Time to 75% immunity'))
fig.add_trace(go.Scatter(x=[t_max] * 30, y=np.linspace(0, extrap(time_to_max_imm), 30), mode='lines',
                         name='Time to 85% immunity'))

fig_layout = fig["layout"]
fig_data = fig["data"]

# fig_data[0]["text"] = deaths_or_rate_by_fips.values.tolist()
# fig_data[0]["marker"]["color"] = "#2cfec1"
fig_data[0]["marker"]["opacity"] = 1
fig_data[0]["marker"]["line"]["width"] = 1.5
# fig_data[0]["textposition"] = "top center"
fig_layout["paper_bgcolor"] = "#1f2630"
fig_layout["plot_bgcolor"] = "#1f2630"
fig_layout["font"]["color"] = "#2cfec1"
fig_layout["title"]["font"]["color"] = "#2cfec1"
fig_layout["xaxis"]["tickfont"]["color"] = "#2cfec1"
fig_layout["yaxis"]["tickfont"]["color"] = "#2cfec1"
fig_layout["xaxis"]["gridcolor"] = "#5b5b5b"
fig_layout["yaxis"]["gridcolor"] = "#5b5b5b"
fig_layout["margin"]["t"] = 75
fig_layout["margin"]["r"] = 50
fig_layout["margin"]["b"] = 100
fig_layout["margin"]["l"] = 50

APP_PATH = str(pathlib.Path(__file__).parent.resolve())


counter = collections.defaultdict(int)
for name, x in data.iteritems():
    try:
        name = abbrev_us_state[name]
    except KeyError:
        pass
    if len(pops[pops['State']==name]):
        counter[name] = np.max(x)
    #print(counter)
for state in counter.keys():
    if len(pops[pops['State']==state]):
        pop_pcts = (pops[pops['State']==state]['Pop']).item()
    else:
        pop_pcts = 1
    counter[state] /= pop_pcts
fig_map = go.Figure(go.Choropleth(locations=[us_state_abbrev[loc] for loc in np.unique(pops['State'])], locationmode="USA-states",
                        z=list(counter.values()), colorscale = 'tealgrn', colorbar_title = "% Covered"),
            layout = go.Layout(geo=dict(bgcolor="#1f2630", lakecolor="#1f2630"),
                                  font = {"size": 9, "color":"White"},
                                  titlefont = {"size": 15, "color":"White"},
                                  geo_scope='usa',
                                  margin={"r":0,"t":40,"l":0,"b":0},
                                  paper_bgcolor='#4E5D6C',
                                  plot_bgcolor='#4E5D6C',
                                  )
            )
fig_layout = fig_map["layout"]
fig_data = fig_map["data"]

fig_data[0]["marker"]["opacity"] = 1
fig_data[0]["marker"]["line"]["width"] = 1.5
fig_layout["paper_bgcolor"] = "#1f2630"
fig_layout["plot_bgcolor"] = "#1f2630"
fig_layout["font"]["color"] = "#2cfec1"
fig_layout["title"]["font"]["color"] = "#2cfec1"
fig_layout["xaxis"]["tickfont"]["color"] = "#2cfec1"
fig_layout["yaxis"]["tickfont"]["color"] = "#2cfec1"
fig_layout["xaxis"]["gridcolor"] = "#5b5b5b"
fig_layout["yaxis"]["gridcolor"] = "#5b5b5b"
fig_layout["margin"]["t"] = 75
fig_layout["margin"]["r"] = 50
fig_layout["margin"]["b"] = 100
fig_layout["margin"]["l"] = 50


DEFAULT_COLORSCALE = [
    "#f2fffb",
    "#bbffeb",
    "#98ffe0",
    "#79ffd6",
    "#6df0c8",
    "#69e7c0",
    "#59dab2",
    "#45d0a5",
    "#31c194",
    "#2bb489",
    "#25a27b",
    "#1e906d",
    "#188463",
    "#157658",
    "#11684d",
    "#10523e",
]

DEFAULT_OPACITY = 0.8

mapbox_access_token = "pk.eyJ1IjoicGxvdGx5bWFwYm94IiwiYSI6ImNrOWJqb2F4djBnMjEzbG50amg0dnJieG4ifQ.Zme1-Uzoi75IaFbieBDl3A"
mapbox_style = "mapbox://styles/plotlymapbox/cjvprkf3t1kns1cqjxuxmwixz"

# App layout

app.layout = html.Div(
    id="root",
    children=[
        html.Div(
            id="header",
            children=[
                html.H4(children="US Population Covered by Coronavirus Vaccinations"),
                html.P(
                    id="description",
                    children="† Joe Biden has promised that all American adults will have access to the Coronavirus "
                             "vaccine by May 1st. In this project, we track the changes in the vaccination rates to "
                             "determine if this deadline will be met at the state level for all US states.",
                ),
            ],
        ),
        html.Div(
            id="app-container",
            children=[
                html.Div(
                    id="left-column",
                    children=[
                        html.Div(
                            id="slider-container",
                            children=[
                                html.P(
                                    id="slider-text",
                                    children="Choose which metric you are interested in:",
                                ),
                                dcc.Dropdown(
                                    id="years-slider",
                                    options = [
                                        {"label": "Total Vaccinations", "value":"total_vaccinations"},
                                        {"label": "Total Vaccinations per Person", "value":"total_vaccinations_per_hundred"},
                                        {"label": "People Fully Vaccinated", "value":"people_fully_vaccinated"},
                                        {"label": "People Fully Vaccinated per Person", "value":"people_fully_vaccinated_per_hundred"},
                                        {"label": "Vaccine Distributed", "value":"total_distributed"},
                                        {"label": "Vaccine Distributed per Person", "value":"distributed_per_hundred"},

                                        ],
                                    value="total_vaccinations_per_hundred",
                                ),
                            ],
                        ),
                        html.Div(
                            id="heatmap-container",
                            children=[
                                html.P(
                                    "Heatmap of age adjusted mortality rates min(YEARS)",
                                    id="heatmap-title",
                                ),
                                dcc.Graph(
                                    id="county-choropleth",
                                    figure=fig_map
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    id="graph-container",
                    children=[
                        html.P(id="chart-selector", children="Select chart:"),
                        dcc.Dropdown(
                            options=[{'label': 'AL', 'value': 'AL'},
                                     {'label': 'AK', 'value': 'AK'},
                                     {'label': 'AS', 'value': 'AS'},
                                     {'label': 'AZ', 'value': 'AZ'},
                                     {'label': 'AR', 'value': 'AR'},
                                     {'label': 'Bureau of Prisons', 'value': 'bureau-of-prisons'},
                                     {'label': 'CA', 'value': 'CA'},
                                     {'label': 'CO', 'value': 'CO'},
                                     {'label': 'CT', 'value': 'CT'},
                                     {'label': 'DE', 'value': 'DE'},
                                     {'label': 'Dept. of Defense', 'value': 'dept-of-defense'},
                                     {'label': 'DC', 'value': 'DC'},
                                     {'label': 'Federal Entities', 'value': 'federal-entities'},
                                     {'label': 'FL', 'value': 'FL'},
                                     {'label': 'FM', 'value': 'FM'},
                                     {'label': 'GA', 'value': 'GA'},
                                     {'label': 'GU', 'value': 'GU'},
                                     {'label': 'HI', 'value': 'HI'},
                                     {'label': 'ID', 'value': 'ID'},
                                     {'label': 'IL', 'value': 'IL'},
                                     {'label': 'Chicago', 'value': 'chicago'},
                                     {'label': 'Indian Health Service', 'value': 'indian-health-service'},
                                     {'label': 'IN', 'value': 'IN'},
                                     {'label': 'IA', 'value': 'IA'},
                                     {'label': 'KS', 'value': 'KS'},
                                     {'label': 'KY', 'value': 'KY'},
                                     {'label': 'LA', 'value': 'LA'},
                                     {'label': 'ME', 'value': 'ME'},
                                     {'label': 'MH', 'value': 'MH'},
                                     {'label': 'MD', 'value': 'MD'},
                                     {'label': 'MA', 'value': 'MA'},
                                     {'label': 'MI', 'value': 'MI'},
                                     {'label': 'MN', 'value': 'MN'},
                                     {'label': 'MS', 'value': 'MS'},
                                     {'label': 'MO', 'value': 'MO'},
                                     {'label': 'MT', 'value': 'MT'},
                                     {'label': 'NE', 'value': 'NE'},
                                     {'label': 'NV', 'value': 'NV'},
                                     {'label': 'NH', 'value': 'NH'},
                                     {'label': 'NJ', 'value': 'NJ'},
                                     {'label': 'NM', 'value': 'NM'},
                                     {'label': 'NY', 'value': 'NY'},
                                     {'label': 'NYC', 'value': 'new-york-city'},
                                     {'label': 'NC', 'value': 'NC'},
                                     {'label': 'ND', 'value': 'ND'},
                                     {'label': 'MP', 'value': 'MP'},
                                     {'label': 'OH', 'value': 'OH'},
                                     {'label': 'OK', 'value': 'OK'},
                                     {'label': 'OR', 'value': 'OR'},
                                     {'label': 'Unassigned', 'value': 'unassigned'},
                                     {'label': 'PW', 'value': 'PW'},
                                     {'label': 'PA', 'value': 'PA'},
                                     {'label': 'PR', 'value': 'PR'},
                                     {'label': 'RI', 'value': 'RI'},
                                     {'label': 'SC', 'value': 'SC'},
                                     {'label': 'SD', 'value': 'SD'},
                                     {'label': 'TN', 'value': 'TN'},
                                     {'label': 'TX', 'value': 'TX'},
                                     {'label': 'UT', 'value': 'UT'},
                                     {'label': 'Veterans\' Health', 'value': 'veterans-health'},
                                     {'label': 'VT', 'value': 'VT'},
                                     {'label': 'VI', 'value': 'VI'},
                                     {'label': 'VA', 'value': 'VA'},
                                     {'label': 'WA', 'value': 'WA'},
                                     {'label': 'WV', 'value': 'WV'},
                                     {'label': 'WI', 'value': 'WI'},
                                     {'label': 'WY', 'value': 'WY'}],
                            value="CA",
                            id="chart-dropdown",
                        ),
                        dcc.Graph(
                            id="selected-data",
                            figure=fig,
                        ),
                    ],
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("county-choropleth", "figure"),
    Input("years-slider", "value")
)
def display_map(year):
    state = None
    counter = collections.defaultdict(int)
    current_data = get_data(year, state)
    for name, x in current_data.iteritems():
        print("name1  = ", name)
        # name = us_state_abbrev[name]
        if len(pops[pops['State'] == name]):
                counter[name] = np.max(x)
    print(counter)
    fig_map = go.Figure(
        go.Choropleth(locations=[us_state_abbrev[loc] for loc in current_data.columns], locationmode="USA-states",
                      z=[counter[loc] for loc in current_data.columns], colorscale='tealgrn', colorbar_title="% Covered"),
        layout=go.Layout(geo=dict(bgcolor="#1f2630", lakecolor="#1f2630"),
                         font={"size": 9, "color": "White"},
                         titlefont={"size": 15, "color": "White"},
                         geo_scope='usa',
                         margin={"r": 0, "t": 40, "l": 0, "b": 0},
                         paper_bgcolor='#1f2630',
                         plot_bgcolor='#1f2630',
                         )
        )
    fig_layout = fig_map["layout"]
    fig_data = fig_map["data"]

    fig_data[0]["marker"]["opacity"] = 1
    fig_data[0]["marker"]["line"]["width"] = 1.5
    fig_layout["paper_bgcolor"] = "#1f2630"
    fig_layout["plot_bgcolor"] = "#1f2630"
    fig_layout["font"]["color"] = "#2cfec1"
    fig_layout["title"]["font"]["color"] = "#2cfec1"
    fig_layout["xaxis"]["tickfont"]["color"] = "#2cfec1"
    fig_layout["yaxis"]["tickfont"]["color"] = "#2cfec1"
    fig_layout["xaxis"]["gridcolor"] = "#5b5b5b"
    fig_layout["yaxis"]["gridcolor"] = "#5b5b5b"
    fig_layout["margin"]["t"] = 75
    fig_layout["margin"]["r"] = 50
    fig_layout["margin"]["b"] = 100
    fig_layout["margin"]["l"] = 50
    fig_layout = fig["layout"]
    fig_data = fig["data"]

    return fig_map


@app.callback(Output("heatmap-title", "children"), [Input("years-slider", "value")])
def update_map_title(year):
    foo = year.split('_')
    c = ''
    for word in foo:
        c += word[0].upper() + word[1:] + ' '
    return f"Heatmap of {c}"


@app.callback(
    Output("selected-data", "figure"),
    [
        Input("county-choropleth", "selectedData"),
        Input("chart-dropdown", "value"),
        Input("years-slider", "value"),
    ],
)
def display_selected_data(selectedData, chart_dropdown, year):
    idx = np.argmax(data.columns == chart_dropdown)
    state = abbrev_us_state[chart_dropdown]
    countdown = False
    if year in ['total_vaccinations_per_hundred','total_vaccinations','people_fully_vaccinated','people_fully_vaccinated_per_hundred']:
        countdown = True
    current_data = get_data(year, state)
    idy = cols.index(year)
    extrap = np.poly1d((zs[idx, idy], ys[idx, idy], xs[idx, idy]))
    fig = go.Figure(layout=go.Layout(font={"size": 9, "color": "White"},
                         titlefont={"size": 15, "color": "White"},
                         geo_scope='usa',
                         margin={"r": 0, "t": 40, "l": 0, "b": 0},
                         paper_bgcolor='#1f2630',
                         plot_bgcolor='#1f2630',
                         ))
    fig.add_trace(go.Scatter(x=current_data.index[:days], y=current_data[:days], mode='lines', name='Data to Date'))
    fig.add_trace(go.Scatter(x=ds_days, y=extrap(ds), mode='lines', name='Projected Data'))

    if countdown:
        if 'hundred' in year:
            pop = 1
        else:
            pop = pops[pops['State'] == state]['Pop'].to_numpy()[0]
        time_to_min_imm = np.max((extrap - (pop * 0.75)).roots)
        time_to_max_imm = np.max((extrap - (pop * 0.85)).roots)
        t_min = datetime(2020, 12, 21) + timedelta(days=round(time_to_min_imm))
        t_max = datetime(2020, 12, 21) + timedelta(days=round(time_to_max_imm))
        fig.add_trace(go.Scatter(x=all_days, y=[pop * 0.75] * len(all_days), mode='lines', name='75% of Pop.'))
        fig.add_trace(go.Scatter(x=all_days, y=[pop * 0.85] * len(all_days), mode='lines', name='85% of Pop.'))
        fig.add_trace(go.Scatter(x=[t_min] * 30, y=np.linspace(0, extrap(time_to_min_imm), 30), mode='lines',
                                 name='Time to 75% immunity'))
        fig.add_trace(go.Scatter(x=[t_max] * 30, y=np.linspace(0, extrap(time_to_max_imm), 30), mode='lines',
                                 name='Time to 85% immunity'))
    fig_layout = fig_map["layout"]
    fig_data = fig_map["data"]

    fig_data[0]["marker"]["opacity"] = 1
    fig_data[0]["marker"]["line"]["width"] = 1.5
    fig_layout["paper_bgcolor"] = "#1f2630"
    fig_layout["plot_bgcolor"] = "#1f2630"
    fig_layout["font"]["color"] = "#2cfec1"
    fig_layout["title"]["font"]["color"] = "#2cfec1"
    fig_layout["xaxis"]["tickfont"]["color"] = "#2cfec1"
    fig_layout["yaxis"]["tickfont"]["color"] = "#2cfec1"
    fig_layout["xaxis"]["gridcolor"] = "#5b5b5b"
    fig_layout["yaxis"]["gridcolor"] = "#5b5b5b"
    fig_layout["margin"]["t"] = 75
    fig_layout["margin"]["r"] = 50
    fig_layout["margin"]["b"] = 100
    fig_layout["margin"]["l"] = 50
    return fig

    
if __name__ == "__main__":
    app.run_server(debug=True)
