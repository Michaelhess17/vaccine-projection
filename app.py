#!/usr/bin/python
       
import os
import pathlib
import re
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
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
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
        {'name': 'og:description', 'content': 'Curious when your state will administer enough Covid-19 vaccinations to reach herd-immunity?'},
        {'name': 'og:title', 'content': 'Covid-19 Vaccine Projections'},
        {'name': 'og:image', 'content': 'http://www.nadadventist.org/sites/default/files/inline-images/iStock-1219398943.jpg'},
        {'name': 'og:url', 'content': 'covid-vaccines.app'},
        {'name': 'author', 'content': 'Michael Hess'},
        {"name":"twitter:card", "content":"summary_large_image"},
        {"name": "twitter:description", "content": "Curious when your state will administer enough Covid-19 vaccinations to reach herd-immunity?"},
        {'name': 'twitter:image', 'content': 'http://www.nadadventist.org/sites/default/files/inline-images/iStock-1219398943.jpg'},
        {'name': 'twitter:url', 'content': 'covid-vaccines.app'},
    ],
    external_stylesheets=[dbc.themes.CYBORG],
)

server = app.server
# Load data
pd.options.mode.chained_assignment = None
path = Path('covid-vaccine-tracker-data/data/')
df = pd.read_csv('data/us_state_vaccinations.csv')

diff = np.max(df['date'].apply(lambda x: parse(x)))- datetime(2020, 12, 21)
days = diff.days


data = pd.read_csv(path / 'historical-usa-doses-administered.csv')
data.fillna(0, inplace=True)
data = data.set_index('date')
data = data.pivot(columns='id', values='value')
data.index = pd.Series([datetime(2020, 12, 21) + timedelta(days=k) for k in range(len(data))])
pops = pd.read_csv('data/state_pops.csv')
num_to_index = {i: datetime(2020, 12, 21) + timedelta(days=k) for i, k in enumerate(range(days + 1300))}
index_to_num = dict(map(reversed, num_to_index.items()))
cols = ["total_vaccinations","total_vaccinations_per_hundred", "people_fully_vaccinated","people_fully_vaccinated_per_hundred", "total_distributed","distributed_per_hundred"]


def get_data(col, state):
    if state == None:
        data = df[['date','location',col]].set_index('date').pivot(columns='location', values=col).fillna(method='ffill').fillna(0)
    else:
        data = df[['date','location',col]].set_index('date').pivot(columns='location', values=col).fillna(method='ffill').fillna(0)[state].fillna(0)

    return data


data.fillna(0, inplace=True)

xs, ys, zs = np.load('data/xs.npy'), np.load('data/ys.npy'), np.load('data/zs.npy')

most_current = index_to_num[np.max([parse(day) for day in df.date])]
ds = list(range(most_current, most_current + 50))
ds_days = [num_to_index[x] for x in ds]
all_days = np.arange(num_to_index[0], num_to_index[140], timedelta(1))


fig = go.Figure()
fig_layout = fig["layout"]
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
for state in counter.keys():
    if len(pops[pops['State']==state]):
        pop_pcts = (pops[pops['State']==state]['Pop']).item()
    else:
        pop_pcts = 100
    counter[state] /= pop_pcts
fig_map = go.Figure(go.Choropleth(locationmode="USA-states", colorscale = 'tealgrn', colorbar_title = "% Covered", locations=[us_state_abbrev[loc] for loc in np.unique(pops['State'])], z=[0.5]*len(np.unique(pops['State']))), 
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


def suffix(d):
    return 'th' if 11<=d<=13 else {1:'st',2:'nd',3:'rd'}.get(d%10, 'th')
    
def custom_strftime(format, t):
    return t.strftime(format).replace('{S}', str(t.day) + suffix(t.day))
 


DEFAULT_OPACITY = 0.8

mapbox_access_token = "pk.eyJ1IjoicGxvdGx5bWFwYm94IiwiYSI6ImNrOWJqb2F4djBnMjEzbG50amg0dnJieG4ifQ.Zme1-Uzoi75IaFbieBDl3A"
mapbox_style = "mapbox://styles/plotlymapbox/cjvprkf3t1kns1cqjxuxmwixz"

# App layout

app.layout = dbc.Container(
    id="root",
    children=[
        dbc.Row(
            dbc.Col([
                html.H4(children="US Population Covered by Coronavirus Vaccinations",
                    style={'text-align': 'center'}),
                html.H6(
                children="† Joe Biden has promised that all American adults will have access to the Coronavirus "
                             "vaccine by May 1st. In this project, we track the changes in the vaccination rates to "
                             "determine if this deadline will be met at the state level for all US states.",
                             style={'border-left': '#2cfec1 solid 0.5rem', 'padding-left': '0.5rem'}),
            ],  xs=12, sm=10, md=8, lg=8, xl=6), justify='center'),
        dbc.Row([
            dbc.Col([
                html.H6(
                    id="slider-text",
                    children="Choose which metric you are interested in:",),
                dcc.Dropdown(
                    id="years-slider",
                    options = [
                    {"label": "Total Vaccinations", "value":"total_vaccinations"},
                    {"label": "Total Vaccinations per Person", "value":"total_vaccinations_per_hundred"},
                    {"label": "People Fully Vaccinated", "value":"people_fully_vaccinated"},
                    {"label": "People Fully Vaccinated per Person", "value":"people_fully_vaccinated_per_hundred"},
                    {"label": "Vaccine Distributed", "value":"total_distributed"},
                    {"label": "Vaccine Distributed per Person", "value":"distributed_per_hundred"}                              ,],
                    value="total_vaccinations_per_hundred",
                style={'bottom-padding': '5rem'},
                    searchable=False), 
                dcc.Graph(
                    id="county-choropleth",
                    figure=fig_map),
                            ],
                            xs=12, sm=8, md=5, lg=6, xl=5), 
                dbc.Col([
                        html.H6(id="chart-selector", children="Select which state you are interested in:"),
                        dcc.Dropdown(
                            options=[{'label': 'United States', 'value': 'US'},
                                     {'label': 'AL', 'value': 'AL'},
                                     {'label': 'AK', 'value': 'AK'},
                                     {'label': 'AS', 'value': 'AS'},
                                     {'label': 'AZ', 'value': 'AZ'},
                                     {'label': 'AR', 'value': 'AR'},
                                     {'label': 'CA', 'value': 'CA'},
                                     {'label': 'CO', 'value': 'CO'},
                                     {'label': 'CT', 'value': 'CT'},
                                     {'label': 'DE', 'value': 'DE'},
                                     {'label': 'DC', 'value': 'DC'},
                                     {'label': 'FL', 'value': 'FL'},
                                     {'label': 'FM', 'value': 'FM'},
                                     {'label': 'GA', 'value': 'GA'},
                                     {'label': 'GU', 'value': 'GU'},
                                     {'label': 'HI', 'value': 'HI'},
                                     {'label': 'ID', 'value': 'ID'},
                                     {'label': 'IL', 'value': 'IL'},
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
                                     {'label': 'NC', 'value': 'NC'},
                                     {'label': 'ND', 'value': 'ND'},
                                     {'label': 'MP', 'value': 'MP'},
                                     {'label': 'OH', 'value': 'OH'},
                                     {'label': 'OK', 'value': 'OK'},
                                     {'label': 'OR', 'value': 'OR'},
                                     {'label': 'PW', 'value': 'PW'},
                                     {'label': 'PA', 'value': 'PA'},
                                     {'label': 'PR', 'value': 'PR'},
                                     {'label': 'RI', 'value': 'RI'},
                                     {'label': 'SC', 'value': 'SC'},
                                     {'label': 'SD', 'value': 'SD'},
                                     {'label': 'TN', 'value': 'TN'},
                                     {'label': 'TX', 'value': 'TX'},
                                     {'label': 'UT', 'value': 'UT'},
                                     {'label': 'VT', 'value': 'VT'},
                                     {'label': 'VI', 'value': 'VI'},
                                     {'label': 'VA', 'value': 'VA'},
                                     {'label': 'WA', 'value': 'WA'},
                                     {'label': 'WV', 'value': 'WV'},
                                     {'label': 'WI', 'value': 'WI'},
                                     {'label': 'WY', 'value': 'WY'}],
                            value="US",
                            id="chart-dropdown",
                            searchable=False
                        ),
                        dcc.Graph(
                            id="selected-data",
                            figure=fig,
                            style={'margin-bottom': 0})
                ],
                xs=12, sm=8, md=5, lg=6, xl=5),
                ],justify='center'), # no_gutters=True),
            dbc.Row(dbc.Col(html.Div(id="text-output",
                    children = [
                    html.H4("",
                    id="timeline-text",
                    style={'text-align': 'center', 'padding-top': '0.5rem'},
                    )]), xs=12, sm=8, md=5, lg=6, xl=5), justify='center'),
        ], fluid=True,
        )

app.title = 'Covid-19 Vaccine Projections'
@app.callback(
    Output("county-choropleth", "figure"),
    Input("years-slider", "value")
)
def display_map(year):
    state = None
    if 'hundred' in year:
        char = '%'
    else:
        char = '#'
    counter = collections.defaultdict(int)
    current_data = get_data(year, state)
    for name, x in current_data.iteritems():
        #print("name1  = ", name)
        # name = us_state_abbrev[name]
        if len(pops[pops['State'] == name]):
                counter[name] = np.max(x)
    #print(counter)
    fig_map = go.Figure(
        go.Choropleth(locations=[us_state_abbrev[loc] for loc in current_data.columns], locationmode="USA-states",
                      z=[counter[loc] for loc in current_data.columns], colorscale='tealgrn', colorbar_title=f"{char} Covered"),
        layout=go.Layout(geo=dict(bgcolor="#1f2630", lakecolor="#1f2630"),
                         font={"size": 9, "color": "White"},
                         titlefont={"size": 15, "color": "White"},
                         geo_scope='usa',
                         margin={"r": 0, "t": 40, "l": 0, "b": 0},
                         paper_bgcolor='#1f2630',
                         plot_bgcolor='#1f2630',
                         legend=dict(orientation='h'),
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


#@app.callback(Output("heatmap-title", "children"), [Input("years-slider", "value")])
#def update_map_title(year):
#    foo = year.split('_')
#    c = ''
#    for word in foo:
#        c += word[0].upper() + word[1:] + ' '
#    return f"Heatmap of {c}"


@app.callback(
    [Output("selected-data", "figure"),
        Output("timeline-text", "children")],
    [
        Input("county-choropleth", "selectedData"),
        Input("chart-dropdown", "value"),
        Input("years-slider", "value"),
    ],
)
def display_selected_data(selectedData, chart_dropdown, year):
    idx = np.argmax(np.array(['US'] + data.columns.to_list()) == chart_dropdown)
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
                         legend=dict(orientation='h', xanchor='center', x=0.5),
                         ))
    fig.add_trace(go.Scatter(x=current_data.index, y=current_data, mode='lines', name='Data to Date'))
    
    if 'hundred' in year:
        pop = 100
    else:
        try:
            pop = pops[pops['State'] == state]['Pop'].to_numpy()[0]
        except:
            pop = get_data('total_vaccinations', state).to_list()
            pop100 = (get_data('total_vaccinations_per_hundred', state)/100).to_list()
            try:
                pop = pop[-1]/pop100[-1]
            except:
                pop = 100
                year += 'per_hundred'
    time_to_min_imm = np.max((extrap - (pop * 0.75)).roots)
    time_to_max_imm = np.max((extrap - (pop * 0.85)).roots)
    #print("time_to_max_imm, state, pop  = ", time_to_max_imm, state, pop )
    if time_to_max_imm < 1000:
        t_min = datetime(2020, 12, 21) + timedelta(days=round(time_to_min_imm))
        t_max = datetime(2020, 12, 21) + timedelta(days=round(time_to_max_imm))
        cds = list(range(most_current, round(time_to_max_imm)))
        cds_days = [num_to_index[x] for x in cds]
        call_days = np.arange(num_to_index[0], num_to_index[most_current+len(cds)], timedelta(1))
    else:
        cds = list(range(most_current, most_current+100))
        cds_days = [num_to_index[x] for x in cds]
        call_days = np.arange(num_to_index[0], num_to_index[most_current+len(cds)], timedelta(1))
    fig.add_trace(go.Scatter(x=cds_days, y=extrap(cds), mode='lines', name='Projected Data'))
    if (countdown) & (time_to_max_imm < 1000): 
        fig.add_trace(go.Scatter(x=call_days, y=[pop * 0.75] * len(call_days), mode='lines', name='75% of Pop.'))
        fig.add_trace(go.Scatter(x=call_days, y=[pop * 0.85] * len(call_days), mode='lines', name='85% of Pop.'))
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
    
    today = datetime.now()
    if time_to_max_imm < 1000:
        t_str = custom_strftime('%B {S}, %Y', t_max)
        diff = round((t_max-today).days)
        text = f"Based on current trends, 85% coverage will be reached in {state} by approximately {t_str}, which is in {diff} days."
    else: text = f""
    return fig, text

    
if __name__ == "__main__":
    app.run_server(debug=True)
