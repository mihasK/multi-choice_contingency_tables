from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from icecream import ic
from dash import dash_table
from . import utils
from .utils import AnyOrNone, get_column_types
from . import cross_bd
import os
import json
import dash_daq as daq
import dash_bootstrap_components as dbc
from .tabs import vis

from .tabs.datasource import dataframes, data_tab_content


@callback([
    Output('feature-selection-1', 'options'), Output('feature-selection-1', 'value'),
    Output('feature-selection-2', 'options'), Output('feature-selection-2', 'value'),
    Output('column-selection', 'options'), Output('column-selection', 'value'),
], [
    Input('data_selector', 'value')
])
def update_feature_selectors(data_selector: str):
    
    df = dataframes[data_selector]
    column_types  =  get_column_types(df)
    
    vals = [
            {
                'label': f'{c} [{column_types[c]}]',
                'value': c,
            }
            for c in df.columns.to_list()
        ]
    return (
        vals, vals[0]['value']
    )*3


@callback([
    Output('data-tab', 'label')
], [
    Input('data_selector', 'value')
])
def update_data_title(data_selector: str):
    return (
        f'Data: {data_selector}',
    )


single_tab_content = dbc.Card(dbc.CardBody([
        html.H3(children='Single breakdowns', style={'textAlign':'center'}),
        dcc.Dropdown(id='column-selection'),
        dcc.Graph(id='single-breakdown-graph-content'),
    ]),     
    className="mt-3",
)

from try_dash.cross_bd import CBD_Types


cross_tab_content = dbc.Card(dbc.CardBody([
    html.H3(children='Cross breakdowns', style={'textAlign':'center'}),
    
    dbc.Row([
        dbc.Col(dcc.Dropdown(id='feature-selection-1')),
        'X',
        dbc.Col(dcc.Dropdown(id='feature-selection-2')),
    ]),    
    dcc.Loading(
        id="loading-1",
        type="default",
        children=html.Div(["Calculating..."], id="loading-output-1")
    ),
    dbc.Row([
        dbc.Col(
            dbc.RadioItems(
                options=list(CBD_Types),
                value='percents',
                inline=True,
                id='table-type-selector'            
            ),
        ),
        
        dbc.Col(
            daq.BooleanSwitch(
                on=False,
                label="Conf. intervals",
                labelPosition="top",
                id='conf-intervals-switch'
            ),
        ),
    ]),
    
    dcc.Graph(id='cross-breakdown-graph-content', responsive=False)
]),     className="mt-3",
)

app = Dash(__name__, title='Categical data analysis',
               external_stylesheets=[dbc.themes.BOOTSTRAP]

)
server = app.server



app.layout = dbc.Container([
    html.H1('Categorical data analysis'),
    
    # The memory store reverts to the default on every page refresh
    dcc.Store(id='CBD-store', ),
    
    dbc.Tabs([

        dbc.Tab(single_tab_content, label='Single breakdowns'),
        dbc.Tab(cross_tab_content, label='Cross breakdowns'),
        dbc.Tab(data_tab_content, label='Data', id='data-tab'),
        vis.vis_tab
    ]),
    # html.Div(id='tabs-content-graph', children=cross_tab_content)
])



def _display_ratio(value):
                    
    if value is None:
        return 'NAN'
    elif value > 1:
        return f'×{round(value, 2)}'
    elif 0<value <1:
        return f'÷{round(1/value, 2)}'
    elif value == 0:
        return '×0'
    elif value == 1:
        return '='
    else:
        return f'{AnyOrNone(value):.2f}'


from plotly.colors import n_colors

def _create_colors_for_percents() -> pd.Series:

    red_colors = n_colors('rgb(255, 255, 255)', 'rgb(255, 100, 100)', 101, colortype='rgb')


    blue_colors = n_colors('rgb(255, 255, 255)', 'rgb(100, 100, 255)', 101, colortype='rgb')
    blue_colors.reverse() 
    all_colors = blue_colors + red_colors

    s = pd.Series(all_colors)
    s.index = pd.Index(range(-101, 101))
    return s

percent_colors = _create_colors_for_percents()

def _scale_to_percents(num, lim_min=0, lim_max=1, ):
    assert lim_max > lim_min
    sign = 1 if num >=0 else -1
    
    anum = abs(num)
    if anum > lim_max:
        anum = lim_max
    if anum < lim_min:
        anum = lim_min
    
    return sign*round(100*(anum - lim_min)/(lim_max-lim_min))

        
        
def _logscale_percent(x, center=100):
    if x is None:
        return 0
    if x == 0:
        return -10000
    if math.isinf(x):
        return 10000
    return math.log(x) - math.log(center)


@app.callback(
    [
        Output('CBD-store', 'data'),
        Output("loading-1", "children"),
    ],
    [
        Input('feature-selection-1', 'value', ),
        Input('feature-selection-2', 'value', ),
        Input('data_selector', 'value', ),
    ]
)
def on_CBD_features_selected(f1, f2, data_name):
    if not (f1 and f2):
        print('not all features selected')
        return ''
    cbd = cross_bd.get_cross_breakdown(
        dataframes[data_name], f1, f2, 
    )
    # return cbd.to_json(date_format='iso', orient='split')
    return (
        json.dumps({
            'f1': f1,
            'f2': f2,
            'df': cbd.to_json(date_format='iso', orient='split')
        }),
        "Ready"
    )

from operator import attrgetter, itemgetter
from functools import partial
from collections import defaultdict
import math

@app.callback(
    Output('cross-breakdown-graph-content', 'figure'),
    [
        Input('CBD-store', 'data'),
        Input('table-type-selector', 'value', ),
        Input('conf-intervals-switch', 'on')
    ]
)
def update_cross_breakdown(
    table_df_json, 
    table_type: CBD_Types,
    conf_intervals_switch: bool
    ):
    if not table_df_json:
        print('No CBD data found')
        return 
    # ic(len(table_df_json))
    table_df_json = json.loads(table_df_json)
    f1 = table_df_json['f1']
    f2 = table_df_json['f2']
    # ic('update_cross_breakdown', f1,f2)
    
    table_df = pd.read_json(table_df_json['df'], orient='split')
    
    # ic(table_df.shape)
    
    def _format_cell(value):
        
        if table_type == CBD_Types.PERCENT_DEVIATIONS:
            if conf_intervals_switch:
                low,up = value['percent_diff_CI']
                return f'{low:+.1f}..{up:+.1f} %'        
            else:
                value = value['percent_diff']
                return f'{value:+.0f} %'      
          
        if table_type == CBD_Types.RISK_DIFFERENCE:
            if conf_intervals_switch:
                low, up = value['RD']['CI']
                return f'{low:+.1f}..{up:+.1f}%'
            else:
                value = value['RD']['exact']
                value = int(value)
                return f'{value:+} %'             
        
        if table_type == CBD_Types.ODDS_RATIO:
            if conf_intervals_switch:
                low, up = value['ODDS_RATIO']['CI']
                low, up = AnyOrNone(low), AnyOrNone(up)

                return f'{low:+.1f}..{up:+.1f}'
            else:
                value = value['ODDS_RATIO']['exact']
                value = AnyOrNone(value)
                return f'{value:.1f}'        
            
        if table_type == CBD_Types.RISK_RATIO:
            if conf_intervals_switch:
                low, up = value['RR']['CI']
                
                return f'{_display_ratio(low)}..{_display_ratio(up)}'
                # low, up = AnyOrNone(low), AnyOrNone(up)
                # return f'{low:.0f}..{up:.0f}%'
            else:
                value = value['RR']['exact']
                return _display_ratio(value)
        
        if table_type == CBD_Types.COUNTS:
            value = value['freq']
            return value        
        if table_type == CBD_Types.CHI2_TEST_INDEPENDENCE:
            value = value['CHI2']['pvalue']
            return f'{AnyOrNone(value):.2f}'       
        
        if table_type == CBD_Types.CRAMER_V_ASSOCIATION:
            value = value['CRAMER']
            return f'{AnyOrNone(value):.2f}'
        
        if table_type == CBD_Types.PERCENTS:
            if conf_intervals_switch:
                low,up = value['CI']
                return f'{low:.1f}..{up:.1f} %'
            else:
                value = value['percent']
                return f'{value:.0f} %'
        
        return value
    
    
    
    attr_for_color = itemgetter('percent')   # Default
    scale_lims = (0,100)

    
    if table_type in (CBD_Types.PERCENT_DEVIATIONS, ):
        scale_lims = (0,30)
        if conf_intervals_switch:
            attr_for_color = lambda x: utils.take_closest_to_zero(x['percent_diff_CI'])
        else:
            attr_for_color = itemgetter('percent_diff')
    elif table_type in (CBD_Types.RISK_DIFFERENCE,):
        scale_lims = (0,30)
        
        if conf_intervals_switch:
            attr_for_color = lambda x: utils.take_closest_to_zero(x['RD']['CI'])
        else:
            attr_for_color = lambda x: x['RD']['exact']      
    elif table_type in (CBD_Types.ODDS_RATIO,):
        scale_lims = (0,3)
        
        if conf_intervals_switch:
            attr_for_color = lambda x: utils.take_closest_to_zero(x['ODDS_RATIO']['CI'])
        else:
            attr_for_color = lambda x: _logscale_percent(x['ODDS_RATIO']['exact'], center=1)
    elif table_type in (CBD_Types.RISK_RATIO,):
        scale_lims = (0,2)

        
        if conf_intervals_switch:
            attr_for_color = lambda x: utils.take_closest_to_zero([
                _logscale_percent(x['RR']['CI'][0], center=1),
                _logscale_percent(x['RR']['CI'][1], center=1)
            ])
        else:
            attr_for_color = lambda x:  _logscale_percent(x['RR']['exact'], center=1)
    elif table_type == CBD_Types.PERCENTS:
        if conf_intervals_switch:
            attr_for_color = lambda x: utils.take_closest_to_zero(x['CI'])
    elif table_type == CBD_Types.CRAMER_V_ASSOCIATION:
        attr_for_color = itemgetter('CRAMER')   # Default
        scale_lims = (0,1)            

    
    if table_type == CBD_Types.CHI2_TEST_INDEPENDENCE:
        main_cells_colors = [
            table_df[col].apply(
                lambda x: 'orangered' if x['CHI2']['pvalue'] <= 0.05 else 'white'
            )
            for col in table_df.columns[:-1]
        ]
    else:
        scaler = partial(_scale_to_percents, lim_min=scale_lims[0], lim_max=scale_lims[1])
        
        main_cells_colors = [
                        percent_colors[
                            table_df[col].apply(attr_for_color).apply(scaler).to_numpy()
                        ]
                        for col in table_df.columns[:-1]
                    ]
        
    cells_values = [table_df.index] + \
        [table_df[col].apply(_format_cell) for col in table_df.columns[:-1]] + \
        [table_df[table_df.columns[-1]].str.get('percent').apply(lambda x: f'{x:.0f} %')]

    
    fig = go.Figure(data=[go.Table(
        header=dict(values=['_'] + list(table_df.columns),
                    fill_color='silver',
                    align='left'),
        cells=dict(values=cells_values,
                fill_color= [['silver']]  + main_cells_colors + [['silver']],
                # fill=dict(color=['paleturquoise', 'red']),
                align='left'))
    ], )
    fig.update_layout(
        margin=dict(l=2, r=2, t=2, b=2),
        # paper_bgcolor="LightSteelBlue",
        )
    fig.update_layout(height=100000)
    return fig

import statsmodels.stats.proportion
from . import utils

@callback(
    Output('single-breakdown-graph-content', 'figure'),
    [
        Input('column-selection', 'value', ),
        Input('data_selector', 'value')
    ]
)
def update_single_breakdown(column_name, data_name):
    
    df = dataframes[data_name]
    xdf = utils._explode(df, column_name)
    
    
    values = list(xdf[column_name].unique())
    
    bar_data = pd.DataFrame(index=values)
    
    bar_data['freqs'] = bar_data.index.map(
        lambda x: (xdf[column_name] == x).sum()
    )
    total = xdf[column_name].notna().sum()
    
    
    # Multinomial CI
    # ci = statsmodels.stats.proportion.multinomial_proportions_confint(bar_data['freqs'],)
    
    # ic(ci)
    # ic(ci[:,1] - ci[:,0])
    
    ci = statsmodels.stats.proportion.proportion_confint(bar_data['freqs'].to_numpy(), total, method='wilson')
    ci = np.array(ci).T
    # ic(ci)
    # ic(ci[:,1] - ci[:,0])
    
    
    bar_data['CI_min'] = 100*ci[:,0]
    bar_data['CI_max'] = 100*ci[:,1]
    
    bar_data['PERC'] = 100 * bar_data.freqs / total
        
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=values,
            y=bar_data['PERC'].to_numpy().round(),
            error_y=dict(
                type='data',
                symmetric=False,
                array=(bar_data.CI_max - bar_data.PERC).round(1),
                arrayminus=(bar_data.PERC - bar_data.CI_min).round(1)
            )
        )
    )
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
