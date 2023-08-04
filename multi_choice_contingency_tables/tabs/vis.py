from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from icecream import ic
from dash import dash_table
import os
import json
import dash_bootstrap_components as dbc
from statsmodels.stats.proportion import proportion_confint
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots


vis_tab = dbc.Tab(
    
    dbc.Card(className="mt-3",
             children=dbc.CardBody([
                 
                 
            html.H3(children='CI width', style={'textAlign':'center'}),
            
            dbc.Row([
                        dbc.Col([
                                dbc.Label("Sample size"),
                                dbc.Input(placeholder="Input goes here...", type="number", value=1000, id='total_N'),
                                dbc.FormText("Total number of items in sample affects widths of confidence interval"),
                        ]),
                        dbc.Col([
                            dbc.Label("Confidence level"),
                            dcc.Slider(80, 100, 1,
                                value=95,
                                id='confidence_level'
                            ),
                            dbc.FormText("Higher confidence produces wider interval."),

                        ]),
                        # dcc.Dropdown(id='column-selection'),    
            ]),

        
        #  dbc.Row([  
        #     dbc.Col([
                dcc.Graph(id='ci-graph-content'),
            # ]),
            # dbc.Col([
            #     dcc.Graph(id='ci-over-N-graph-content'),
            # ])
            # ]),     
            
            
            
        html.H3(children='2 proportions difference CI', style={'textAlign':'center'}),
            
        dbc.Row([
                    dbc.Label("Proportion value 1"),
                    dcc.Slider(0, 100,
                        value=30,
                        id='prop-1',
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    # dbc.FormText("Higher confidence produces wider interval."),
                    dbc.Label("Proportion value 2"),
                    dcc.Slider(0, 100,
                        value=60,
                        id='prop-2',
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    # dbc.FormText("Higher confidence produces wider interval."),
        ]),
        
        dcc.Graph(id='ci-2-graph-content'),

        
    ])), 
    label='Proportions Playground',
)


# @callback(
#     Output('ci-over-N-graph-content', 'figure'),
#     Input('confidence_level', 'value'),
# )
def update_ci_over_N_plot(confidence_level):
    alpha = (100-confidence_level)/100
    max_N = 3500

    NN = np.arange(2, max_N, 2)

    low, high = proportion_confint(
        NN//2,  NN,
        method='wilson',
        alpha=alpha
    )

    # f = px.line(
    #     x=NN,
    #     y=((high-low)*100).round()
    # ) 
    f = go.Scatter(
        x=NN,
        y=((high-low)*100).round(2),
        name='Max width (%) of CI for different N'
    )
    return f
    
    
@callback(
    Output('ci-graph-content', 'figure'),
    [
        Input('total_N', 'value'),
        Input('confidence_level', 'value'),
    ]
)
def update_ci_width_plot(total_N, confidence_level):

    # if not total_N:
    #     return
    alpha = (100-confidence_level)/100

    N = total_N

    low, high = proportion_confint(
        np.arange(0,N+1), N,
        method='wilson',
        alpha=alpha
    )

    pd.DataFrame(dict(
        low=low,
        high=high,
        delta=high-low
    ))
    ci_data = pd.DataFrame(dict(
            low=low,
            high=high,
            delta=high-low
        ))
    
    
    fig = make_subplots(rows=1, cols=2)


    fig.add_trace(
        go.Scatter(
            y=100*(high-low), x=100*np.arange(0,N+1)/N,
            name='Width (%) of interval for <br>different proportion values at fixed N'
        ),
        row=1, col=1,

    )
    
    fig.add_trace(
        update_ci_over_N_plot(confidence_level),
        row=1, col=2

    )
    return fig
    # return px.line(y=100*(high-low), x=100*np.arange(0,N+1)/N,
    #                labels={
    #                    'y':  'Width of interval (%)',
    #                    'x': 'Proportion value (%)'
    #                })



from scipy.stats import binom
from statsmodels.stats.proportion import proportion_confint, confint_proportions_2indep

@callback(
    Output('ci-2-graph-content', 'figure'),
    [
        Input('total_N', 'value'),
        Input('confidence_level', 'value'),
        Input('prop-1', 'value'),
        Input('prop-2', 'value'),

    ]
)
def update_ci_2(N, confidence_level, p1, p2):
    alpha = (100-confidence_level)/100
    # ic(alpha)
    
    
    f = go.Figure(layout_xaxis_range=[0,120],layout_yaxis_range=[0,1.2])


    data = []



    for i, p in enumerate((p1,p2)):

        k = round(p*N/100)

        low, high = proportion_confint(
            k,  N,
            method='wilson',
            alpha=alpha
        )
        
        data.append(
            [low, k/N, high]
        )
        vv = np.linspace(low, high, 100)

        lh = binom.pmf(k, N, vv)

        # ic(lh)

        f.add_trace(
            go.Scatter(
                    x=vv*100,
                    y=lh / binom.pmf(k, N, k/N),
                name=f'likelihood {i+1}'
            ),
        )
        f.add_vline(x=low*100, line_dash="dash", line_color="green",
                annotation_text=f'{round(low*100,1)}', 
                annotation_position="top left",
                annotation_font_size=10,
    #               annotation_font_color="blue"
                )
        f.add_vline(x=high*100, line_dash="dash", line_color="green",
                            annotation_text=f'{round(high*100,1)}', 
                annotation_position="top left",
                annotation_font_size=10,)

        f.add_vrect(
            x0=low*100,
            x1=high*100,
    #         label=dict(
    #             text=f"CI width: {100*(high-low):.1f}%",
    #             textposition="bottom center",
    #             font=dict(size=20, family="Times New Roman"),
    #         ),
            fillcolor="green",
            opacity=0.25,
            line_width=0,
        )


    # f = px.line(
    #     x=NN,
    #     y=((high-low)*100).round()
    # )

    d = np.array(data)

    CI_diff = (100 * np.array(
                        confint_proportions_2indep(
                            round(p2*N/100), N,
                            round(p1*N/100), N
                        )
                    ))
    # ic(CI_diff)
    text_annot = f'''
    CI 1: {100*d[0].min():.1f} - {100*d[0].max():.1f}, width={100*(d[0].max() - d[0].min()):.1f} 
    <br>
    CI 2: {100*d[1].min():.1f} - {100*d[1].max():.1f}, width={100*(d[1].max() - d[1].min()):.1f} 
    <br>
    Diff (p2 - p1): <br>
    naive: [{(d1:=100*(d[1].min()-d[0].max())):.1f}, {(d2:=100*(d[1].max()-d[0].min())):.1f}], width={abs(d1-d2):.1f}
    <br>
    right: {CI_diff.round(1)}, width={abs(CI_diff[0]-CI_diff[1]):.1f}
    '''

    f.add_annotation(text=text_annot, 
                        align='left',
                        showarrow=False,
                        xref='paper',
                        yref='paper',
                        x=1.1,
                        y=0.8,
                        bordercolor='black',
                        borderwidth=1)
    return f