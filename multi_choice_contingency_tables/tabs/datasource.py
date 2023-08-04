from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from icecream import ic
from dash import dash_table
from multi_choice_contingency_tables import utils
from multi_choice_contingency_tables.utils import AnyOrNone
from multi_choice_contingency_tables import cross_bd
import os
import json
import dash_daq as daq
import dash_bootstrap_components as dbc
from dash import dash_table
from collections import namedtuple
from itertools import chain

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR = os.path.join(BASE_DIR, '..', 'data/')

DATA_DIR = 'multi_choice_contingency_tables/data'

files = os.listdir(DATA_DIR)
# ic(files)
files = [f for f in files if (
    f.endswith('pickle') 
    or
    f.endswith('csv')
    )
]

ic(files)

def _shrink_ext_from_fname(filename):
    return ''.join(
        filename.split('.')[:-1]
    )

def load_df(file_name: str):
    p = os.path.join(DATA_DIR, file_name)
    
    if file_name.endswith('.pickle'):
        return pd.read_pickle(p)
    elif file_name.endswith('.csv'):
        df = pd.read_csv(p).convert_dtypes()
        return df


dataframes = {
    _shrink_ext_from_fname(f): load_df(file_name=f)
    for f in files
}



data_view = dash_table.DataTable(
    # columns=[]
    style_table={'overflowX': 'auto'},

    # style_data={
    #     # 'whiteSpace': 'normal',
    #     # 'height': 'auto',
    #     'overflow': 'hidden',
    #     'textOverflow': 'ellipsis',
    #     'maxWidth': 0
        
        
    # },
)


@callback([
    Output(data_view, 'columns'),
    Output(data_view, 'data')
    ],
          Input('data_selector', 'value')
)
def update_data_view(data_name):
    df = dataframes[data_name]
    
    column_types  =  utils.get_column_types(df)
    
    headers = ('Feature', 'Value', 'Translation', 'Type', 'UniqValues')
    Row = namedtuple('Row', headers)
    
    
    def _row_for_f(f_name):
        return Row(
            Feature=f_name,
            Value='',
            Translation='',
            Type=column_types[f_name],
            UniqValues=len(utils._explode(df, f_name)[f_name].unique())
        )._asdict()
    
    def _row_for_v(value, f_name):
        return Row(
            Feature='',
            Value=value,
            Translation='',
            Type=''
        )._asdict()
    
    def _all_rows_v(f_name):
        return [
            _row_for_v(value, f_name)
            for value in utils._explode(df, f_name)[f_name].unique()
        ]
    return (
        [dict(id=h,name=h) for h in headers],
        list(chain(*(
            [_row_for_f(f_name)] + []#_all_rows_v(f_name)
            for f_name in df.columns
        )))
    )
    


data_tab_content = dbc.Card(dbc.CardBody([
    
            html.H3('Select data'),
            dcc.Dropdown(
                list(map(_shrink_ext_from_fname, files)), 
                _shrink_ext_from_fname(files[0]),
                id='data_selector'
            ),
            data_view
        ]),
    className="mt-3",
)



# df = df.infer_objects(copy=False)