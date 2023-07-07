import pandas as pd
import numpy as np
from icecream import ic

def _is_list(obj):
    # ic(isinstance(obj, np.ndarray), obj)
    if obj is None or obj is np.nan: #pd.isnull(obj):

        return True
    return isinstance(obj, (
        list, tuple, set, np.ndarray
    ))

def is_multi_value_category(series: pd.Series):
    return series.apply(_is_list).all()


def _explode(_df, feature):
    if is_multi_value_category(_df[feature]):
        return _df.explode(feature)
    return _df

def unique_exploded(_df, feature):
    return _explode(_df, feature)[feature].unique()

def get_all_values(_df, feature):
    
    return _explode(_df, feature)[feature].unique()

    
def take_closest_to_zero(pair: list):
    if pair[0] * pair[1] < 0:
        return 0
    else:
        return min(pair, key=abs)
    
class AnyOrNone(object):  # The wrapper is not type-specific
    def __init__(self, value):
        self.value = value

    def __format__(self, *args, **kwargs):
        if self.value is None:
            return "NAN"
        else:
            return self.value.__format__(*args, **kwargs)
        
        



def _detect_column_type(c, df):
    if df[c].dtype == 'category':
        return 'category'
    elif is_multi_value_category(df[c]):
        return 'multi'
    
    return str(df[c].dtype)
        
    
    
def get_column_types(df): 
    return {
        name: _detect_column_type(name, df)
        for name in df.columns
    }