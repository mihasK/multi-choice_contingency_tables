import pandas as pd
from .utils import is_multi_value_category, _explode
from icecream import ic
from enum import StrEnum, auto
import math

def _log(x):
    return x
    # if x is None:
    #     return None
    # if x ==0:
    #     return float('-inf')
    # else:
    #     return math.log(ic(x))
class CBD_Types(StrEnum):
    
    COUNTS = auto()
    
    PERCENTS = auto()
    
    PERCENT_DEVIATIONS = auto()
    
    RISK_DIFFERENCE = auto()
    RISK_RATIO = auto()
    
    ODDS_RATIO = auto()
    
    CHI2_TEST_INDEPENDENCE = auto()
    
    CRAMER_V_ASSOCIATION = auto()
    
    # Cohen's omega (ω)https://en.wikipedia.org/wiki/Effect_size#Cohen's_omega_(%CF%89)
    # Cohen's h
    

import scipy.stats.contingency
import scipy.stats
from statsmodels.stats.proportion import proportion_confint, confint_proportions_2indep
import numpy as np

def _calc_proportion(_df, feature, value, ci=False) ->dict:
    t = _df.shape[0]
    
    _df = _explode(_df, feature)
    
    n =  _df[
        _df[feature] == value
    ].shape[0]
    
    
    ci_value = proportion_confint(
        n, t,
        method='wilson'
    )
    ci_value = (100 * np.array(ci_value)).round(2)

    return {
        'percent': round(100*n/t, 2),
        'freq': n,
        'CI': ci_value
    }


def get_cross_breakdown(df, f1,f2, cbd_type=CBD_Types.PERCENTS) -> pd.DataFrame:
    
    fdf = df[
        ~df[[f1,f2]].isnull().any(axis=1)
    ]
    
    total_N = fdf.shape[0]
    
    # shrink to only these 2 features
    fdf = fdf[list({f1,f2})]  # usage of set handles case of f1==f2
    
    
    row_headers = sorted(
        _explode(fdf, f1)[f1].unique()
    )

    table = pd.DataFrame(
        index=row_headers
    )

    total_column = table.index.map(
        lambda f1_value: _calc_proportion(fdf, f1, f1_value,)
    )
    total_column = pd.Series(total_column, index=table.index)


    f2_options = list(sorted(_explode(fdf, f2)[f2].unique()))

    for f2_value in f2_options:
        # df with f2 eq f2_value
        # alternative code: f2_value_df = _explode(fdf, f2).query('@f2 == ')
        f2_value_df = _explode(fdf, f2)
        f2_value_df = f2_value_df[f2_value_df[f2] == f2_value]
        non_f2_value_df = fdf.drop(f2_value_df.index, axis=0)
        
        
        header_name = f'{f2_value} ({len(f2_value_df)})'
        
        __df = _explode(f2_value_df, feature=f1)
        
        def _calc_for_cell(f1_value):  # will be applied for each f1_value
            prop_info = _calc_proportion(
                f2_value_df, f1, f1_value,
            )
            
            res = {**prop_info}
            # print(total_column)
            overall = total_column[f1_value]
            res['percent_diff'] = prop_info['percent'] - overall['percent']
            
            
            freq = (__df[f1] == f1_value).sum()
            res['freq'] = freq
            
            res['percent_diff_CI']  = (100 * np.array(
                confint_proportions_2indep(*(
                    freq, f2_value_df.shape[0], 
                    overall['freq'], total_N
                ))
            )).round(1)
            
            
            # Risk Difference: risk of f1_value among f2_value versus risk of f1_value among (not f2_value)
            
            freq_outside = (_explode(non_f2_value_df, f1)[f1] == f1_value).sum()
            total_outside = len(non_f2_value_df)
            total_inside = len(f2_value_df)
            rd = (
                freq / total_inside
                -
                freq_outside/total_outside
            )
            rd = round(100*rd, 1)
            
            res['RD'] = {
                'exact': rd,
                'CI': (100 * np.array(
                    confint_proportions_2indep(
                        freq, total_inside,
                        freq_outside, total_outside
                    )
                )).round(1)
            }
            
            rr = scipy.stats.contingency.relative_risk(
                freq, total_inside, freq_outside, total_outside
            )
            rr_ci = rr.confidence_interval()
            
            res['RR'] = {
                'exact': rr.relative_risk,
                'CI': (rr_ci.low, rr_ci.high),
            }
            
            odds_ratio = scipy.stats.contingency.odds_ratio(
                [
                        [freq, freq_outside],
                        [total_inside-freq, total_outside-freq_outside],
                ]
            )
            odds_ratio_ci = odds_ratio.confidence_interval()
            
            res['ODDS_RATIO'] = {
                'exact': _log(odds_ratio.statistic),
                'CI': (
                    _log(odds_ratio_ci.low), 
                    _log(odds_ratio_ci.high),
                )   
            }
            # ic(f1_value, f2_value)
            chi2 = scipy.stats.chi2_contingency(
                (np.array([
                    [freq, freq_outside],
                    [total_inside-freq, total_outside-freq_outside],
                ])),
                correction=True
            )
            
            res['CHI2'] = {
                'pvalue': (chi2.pvalue),
            }
            
            from scipy.stats.contingency import association
            
            ass = association(
                np.array([
                    [freq, freq_outside],
                    [total_inside-freq, total_outside-freq_outside],
                ]),
                correction=True,
                # method='tschuprov'
                    
            )
            res['CRAMER'] = ass
            
            
            return res
            
        table[header_name] = table.index.map(_calc_for_cell)
        
        # if cbd_type == CBD_Types.PERCENT_DEVIATIONS:
        #     table[header_name] -= total_column
        
        
        


    table[f'Total ({fdf.shape[0]})'] = total_column
    
    return table