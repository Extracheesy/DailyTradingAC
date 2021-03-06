import sys
import os
import pandas as pd
import numpy as np

"""
Risk-aversion reflects whether an investor will choose to preserve the capital.
It also influences one’s trading strategy when facing different market volatility level.

To control the risk in a worst-case scenario, such as financial crisis of 2007–2008,
FinRL employs the financial turbulence index that measures extreme asset price fluctuation.
"""

def add_turbulence(df):
    """
    add turbulence index from a precalcualted dataframe
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    print("compute turbulence...")
    turbulence_index = calcualte_turbulence(df)
    df = df.merge(turbulence_index, on='date')
    df = df.sort_values(['date','tic']).reset_index(drop=True)
    return df

def calcualte_turbulence(df):
    """calculate turbulence index based on dow 30"""
    # can add other market assets

    df_price_pivot=df.pivot(index='date', columns='tic', values='Adj Close')
    unique_date = df.date.unique()
    # start after a year
    start = 33
    turbulence_index = [0]*start
    #turbulence_index = [0]
    count=0
    for i in range(start,len(unique_date)):
        current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
        hist_price = df_price_pivot[[n in unique_date[0:i] for n in df_price_pivot.index ]]
        cov_temp = hist_price.cov()
        current_temp=(current_price - np.mean(hist_price,axis=0))
        temp = current_temp.values.dot(np.linalg.inv(cov_temp)).dot(current_temp.values.T)
        if temp>0:
            count+=1
            if count>2:
                turbulence_temp = temp[0][0]
            else:
                #avoid large outlier because of the calculation just begins
                turbulence_temp=0
        else:
            turbulence_temp=0
        turbulence_index.append(turbulence_temp)


    turbulence_index = pd.DataFrame({'date':df_price_pivot.index,
                                     'turbulence':turbulence_index})
    return turbulence_index