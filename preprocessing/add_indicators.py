import sys
import pandas as pd
import numpy as np

from stockstats import StockDataFrame as Sdf

def add_technical_indicator(df, tic):
    """
    calcualte technical indicators
    use stockstats package to add technical inidactors
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """

    df['date'] = df.index
    df = df.reset_index(drop=True)
    cols = ['date'] + [col for col in df if col != 'date']
    df = df[cols]

    # drop duplicates
    df = df.drop_duplicates()

    # convert Date column to datetime
    df['date'] = pd.to_datetime(df['date'], format = '%Y-%m-%d')
    # df['date'] = pd.to_datetime(df['date'])


    # sort by datetime
    df.sort_values(by = 'date', inplace = True, ascending = True)

    stock = Sdf.retype(df.copy())

    temp_macd = stock['macd']
    temp_macds = stock['macds']
    temp_macdh = stock['macdh']
    macd = pd.DataFrame(temp_macd)
    macds = pd.DataFrame(temp_macds)
    macdh = pd.DataFrame(temp_macdh)

    temp_rsi = stock['rsi_6']
    rsi = pd.DataFrame(temp_rsi)

    temp_cci = stock['cci']
    cci = pd.DataFrame(temp_cci)

    temp_adx = stock['adx']
    adx = pd.DataFrame(temp_adx)

    temp_pdi = stock['pdi']
    temp_mdi = stock['mdi']
    pdi = pd.DataFrame(temp_pdi)
    mdi = pd.DataFrame(temp_mdi)

    df.insert(len(df.columns), "daydate",0)
    df.insert(len(df.columns), "tic",tic)

    df.insert(len(df.columns), "macd",0)
    df.insert(len(df.columns), "macd_signal_line",0)
    df.insert(len(df.columns), "macd_hist",0)

    df.insert(len(df.columns), "rsi",0)

    df.insert(len(df.columns), "cci",0)

    df.insert(len(df.columns), "adx",0)

    df.insert(len(df.columns), "+DI",0)
    df.insert(len(df.columns), "-DI",0)

    len_df = len(df)
    for i in range(0,len_df,1):

        df.loc[i,"daydate"] = str(df.iloc[i]["date"])[0:10]

        df.loc[i,"macd"] = macd.iloc[i][0]
        df.loc[i,"macd_signal_line"] = macds.iloc[i][0]
        df.loc[i,"macd_hist"] = macdh.iloc[i][0]

        df.loc[i,"rsi"] = rsi.iloc[i][0]

        df.loc[i,"cci"] = cci.iloc[i][0]

        df.loc[i,"adx"] = adx.iloc[i][0]

        df.loc[i,"+DI"] = pdi.iloc[i][0]
        df.loc[i,"-DI"] = mdi.iloc[i][0]

    df['daydate'] = pd.to_datetime(df['daydate'], format = '%Y-%m-%d')


    cols = ['daydate'] + ['date'] + ['tic'] + [col for col in df if ((col != 'date') and (col != 'daydate') and (col != 'tic'))]
    df = df[cols]

    #df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    df = df.reset_index(drop=True)

    return df