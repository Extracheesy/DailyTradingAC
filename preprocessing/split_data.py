import os
import pandas as pd

def save_csv_file(df,filename):

    os.chdir("./data_processed/")

    if(os.path.exists(filename + ".csv")):
        print("remove: ",filename + ".csv")
        os.remove(filename + ".csv")

    print("save in ./data_processed/ : ", filename + ".csv")
    df.to_csv(filename + ".csv", index=False, encoding='utf-8-sig')

    os.chdir("./..")

def read_csv_file(filename):

    os.chdir("./data_processed/")

    if(os.path.exists(filename + ".csv")):
        df = pd.read_csv(filename + ".csv")
    else:
        print("missing data: ",filename + ".csv")

    os.chdir("./..")

    return df

def data_split(df,start,end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df.daydate >= start) & (df.daydate <= end)]
    data = data.sort_values(['date'], ignore_index=True)
    data.index = data.date.factorize()[0]
    return data

def data_factorize_index(data):

    data=data.sort_values(['datadate','tic'],ignore_index=True)
    data.index = data.datadate.factorize()[0]
    return data

def split_data_train_val_trade(df):

    # val split not implemented

    df_unique = df["daydate"].unique()
    df_len = len(df_unique)
    #print("train split: ", df_unique.iloc[0]["daydate"], " to: ", df_unique.iloc[int(df_len/3)]["daydate"])
    end_date_train_data = int(df_len*2/3)
    ### 2/3 of the data for training purpose
    print("train split: ", df_unique[0], " to: ", df_unique[end_date_train_data])
    df_train_data = data_split(df, df_unique[0], df_unique[end_date_train_data])
    ### 1/3 of the data for trading purpose
    print("trade split: ", df_unique[end_date_train_data + 1], " to: ", df_unique[df_len - 1])
    df_trade_data = data_split(df, df_unique[end_date_train_data + 1], df_unique[df_len - 1])

    return df_train_data, df_trade_data

def get_one_day_data(df, date):

    data = df[(df.daydate >= date) & (df.daydate <= date)]
    data = data.sort_values(['datadate'], ignore_index=True)
    data.index = data.datadate.factorize()[0]
    return data

def format_df(df):

    df = df.rename({'Adj Close': 'adjcp'}, axis=1)
    df = df.rename({'Open': 'open'}, axis=1)
    df = df.rename({'High': 'high'}, axis=1)
    df = df.rename({'Low': 'low'}, axis=1)
    df = df.rename({'Volume': 'volume'}, axis=1)
    df = df.rename({'date': 'datadate'}, axis=1)

    # df.drop(['daydate'], axis=1, inplace=True)
    df.drop(['Close'], axis=1, inplace=True)
    df.drop(['macd_signal_line'], axis=1, inplace=True)
    df.drop(['macd_hist'], axis=1, inplace=True)
    df.drop(['+DI'], axis=1, inplace=True)
    df.drop(['-DI'], axis=1, inplace=True)

    df = data_factorize_index(df)

    return df

def drop_daydate_column(df):

    df.drop(['daydate'], axis=1, inplace=True)
    return df
