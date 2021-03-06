import sys
import os, fnmatch
import glob

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime

from  add_indicators import add_technical_indicator
from  split_data import split_data_train_val_trade

"""
values_cac40 = ['AI.PA', 'AIR.PA', 'ALO.PA', 'MT', 'ATOS', 'CS.PA', 'BNP.PA', 'EN.PA', 'CAP.PA', 'CA.PA',
                'ACA.PA', 'BN.PA', 'DSY.PA', 'ENGI.PA', 'EL.PA', 'RMS.PA', 'KER.PA', 'LR.PA', 'OR.PA', 'MC.PA',
                'ML.PA', 'ORA.PA', 'RI.PA', 'PUB.PA', 'RNO.PA', 'SAF.PA', 'SGO.PA', 'SAN.PA', 'SU.PA', 'GLE.PA',
                'STLA', 'STM.PA', 'TEP.PA', 'HO.PA', 'FP.PA', 'URW.AS', 'VIE.PA', 'DG.PA', 'VIV.PA', 'WLN.PA']
"""
# remove STLA
values_cac40 = ['AI.PA', 'AIR.PA', 'ALO.PA', 'MT', 'ATOS', 'CS.PA', 'BNP.PA', 'EN.PA', 'CAP.PA', 'CA.PA',
                'ACA.PA', 'BN.PA', 'DSY.PA', 'ENGI.PA', 'EL.PA', 'RMS.PA', 'KER.PA', 'LR.PA', 'OR.PA', 'MC.PA',
                'ML.PA', 'ORA.PA', 'RI.PA', 'PUB.PA', 'RNO.PA', 'SAF.PA', 'SGO.PA', 'SAN.PA', 'SU.PA', 'GLE.PA',
                'STM.PA', 'TEP.PA', 'HO.PA', 'FP.PA', 'URW.AS', 'VIE.PA', 'DG.PA', 'VIV.PA', 'WLN.PA']

value_apple = ['AAPL']


def clear_data_directory():

    listOfFilesToRemove = os.listdir('./data/')
    pattern = "*.csv"
    for entry in listOfFilesToRemove:
        if fnmatch.fnmatch(entry, pattern):
            print("remove : ",entry)
            os.remove("./data/" + entry)

def clear_model_directory():

    listOfFilesToRemove = os.listdir('./model_saved/A2C_Model')
    pattern = "*.zip"
    for entry in listOfFilesToRemove:
        if fnmatch.fnmatch(entry, pattern):
            print("remove : ",entry)
            print(os.getcwd())
            os.remove("./model_saved/A2C_Model/" + entry)

    listOfFilesToRemove = os.listdir('./model_saved/A2C_Model/results')
    pattern1 = "*.csv"
    pattern2 = "*.png"
    for entry in listOfFilesToRemove:
        if fnmatch.fnmatch(entry, pattern1) or fnmatch.fnmatch(entry, pattern2):
            print("remove : ",entry)
            os.remove("./model_saved/A2C_Model/results/" + entry)

    listOfFilesToRemove = os.listdir('./model_saved/PPO_Model')
    pattern = "*.zip"
    for entry in listOfFilesToRemove:
        if fnmatch.fnmatch(entry, pattern):
            print("remove : ",entry)
            print(os.getcwd())
            os.remove("./model_saved/PPO_Model/" + entry)

    listOfFilesToRemove = os.listdir('./model_saved/PPO_Model/results')
    pattern1 = "*.csv"
    pattern2 = "*.png"
    for entry in listOfFilesToRemove:
        if fnmatch.fnmatch(entry, pattern1) or fnmatch.fnmatch(entry, pattern2):
            print("remove : ",entry)
            os.remove("./model_saved/PPO_Model/results/" + entry)

    listOfFilesToRemove = os.listdir('./model_saved/SAC_Model')
    pattern = "*.zip"
    for entry in listOfFilesToRemove:
        if fnmatch.fnmatch(entry, pattern):
            print("remove : ",entry)
            print(os.getcwd())
            os.remove("./model_saved/SAC_Model/" + entry)

    listOfFilesToRemove = os.listdir('./model_saved/SAC_Model/results')
    pattern1 = "*.csv"
    pattern2 = "*.png"
    for entry in listOfFilesToRemove:
        if fnmatch.fnmatch(entry, pattern1) or fnmatch.fnmatch(entry, pattern2):
            print("remove : ",entry)
            os.remove("./model_saved/SAC_Model/results/" + entry)

    listOfFilesToRemove = os.listdir('./model_saved/TD3_Model')
    pattern = "*.zip"
    for entry in listOfFilesToRemove:
        if fnmatch.fnmatch(entry, pattern):
            print("remove : ",entry)
            print(os.getcwd())
            os.remove("./model_saved/TD3_Model/" + entry)

    listOfFilesToRemove = os.listdir('./model_saved/TD3_Model/results')
    pattern1 = "*.csv"
    pattern2 = "*.png"
    for entry in listOfFilesToRemove:
        if fnmatch.fnmatch(entry, pattern1) or fnmatch.fnmatch(entry, pattern2):
            print("remove : ",entry)
            os.remove("./model_saved/TD3_Model/results/" + entry)

def mk_directories():

    if not os.path.exists("./data"):
        os.makedirs("./data")
    else:
        clear_data_directory()

    if not os.path.exists("./model_saved"):
        os.makedirs("./model_saved")

    if not os.path.exists("./model_saved/A2C_Model"):
        os.makedirs("./model_saved/A2C_Model")

    if not os.path.exists("./model_saved/PPO_Model"):
        os.makedirs("./model_saved/PPO_Model")

    if not os.path.exists("./model_saved/SAC_Model"):
        os.makedirs("./model_saved/SAC_Model")

    if not os.path.exists("./model_saved/TD3_Model"):
        os.makedirs("./model_saved/TD3_Model")

    clear_model_directory()

def DownloadFromYahoo(values):
    for value in values:
        data_df = yf.download(value, period="max")
        filename = './data/' + value + '.csv'
        print(filename)
        data_df.to_csv(filename)

def DownloadFromYahooDailyData(values):
    for value in values:
        #nb_years = 5 * 52  # 5 * 12 months for 5 years
        #today = date.today()
        #start_date = today - timedelta(weeks=nb_years)
        #data = pdr.get_data_yahoo(ticker, start=start_date, end=today)

        # use "period" instead of start/end
        # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        # (optional, default is '1mo')
        # fetch data by interval (including intraday if period < 60 days)
        # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        # (optional, default is '1d')
        df_data = yf.download(value, period="60d", interval='15m')

        print("insert technical indicators: ")
        df_data = add_technical_indicator(df_data, value)

        # value only for period="60d", interval='15m' / - 1 due to macd nan on first row
        if (len(df_data)  == (2013)):
            filename = './data/' + value + "_Daily" + '.csv'
            print("data saved to csv: ", value)
            df_data.to_csv(filename)
        else:
            print("missing data: ",len(df_data) ," - ticker not saved: ", value)


def GetDataFrameFromYahoo(value):
    result = yf.Ticker(value)
    # print(result.info)
    hist = result.history(period="max")
    return hist

def GetDailyDataFrameFromYahoo(value):
    result = yf.Ticker(value)
    # print(result.info)
    #hist = result.history(period='5d', interval='1m')
    hist = result.history(period="1d", interval='1m')
    return hist

def GetDataFrameFromCsv(csvfile):
    dateparse = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d')
    dataframe = pd.read_csv(csvfile, parse_dates=[0], index_col=0, skiprows=0, date_parser=dateparse)
    # df.index.rename('Time',inplace=True)
    # openValues2 = df.sort_values(by='Time')['open'].to_numpy()
    dataframe = dataframe.dropna()  # remove incoherent values (null, ...)
    return dataframe

def DisplayFromDataframe(df, name):
    plt.figure(figsize=(15, 5))
    plt.plot(df[name])
    # plt.xticks(range(0, df.shape[0], 2000), df['Date'].loc[::2000], rotation=0)
    plt.ylabel(name, fontsize=18)
    plt.title(name, fontsize=20)
    plt.legend([name], fontsize='x-large', loc='best')
    plt.show()

###
### AS A SCRIPT
### python -m fdata.py
###

_usage_str = """
Options:
    [ --test, -t]
"""

def _usage():
    print(_usage_str)

def _test1():
    hist = GetDataFrameFromYahoo('AI.PA')
    print(hist)
    DisplayFromDataframe(hist, "Close")

def _test2():
    hist = GetDataFrameFromCsv('./data/AI.PA.csv')
    DisplayFromDataframe(hist, "Close")

#def _test3():
    #hist = GetDailyDataFrameFromYahoo('./data/AI.PA.csv')
#    hist = GetDataFrameFromCsv('./data/AI.PA.csv')
#    DisplayFromDataframe(hist, "Close")

def _test3():
    hist = GetDailyDataFrameFromYahoo('AI.PA')
    print(hist)
    DisplayFromDataframe(hist, "Close")

def _download(values):

    mk_directories()

    if values == "cac40":
        #DownloadFromYahoo(values_cac40)
        DownloadFromYahooDailyData(values_cac40)
    else:
        if values == "apple":
            #DownloadFromYahoo(value_apple)
            DownloadFromYahooDailyData(value_apple)

def _get_data_from_csv(tick):

    print(os.getcwd())
    filename = tick + "_Daily.csv"
    df = pd.read_csv("./data/" + filename)



def merge_full_datset():

    os.chdir("./data/")

    extension = 'Daily.csv'
    all_filenames = [i for i in glob.glob('*{}'.format(extension))]

    # combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
    # export to csv

    combined_csv = combined_csv.sort_values(['date'], ignore_index=True)

    del combined_csv['Unnamed: 0']

    combined_csv = combined_csv.reset_index(drop=True)

    os.chdir("./..")

    return combined_csv

def dataset_analyse(df):


    os.chdir("./data/")

    value_counts = df["tic"].value_counts(sort=True)
    df_val_counts = pd.DataFrame(value_counts)
    df_val_counts.columns = ['counts']
    df_val_counts['tic'] = df_val_counts.index
    df_val_counts = df_val_counts.reset_index(drop=True)

    os.chdir("./..")

    return df_val_counts



