from load_yfinance_data import merge_full_datset
from load_yfinance_data import dataset_analyse
from split_data import split_data_train_val_trade
from split_data import data_factorize_index
from split_data import save_csv_file
from split_data import read_csv_file
from split_data import format_df
from split_data import get_one_day_data
from split_data import drop_daydate_column
from turbulence import add_turbulence
from model import train_model
from model import trade_data
from model import read_model

def run_trading_strategy(preprocess_data):

    COMPUTE_MODEL = False

    ############## Preprocess data ##############
    if (preprocess_data == True):
        df_full_data = merge_full_datset()

        df_full_data = add_turbulence(df_full_data)

        df_data_analysis = dataset_analyse(df_full_data)

        df_train_data, df_trade_data = split_data_train_val_trade(df_full_data)

        save_csv_file(df_full_data,"full_data")
        save_csv_file(df_train_data,"train_data")
        save_csv_file(df_trade_data,"trade_data")
        save_csv_file(df_data_analysis,"tick_val_count_csv")
    else:
        # df_full_data = read_csv_file("full_data")
        # df_data_analysis = read_csv_file("tick_val_count_csv")
        df_train_data = read_csv_file("train_data")
        df_trade_data = read_csv_file("trade_data")

    df_train_data = format_df(df_train_data)
    df_train_data = drop_daydate_column(df_train_data)
    df_trade_data = format_df(df_trade_data)

    if(COMPUTE_MODEL == True):
        # Train the model
        trained_model = train_model(df_train_data)
    else:
        # Read saved model
        #trained_model = read_model("A2C")
        trained_model = read_model("TD3")

    df_trade_date_unique = df_trade_data["daydate"].unique()
    for date_trading in df_trade_date_unique:
        df_data_day = get_one_day_data(df_trade_data, date_trading)
        df_data_day = drop_daydate_column(df_data_day)
        print("trading date: ",date_trading)

        trade_data(df_data_day, trained_model)
