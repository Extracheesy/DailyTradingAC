import time, os, sys
import pandas as pd
import numpy as np

from stable_baselines import A2C
from stable_baselines import PPO2
from stable_baselines import SAC
from stable_baselines import TD3
from stable_baselines.common.vec_env import DummyVecEnv
from load_yfinance_data import clear_model_directory

from EnvMultipleStock_trade import StockEnvTrade
from EnvMultipleStock_train import StockEnvTrain

def train_A2C(env_train, model_name, timesteps=100000):

    # train A2C model
    os.chdir("./model_saved/")
    start = time.time()
    print("Train A2C Model with MlpPolicy: ")

    model = A2C('MlpPolicy', env_train, verbose=0)
    print("A2C Learning time steps: ",timesteps)

    model.learn(total_timesteps=timesteps)
    print("A2C Model learning completed: ")

    end = time.time()
    timestamp = time.strftime('%b-%d-%Y_%H%M')
    model_file_name = (model_name + timestamp)
    model.save(model_file_name)
    print("A2C Model save completed     :")
    print('Training time A2C: ', (end - start) / 60, ' minutes')

    os.chdir("./..")

    return model

def train_PPO(env_train, model_name, timesteps=100000):

    # train PPO model
    os.chdir("./model_saved/")
    start = time.time()
    print("Train PPO Model with MlpPolicy: ")

    model = PPO2('MlpPolicy', env_train, verbose=0)
    print("PPO Learning time steps: ",timesteps)

    model.learn(total_timesteps=timesteps)
    print("PPO Model learning completed: ")

    end = time.time()
    timestamp = time.strftime('%b-%d-%Y_%H%M')
    model_file_name = (model_name + timestamp)
    model.save(model_file_name)
    print("PPO Model save completed     :")
    print('Training time PPO: ', (end - start) / 60, ' minutes')

    os.chdir("./..")

    return model

def train_SAC(env_train, model_name, timesteps=100000):

    # train SAC model
    os.chdir("./model_saved/")
    start = time.time()
    print("Train SAC Model with MlpPolicy: ")

    model = SAC('MlpPolicy', env_train, verbose=0)
    print("SAC Learning time steps: ",timesteps)
    model.learn(total_timesteps=timesteps)
    print("SAC Model learning completed: ")

    end = time.time()
    timestamp = time.strftime('%b-%d-%Y_%H%M')
    model_file_name = (model_name + timestamp)
    model.save(model_file_name)
    print("SAC Model save finish     :")
    print('Training time SAC: ', (end - start) / 60, ' minutes')
    os.chdir("./..")

    return model

def train_TD3(env_train, model_name, timesteps=100000):

    # train TD3 model
    os.chdir("./model_saved/")
    start = time.time()
    print("Train TD3 Model with MlpPolicy: ")

    model = TD3('MlpPolicy', env_train, verbose=0)
    print("TD3 Learning time steps: ",timesteps)

    model.learn(total_timesteps=timesteps)
    print("TD3 Model learning completed: ")
    end = time.time()
    timestamp = time.strftime('%b-%d-%Y_%H%M')
    model_file_name = (model_name + timestamp)
    model.save(model_file_name)
    print("TD3 Model save finish     :")
    print('Training time TD3: ', (end - start) / 60, ' minutes')

    os.chdir("./..")

    return model

def model_training_learning(env_train, model_name, timesteps=100000):

    # train model
    os.chdir("./model_saved/" + model_name)
    start = time.time()
    print("Train ", model_name ," Model with MlpPolicy: ")

    if model_name == "A2C_Model":
        model = A2C('MlpPolicy', env_train, verbose=0)
    elif model_name == "PPO_Model":
        model = PPO2('MlpPolicy', env_train, verbose=0)
    elif model_name == "TD3_Model":
        model = TD3('MlpPolicy', env_train, verbose=0)
    elif model_name == "SAC_Model":
        model = SAC('MlpPolicy', env_train, verbose=0)

    print("Learning ",model_name," time steps: ",timesteps)

    model.learn(total_timesteps=timesteps)
    print("TD3 Model learning completed: ")
    end = time.time()
    timestamp = time.strftime('%b-%d-%Y_%H%M')
    model_file_name = (model_name + timestamp)
    model.save(model_file_name)
    print("- ",model_name," save finish     :")
    print("Training time  ",model_name," : ", (end - start) / 60, " minutes")

    os.chdir("./..")
    os.chdir("./..")
    return model

def train_model(df_train_data):

    clear_model_directory()

    A2C_File_Name = "A2C_Model"
    PPO_File_Name = "PPO_Model"
    TD3_File_Name = "TD3_Model"
    SAC_File_Name = "SAC_Model"

    ############## Environment Setup starts ##############
    env_train = DummyVecEnv([lambda: StockEnvTrain(df_train_data)])
    model_a2c = model_training_learning(env_train, model_name=A2C_File_Name, timesteps=100000)

    env_train = DummyVecEnv([lambda: StockEnvTrain(df_train_data)])
    model_ppo = model_training_learning(env_train, model_name=PPO_File_Name, timesteps=100000)

    env_train = DummyVecEnv([lambda: StockEnvTrain(df_train_data)])
    model_td3 = model_training_learning(env_train, model_name=TD3_File_Name, timesteps=100000)

    env_train = DummyVecEnv([lambda: StockEnvTrain(df_train_data)])
    model_sac = model_training_learning(env_train, model_name=SAC_File_Name, timesteps=100000)

    return model_a2c

def read_model(model_type):

    if model_type == "A2C":
        model = A2C.load("./model_saved/Selected/A2C_ModelMar-05-2021_0815/A2C_ModelMar-05-2021_0815")
    if model_type == "TD3":
        model = TD3.load("./model_saved/Selected/TD3_ModelMar-05-2021_1442/TD3_ModelMar-05-2021_1442")

    return model

def addRow(df,ls):
    """
    Given a dataframe and a list, append the list as a new row to the dataframe.

    :param df: <DataFrame> The original dataframe
    :param ls: <list> The new row to be added
    :return: <DataFrame> The dataframe with the newly appended row
    """

    numEl = len(ls)

    newRow = pd.DataFrame(np.array(ls).reshape(1,numEl), columns = list(df.columns))

    df = df.append(newRow, ignore_index=True)

    return df

def init_lst_stks(nb_ticker):
    lst = []
    for i in range(nb_ticker):
        lst.append(0)
    return lst

def trade_data(df_data_day, model, df_trace):

    turbulence_threshold = 140
    initial = True

    print("Env Trade:")
    env_trade = DummyVecEnv([lambda: StockEnvTrade(df_data_day)])
    obs_trade = env_trade.reset()

    list_unique = df_data_day.datadate.unique()

    nb_ticker = int( (len(df_trace.columns) - 4)/4)

    i = 0
    list_previous_nb_stk = init_lst_stks(nb_ticker)
    #list_previous_price_stk = init_lst_stks(nb_ticker)
    list_data_date = []
    list_previous_price_stk = []
    list_data_date.append(list_unique[i])
    list_data_date.append(obs_trade[0][0])
    stock_value = 0
    for iter in range(0, nb_ticker, 1):
        list_data_date.append(0)
        list_data_date.append(obs_trade[0][iter + 1])
        list_previous_price_stk.append(obs_trade[0][iter + 1])
        list_data_date.append(obs_trade[0][iter + 1 + nb_ticker])
        stock_value = stock_value + (obs_trade[0][iter + 1]) * (obs_trade[0][iter + 1 + nb_ticker])
        # diff of nb stock own * actual stock price => (_nb - previous_nb) * actual stock price
        list_data_date.append(0)  # stk_flow

    list_data_date.append(stock_value)
    list_data_date.append(stock_value + obs_trade[0][0])
    list_data_date.append(0)  # my_rewards

    df_trace = addRow(df_trace, list_data_date)



    for i in range(0, len(df_data_day.index.unique()), 1):
        list_data_date = []

        action, _states = model.predict(obs_trade)
        obs_trade, rewards, dones, info = env_trade.step(action)

        #if i == (len(trade_data.index.unique()) - 2):
        #    print(env_test.render())
        #    last_state = env_trade.render()

        if dones == True:
            total_rewards = rewards[0]
            coef = 1
        else:
            total_rewards = 0
            coef = 10000

        list_data_date.append(list_unique[i])     # date
        list_data_date.append(obs_trade[0][0] + total_rewards)    # account
        stock_value = 0
        for iter in range(0,nb_ticker,1):
            list_data_date.append(action[0][iter])
            list_data_date.append(obs_trade[0][iter + 1])
            list_data_date.append(obs_trade[0][iter + 1 + nb_ticker])
            # list_data_date.append(stock_flow) # stk_flow
            stock_value = stock_value +  (obs_trade[0][iter + 1]) * (obs_trade[0][iter + 1 + nb_ticker])
            # diff of nb stock own * actual stock price => (_nb - previous_nb) * actual stock price
            toto = list_previous_nb_stk[iter]
            tutu = obs_trade[0][iter + 1 + nb_ticker]
            tata = obs_trade[0][iter + 1]
            titi = (obs_trade[0][iter + 1 + nb_ticker] - list_previous_nb_stk[iter]) * obs_trade[0][iter + 1]

            list_data_date.append((list_previous_nb_stk[iter] - obs_trade[0][iter + 1 + nb_ticker]) * list_previous_price_stk[iter])  # stk_flow
            list_previous_nb_stk[iter] = obs_trade[0][iter + 1 + nb_ticker]
            list_previous_price_stk[iter] = obs_trade[0][iter + 1]


        list_data_date.append(stock_value)                       # stocks_$
        list_data_date.append(stock_value + obs_trade[0][0] + total_rewards)     # total_val_$
        list_data_date.append(rewards[0] * coef)    # my_rewards

        df_trace = addRow(df_trace,list_data_date)

    return df_trace