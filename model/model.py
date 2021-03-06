import time, os, sys
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

def trade_data(df_data_day, model):

    turbulence_threshold = 140
    initial = True

    print("Env Trade:")
    env_trade = DummyVecEnv([lambda: StockEnvTrade(df_data_day)])
    obs_trade = env_trade.reset()


    for i in range(len(df_data_day.index.unique())):
        action, _states = model.predict(obs_trade)
        obs_trade, rewards, dones, info = env_trade.step(action)
        #if i == (len(trade_data.index.unique()) - 2):
        #    print(env_test.render())
        #    last_state = env_trade.render()
