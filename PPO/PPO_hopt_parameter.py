import pandas as pd
import numpy as np
import os
import math
import pickle

from hyperopt import fmin, tpe, hp, Trials, space_eval
from PPO_train import *
from PPO_helper import *
from data_load import *




def objective(params):
    # print(params)
    model = PPOagent(state_dim, action_dim, params)
    model.agent_mode('train')
    num_steps = 512
    Actions = []
    States = []
    Rewards = []
    Log_probs = []
    Values = []
    
    index = np.random.choice(range(len(states_train)))  
    for step in range(index, states_train.shape[0]-1):
        state_tensor = torch.FloatTensor(states_train[step]).reshape(1,-1).to(device)
        state_value = model.critic(state_tensor).to(device)
        States.append(state_tensor)
        Values.append(state_value)
        
        action, logs_probs, entropy = model.get_action(state_tensor)
        reward = -abs(action-train_y[step])
                       
        next_state = states_train[step+1]
        Actions.append(action)
        Rewards.append(reward)
        Log_probs.append(logs_probs)
    
    actions = torch.cat(Actions).to(device)
    states = torch.cat(States).to(device)
    values = torch.cat(Values).to(device)
    log_prob_old = torch.cat(Log_probs).to(device)
    returns = (model.calculate_returns(Rewards, model.gamma)).to(device)
    returns = torch.reshape(returns, (returns.shape[0], 1))
    advantages = (returns - values).to(device)
    advantages = ((advantages - advantages.mean()) / (advantages.std() + 1e-5)).to(device)
    
    model.update_policy(states, actions, log_prob_old, advantages, returns)
    model.agent_mode('eval')
    Test_actions = model.regression_prediction(states_val)
    Test_actions = torch.reshape(torch.tensor(Test_actions), (val_y.shape[0],1))
    Reward = model.MseLoss(Test_actions, torch.tensor(val_y))
    print(Reward)
    return Reward.item()



space = {
    'Ahidden_dim': hp.quniform('Ahidden_dim', 2, 256,1),
    'Anum_layers': hp.quniform('Anum_layers', 1, 8,1),
    'Chidden_dim': hp.quniform('Chidden_dim', 2, 256, 1),
    'Cnum_layers': hp.quniform('Cnum_layers', 1, 8,1),
    'alr': hp.loguniform('alr', -8, -1),
    'clr': hp.loguniform('clr', -8, -1),
    'gamma': hp.uniform('gamma', 0.9, 0.99),
    'PPO_epochs': hp.quniform('PPO_epochs', 5, 50, 5),
    'val_coeff': hp.uniform('val_coeff', 0.5, 1.0),
    'ent_coeff': hp.uniform('ent_coeff', 0.01, 0.1),
    'Aact_fn': hp.choice('Aact_fn', ['relu', 'tanh', 'sigmoid']),
    'Adr': hp.uniform('Adr', 0, 0.5),
    'Cact_fn': hp.choice('Cact_fn', ['relu', 'tanh', 'sigmoid']),
    'Cdr': hp.uniform('Cdr', 0, 0.5)
}

########################################################################################

datasets = ['EURUSD', 'AUDUSD', 'GBPUSD', 'CNYUSD', 'CADUSD']
for dataset_name in datasets:
    covp, iw = [], []
    print(f"\nDataset: {dataset_name}")
    data_dir = f'datasets/{dataset_name}.csv'
    # Reading the data
    
    
    df = pd.read_csv(data_dir, index_col='Date')
    df.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1, inplace=True)
    
    # print(df)
    # length of time series data
    L = int(df.shape[0])
    
    # length of train and validation data
    len_train = int(math.ceil(L * .8))
    len_test = int(math.ceil(L * .1))
    len_val = L - len_train - len_test
    
    # splitting the original time series into train, validation, and test data
    df_train = df[0: len_train]
    df_val = df[len_train: len_train + len_val]
    df_test = df[len_train + len_val: L]
    
    # train, validation, and test data values using their DataFrames
    train_data = df_train.values
    val_data = df_val.values
    test_data = df_test.values
    
    
    # ###############################################################################
    # look_back is the time-horizon taken to predict one-day ahead prediction
    look_back = 20
    
    states_train, states_val, train_y, val_y = one_dimen_transform(train_data, val_data, look_back)
    _, states_test, _, test_y = one_dimen_transform(val_data, test_data, look_back)


    state_dim = states_train.shape[1]
    action_dim = 1
    action_low = 0.0
    action_high = 1.0   
    
        
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=500, trials=Trials())
    best['Aact_fn'] = ['relu', 'tanh', 'sigmoid'][best['Aact_fn']]
    best['Cact_fn'] = ['relu', 'tanh', 'sigmoid'][best['Cact_fn']]
    
    output_loc = f"\\{dataset_name}\\"

    if not os.path.exists(output_loc):
        os.makedirs(output_loc)
    
    with open(os.getcwd() + f'\\{dataset_name}'+'_PPO_best_params.pkl', 'wb') as file:
        pickle.dump(best, file)

