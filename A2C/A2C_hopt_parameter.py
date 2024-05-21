import pandas as pd
import numpy as np
import os
import math
import pickle

from hyperopt import fmin, tpe, hp, Trials, space_eval
from A2C_train import *
from A2C_helper import *
from data_load import *




def objective(params):
    # print(params)
    model = A2Cagent(state_dim, action_dim, params)
    model.agent_mode('train')
    index = np.random.choice(range(len(states_train))) 
    for step in range(index, states_train.shape[0]-1):
        state = states_train[step]
        action, logs_probs = model.get_action(state)
        action = action.cpu()
        action = action.detach().numpy()
        logs_probs = logs_probs.cpu()
        logs_probs = logs_probs.detach().numpy()
        
        reward = -abs(action-train_y[step])          
        next_state = states_train[step+1]
        model.update(state, action, reward, logs_probs, next_state)
        
    model.agent_mode('eval')
    Test_actions = model.regression_prediction(states_val)
    Test_actions = torch.reshape(torch.tensor(Test_actions), (val_y.shape[0],1))
    Reward = model.critic_criterion(Test_actions, torch.tensor(val_y))
    print(Reward)
    return Reward.item()



space = {
    'Ahidden_dim': hp.quniform('Ahidden_dim', 2, 512,1),
    'Anum_layers': hp.quniform('Anum_layers', 1, 8,1),
    'Chidden_dim': hp.quniform('Chidden_dim', 2, 512, 1),
    'Cnum_layers': hp.quniform('Cnum_layers', 1, 8,1),
    'alr': hp.loguniform('alr', -8, -1),
    'clr': hp.loguniform('clr', -8, -1),
    'gamma': hp.uniform('gamma', 0.9, 0.99),
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
    
    with open(os.getcwd() + f'\\{dataset_name}' + '_A2C_best_params.pkl', 'wb') as file:
        pickle.dump(best, file)

