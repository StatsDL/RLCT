import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import torch
import pickle

from PPO_train import *
from PPO_helper import *
from data_load import *



device = 'cpu'

def coverage(intervals, target):

    lower, upper = intervals[:, 0].reshape(-1,1), intervals[:, 1].reshape(-1,1)
    horizon_coverages = np.logical_and(target >= lower, target <= upper)

    return horizon_coverages, np.all(horizon_coverages, axis=1)

def get_critical_scores(calibration_scores, q):

    return np.transpose(np.array([
        np.percentile(position_calibration_scores, q * 100)
        for position_calibration_scores in calibration_scores
    ]))



class RLCT:
    
    def __init__(self, alpha, agent):
    
        self.agent = agent
        self.alpha = alpha
        self.calibration_scores = None
        self.critical_calibration_scores = None
        self.corrected_critical_calibration_scores = None
        
    def nonconformity(self, output, action):
        return np.abs(output - action)
    
    
    def fit(self, s, states_train, train_y, states_cal, cal_y, MAX_EPISODES, action_low, action_high):

        initial_mse = 1.0
        Reward = []
        Final_Score = []
        for episode  in range(MAX_EPISODES):
            index = np.random.choice(range(len(states_train)))
            num_steps = 512
            Actions = []
            States = []
            Rewards = []
            Log_probs = []
            Values = []
            ep_reward = 0
            self.agent.agent_mode('train')
            for step in range(index, states_train.shape[0]-1): 
                state_tensor = torch.FloatTensor(states_train[step]).reshape(1,-1).to(device)
                state_value = self.agent.critic(state_tensor).to(device)
                States.append(state_tensor)
                Values.append(state_value)
                
                action, logs_probs, entropy = self.agent.get_action(state_tensor)        
                reward = -abs(action-train_y[step])
                ep_reward += reward 
                        
                next_state = states_train[step+1]
                Actions.append(action)
                Rewards.append(reward)
                Log_probs.append(logs_probs)
                
            actions = torch.cat(Actions).to(device)
            states = torch.cat(States).to(device) 
            values = torch.cat(Values).to(device)
            log_prob_old = torch.cat(Log_probs).to(device)
            returns = (self.agent.calculate_returns(Rewards, self.agent.gamma)).to(device)
            returns = torch.reshape(returns, (returns.shape[0],1))
            advantages = (returns - values).to(device)
            advantages = ((advantages - advantages.mean()) / (advantages.std() + 1e-5)).to(device)
            
            self.agent.update_policy(states, actions, log_prob_old, advantages, returns)
                           
            print('Episode %d : %.2f'%(episode+1,ep_reward))
            Reward.append(ep_reward)
            
            self.agent.agent_mode('eval')
            Test_actions = self.agent.regression_prediction(states_val)
            Test_actions = torch.reshape(torch.tensor(Test_actions), (val_y.shape[0],1))
            final_result = self.agent.MseLoss(Test_actions, torch.tensor(val_y))
            print('Episode %d final result: %.2f'%(episode+1, final_result))
            
            if final_result <= initial_mse:
                model_states = {
                                'policy': self.agent.policy.state_dict(),
                                'critic': self.agent.critic.state_dict()  }
                torch.save(model_states, os.getcwd() + f'\\{dataset_name}'+'_PPO_best_agent.pth')
                initial_mse = final_result
        
        
        model_states = torch.load(f'{dataset_name}/PPO_best_agent_{s}.pth')
        
        self.agent.policy.load_state_dict(model_states['policy'])
        self.agent.critic.load_state_dict(model_states['critic'])
        
        self.calibrate(states_cal, cal_y)
        
    def calibrate(self, states_cal, cal_y):
        
        self.agent.agent_mode('eval')
        cal_actions = self.agent.regression_prediction(states_cal)
        cal_actions = torch.reshape(torch.tensor(cal_actions), (cal_y.shape[0],1))
        
        score = self.nonconformity(cal_actions, cal_y)
        self.calibration_scores = np.transpose(np.array(score))
        
        
        q = min(((len(cal_y) + 1.0) * (1 - self.alpha) / len(cal_y)), 1)
        self.critical_calibration_scores = get_critical_scores(calibration_scores= self.calibration_scores, q=q)
        
    
    
    def predict(self, states):
        
        self.agent.agent_mode('eval')
        out = self.agent.regression_prediction(states)
        out = torch.tensor(out).reshape(-1,1)
        
        
        lower = out - self.critical_calibration_scores
        upper = out + self.critical_calibration_scores
        
        
        return np.hstack((lower.reshape(-1, 1), upper.reshape(-1, 1)))
    
    def evaluate_coverage(self, test_states, test_actions):
    
        pred_intervals = self.predict(test_states)
        independent_coverages, joint_coverages = coverage(pred_intervals, test_actions)

        return independent_coverages, joint_coverages, pred_intervals

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

    #####################  hyper parameters  ####################
    action_low = 0.0
    action_high = 1.0   
    MAX_EPISODES = 500
    #############################################################
    
    state_dim = states_train.shape[1]
    action_dim = 1

    with open(os.getcwd() + f'/{dataset_name}/PPO_best_params.pkl', 'rb') as file:
      best = pickle.load(file)
      
    agent = PPOagent(state_dim, action_dim, best)

    for s in range(1, 6):
        model = RLCT(alpha = 0.1, agent = agent)
        model.fit(s, states_train, train_y, states_val, val_y, MAX_EPISODES, action_low, action_high)
        independent_coverages, joint_coverages, intervals = model.evaluate_coverage(states_test, test_y)
        
            
        mean_independent_coverage = np.mean(np.array(independent_coverages, dtype=np.float32), axis=0)
        interval_width = np.mean(intervals[:, 1] - intervals[:, 0], axis=0)
        
            
        # print(mean_independent_coverage, interval_width)
        
        covp.append(mean_independent_coverage)
        iw.append(interval_width)
    
    with open(f'{dataset_name}/coverage_prob_mean.pkl', 'wb') as file: pickle.dump(np.mean(covp), file)
    with open(f'{dataset_name}/coverage_prob_std.pkl', 'wb') as file: pickle.dump(np.std(covp), file)
    with open(f'{dataset_name}/interval_width_mean.pkl', 'wb') as file: pickle.dump(np.mean(iw), file)
    with open(f'{dataset_name}/interval_width_std.pkl', 'wb') as file: pickle.dump(np.std(iw), file)
    
    
    with open(f'{dataset_name}/coverage_prob_mean.pkl', 'rb') as file:
        covp_mean = pickle.load(file)
    with open(f'{dataset_name}/coverage_prob_std.pkl', 'rb') as file:
        covp_std = pickle.load(file)

    with open(f'{dataset_name}/interval_width_mean.pkl', 'rb') as file:
        iw_mean = pickle.load(file)
    with open(f'{dataset_name}/interval_width_std.pkl', 'rb') as file:
        iw_std = pickle.load(file)
        
    print(covp_mean, covp_std)
    print(iw_mean, iw_std)
    
