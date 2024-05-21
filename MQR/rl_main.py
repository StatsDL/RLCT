import os
import math
import torch
import pickle
import numpy as np
import pandas as pd


from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler


import warnings
warnings.filterwarnings("ignore")


#####################################################################################################
#####################################################################################################

def one_dimen_transform(Y_train, Y_predict, d):
    
    train_scaler, predict_scaler = MinMaxScaler(), MinMaxScaler()
    Y_train = train_scaler.fit_transform(Y_train)  
    Y_predict= predict_scaler.fit_transform(Y_predict)
    
    
    n = len(Y_train)
    n1 = len(Y_predict)
    X_train = np.zeros((n-d, d))  # from d+1,...,n
    X_predict = np.zeros((n1, d))  # from n-d,...,n+n1-d
    for i in range(n-d):
        X_train[i, :] = Y_train[i:i+d].squeeze()
    for i in range(n1):
        if i < d:
            X_predict[i, :] = np.r_[Y_train[n-d+i:].squeeze(), Y_predict[:i].squeeze()]
        else:
            X_predict[i, :] = Y_predict[i-d:i].squeeze()
            
    Y_train = Y_train[d:]

    
    return([X_train.astype(np.float32), X_predict.astype(np.float32), Y_train.astype(np.float32), Y_predict.astype(np.float32), train_scaler, predict_scaler])

##############################################################################################################################################################

def coverage(lower, upper, target):
    """
    Determines whether intervals cover the target prediction
    considering each target horizon either separately or jointly.

    Args:
        lower: lower bound
        upper: upper bound
        target: ground truth forecast values

    Returns:
        individual and joint coverage rates
    """

    # [batch, horizon, n_outputs]
    horizon_coverages = torch.logical_and(target >= lower, target <= upper)
    # [batch, horizon, n_outputs], [batch, n_outputs]
    return horizon_coverages, torch.all(horizon_coverages, dim=1)


##############################################################################################################################################################

def quantile_loss(output, target,  q):
    single_loss = (output - target) * (output >= target) * q + (target - output) * (output < target) * (1 - q)
    loss = torch.mean(torch.mean(single_loss, dim=1))

    return loss

###############################################################################




class QRNN(torch.nn.Module):
    def __init__(
        self,
        epochs = 10,
        batch_size = 32,
        input_size=1,
        learning_rate = 0.001,
        hidden_size=20,
        output_size=2,
        num_layers=1,
        alpha=0.05
    ):

        super(QRNN, self).__init__()

        self.EPOCH = epochs
        self.BATCH_SIZE = batch_size
        self.INPUT_SIZE = input_size
        self.LR = learning_rate
        self.HIDDEN_UNITS = hidden_size
        self.OUTPUT_SIZE = output_size
        self.NUM_LAYERS = num_layers
        self.q = alpha

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 2 * output_size)
        )

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # reshape x to (batch * time_step, input_size)

        # pass through MLP
        out = self.mlp(x)
       
        # reshape to match the desired output shape [1, 32, 2]
        out = out.view(1, -1, 2*self.OUTPUT_SIZE)

        return out



    def fit(self, X, Y):

        Y = torch.stack([Y, Y], dim=1)

        X = Variable(torch.tensor(X), volatile=True).type(torch.FloatTensor)
        Y = Variable(torch.tensor(Y), volatile=True).type(torch.FloatTensor)

        self.X = X
        self.Y = Y

        optimizer = torch.optim.Adam(self.parameters(), lr=self.LR)  # optimize all rnn parameters
        self.loss_func = quantile_loss

        # training and testing
        for epoch in range(self.EPOCH):

            batch_indexes = np.random.choice(list(range(X.shape[0])), size=self.BATCH_SIZE, replace=True, p=None)

            x = torch.tensor(X[batch_indexes, :]).detach()
            y = torch.tensor(Y[batch_indexes]).detach()

            output = self(x).reshape(-1, self.OUTPUT_SIZE, 2)  # rnn output

            # quantile loss
            loss = self.loss_func(output[:, :, 0], y[:, :, 0], self.q) + self.loss_func(
                output[:, :, 1], y[:, :, 0],  1 - self.q)

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            # if (epoch+1) % 10 == 0:
            #     print(f'Epoch [{epoch+1}/{self.EPOCH}], Loss: {loss.item():.4f}')

    def predict(self, X):

        X_test = Variable(torch.tensor(X), volatile=True).type(torch.FloatTensor)
        predicts_ = self(X_test).view(-1, self.OUTPUT_SIZE, 2)
        prediction_0 = predicts_[:, :, 0]
        prediction_1 = predicts_[:, :, 1]

        return prediction_0, prediction_1
    
##############################################################################################################################################################

datasets = ['EURUSD', 'AUDUSD' ,'GBPUSD', 'CNYUSD', 'CADUSD']
seeds = 5

for dataset_name in datasets:
    # covp, iw = [], []
    
    # print(f"\nDataset: {dataset_name}")
    # data_dir = f'datasets/{dataset_name}.csv'
    # # Reading the data
    
    
    # df = pd.read_csv(data_dir, index_col='Date')
    # df.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1, inplace=True)
    
    # # print(df)
    # # length of time series data
    # L = int(df.shape[0])
    
    # # length of train and validation data
    # len_train = int(math.ceil(L * .8))
    # len_test = int(math.ceil(L * .1))
    # len_val = L - len_train - len_test
    
    # # splitting the original time series into train, validation, and test data
    # df_train = df[0: len_train]
    # df_val = df[len_train: len_train + len_val]
    # df_test = df[len_train + len_val: L]
    
    # # train, validation, and test data values using their DataFrames
    # train_data = df_train.values
    # val_data = df_val.values
    # test_data = df_test.values
    
    
    # # ###############################################################################
    # # look_back is the time-horizon taken to predict one-day ahead prediction
    # look_back = 20
    
    # X_train, X_val, y_train, y_val, _, _ = one_dimen_transform(train_data, val_data, look_back)
    # _, X_test, _, y_test,_, test_scaler = one_dimen_transform(val_data, test_data, look_back)

    # X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train)
    # X_test, y_test = torch.Tensor(X_test), torch.Tensor(y_test)
    
    # for s in range(seeds):
    #     model = QRNN(epochs=100, batch_size=32, input_size = 20, learning_rate = 0.001, output_size = 1, hidden_size = 64, num_layers = 1, alpha = 0.1)
    #     model.fit(X_train, y_train)
        
    #     y_u, y_l = model.predict(X_test)
    #     independent_coverage, joint_coverage = coverage(y_l ,y_u, torch.Tensor(y_test))
        
    #     mean_independent_coverage = torch.mean(independent_coverage.float(), dim=0)*100
    #     interval_width = np.mean((test_scaler.inverse_transform(y_u.detach().numpy()) - test_scaler.inverse_transform(y_l.detach().numpy())).squeeze(), axis=0)
        
        
    #     covp.append(mean_independent_coverage)
    #     iw.append(interval_width)
        
    #     output_loc = f"{dataset_name}//"
        
    #     if not os.path.exists(output_loc):
    #         os.makedirs(output_loc)
        
    #     with open(f'{dataset_name}/coverage_prob_{s}.pkl', 'wb') as file: pickle.dump(mean_independent_coverage, file)
    #     with open(f'{dataset_name}/interval_width_{s}.pkl', 'wb') as file: pickle.dump(interval_width, file)
        
    #     # with open(f'{dataset_name}/coverage_prob.pkl', 'rb') as file:
    #     #     mean_independent_coverage = pickle.load(file)
    
    #     # with open(f'{dataset_name}/interval_width.pkl', 'rb') as file:
    #     #     interval_width = pickle.load(file)
            
    #     # print(mean_independent_coverage, interval_width)
    #     covp.append(mean_independent_coverage)
    #     iw.append(interval_width)
        
        
        
    # with open(f'{dataset_name}/coverage_prob_mean.pkl', 'wb') as file: pickle.dump(np.mean(covp), file)
    # with open(f'{dataset_name}/coverage_prob_std.pkl', 'wb') as file: pickle.dump(np.std(covp), file)
    # with open(f'{dataset_name}/interval_width_mean.pkl', 'wb') as file: pickle.dump(np.mean(iw), file)
    # with open(f'{dataset_name}/interval_width_std.pkl', 'wb') as file: pickle.dump(np.std(iw), file)
    
    
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