import os
import math
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
from scipy import stats as st

from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


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

def model_loss(output, target):
    loss = torch.nn.functional.mse_loss(output, target)
    return loss

###############################################################################

class DPRNN(nn.Module):
    def __init__(
        self,
        epochs=100,
        batch_size=32,

        input_size=1,
        lr=0.01,
        output_size=1,
        embedding_size=64,
        n_layers=1,

        dropout_prob=0.5,
        **kwargs
    ):

        super(DPRNN, self).__init__()

        self.EPOCH = epochs
        self.BATCH_SIZE = batch_size



        self.dropout_prob = dropout_prob
        self.dropout = nn.Dropout(p=dropout_prob)
        self.LR = lr
        
        
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_size, embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_size, embedding_size),
            torch.nn.ReLU(),
        )

        
        self.out = torch.nn.Linear(embedding_size, output_size)
        
    def forward(self, x):

        out = self.mlp(x)
        out = self.out(self.dropout(out))
        out = out.view(-1, self.output_size)
        
    
        return out


    def fit(self, X, Y):
        
        X = Variable(torch.tensor(X), volatile=True).type(torch.FloatTensor)
        Y = Variable(torch.tensor(Y), volatile=True).type(torch.FloatTensor)

        self.X = X
        self.Y = Y

        optimizer = torch.optim.Adam(self.parameters(), lr=self.LR)  # optimize all rnn parameters
        self.loss_func = model_loss  # nn.MSELoss()

        # training and testing
        for epoch in range(self.EPOCH):

            batch_indexes = np.random.choice(list(range(X.shape[0])), size=self.BATCH_SIZE, replace=True, p=None)

            x = torch.tensor(X[batch_indexes, :]).detach()
            y = torch.tensor(Y[batch_indexes]).detach()

            output = self(x).reshape(-1, self.output_size)  # rnn output


            loss = self.loss_func(output, y)  # MSE loss

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            # if (epoch+1) % 10 == 0:
            #     print(f'Epoch [{epoch+1}/{self.EPOCH}], Loss: {loss.item():.4f}')

    def predict(self, X, num_samples=100, alpha=0.05):
        z_critical = st.norm.ppf((1 - alpha) + (alpha) / 2)


        predictions = []
        X_test = Variable(torch.tensor(X), volatile=True).type(torch.FloatTensor)

        for idx in range(num_samples):
            predicts_ = self(X_test).view(-1, self.output_size)
            predictions.append(predicts_.detach().numpy().reshape((-1, 1, self.output_size)))

        pred_mean = np.mean(np.concatenate(predictions, axis=1), axis=1)
        pred_std = z_critical * np.std(np.concatenate(predictions, axis=1), axis=1)

        return pred_mean, pred_std
    
    
##############################################################################################################################################################

datasets = ['EURUSD', 'AUDUSD' ,'GBPUSD', 'CNYUSD', 'CADUSD']
seeds = 5


for dataset_name in datasets:
    covp, iw =[],[]
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
    
    X_train, X_val, y_train, y_val, _, _ = one_dimen_transform(train_data, val_data, look_back)
    _, X_test, _, y_test,_, test_scaler = one_dimen_transform(val_data, test_data, look_back)

    X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train)
    X_val, y_val = torch.Tensor(X_val), torch.Tensor(y_val)
    X_test, y_test = torch.Tensor(X_test), torch.Tensor(y_test)
    ###############################################################################

    train_dataset = TensorDataset(X_train, y_train)
    calibration_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    for s in range(seeds):
    
        model = DPRNN(epochs=100, batch_size=32, input_size = 20, learning_rate = 0.001, output_size = 1, hidden_size = 64, dropout_prob=0.5)
        model.fit(X_train, y_train)    
        
        y_pred, y_std  = model.predict(X_test, num_samples=100, alpha=0.1)
        y_u = torch.Tensor([y_pred[k] + y_std[k] for k in range(len(y_pred))])
        y_l = torch.Tensor([y_pred[k] - y_std[k] for k in range(len(y_pred))])
        
    
        independent_coverage, joint_coverage = coverage(y_l ,y_u, torch.Tensor(y_test))
        
        mean_independent_coverage = torch.mean(independent_coverage.float(), dim=0)*100
        interval_width = np.mean((test_scaler.inverse_transform(y_u.detach().numpy()) - test_scaler.inverse_transform(y_l.detach().numpy())).squeeze(), axis=0)
        
        
        covp.append(mean_independent_coverage)
        iw.append(interval_width)
        
        output_loc = f"{dataset_name}//"
        
        if not os.path.exists(output_loc):
            os.makedirs(output_loc)
        
        with open(f'{dataset_name}/coverage_prob_{s}.pkl', 'wb') as file: pickle.dump(mean_independent_coverage, file)
        with open(f'{dataset_name}/interval_width_{s}.pkl', 'wb') as file: pickle.dump(interval_width, file)
        
        # with open(f'{dataset_name}/coverage_prob.pkl', 'rb') as file:
        #     mean_independent_coverage = pickle.load(file)
    
        # with open(f'{dataset_name}/interval_width.pkl', 'rb') as file:
        #     interval_width = pickle.load(file)
            
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