import os
import math
import torch
import pickle
import numpy as np
import pandas as pd


from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


import warnings
warnings.filterwarnings("ignore")


#####################################################################################################
#####################################################################################################

def one_dimen_transform(Y_train, Y_predict, d, h):
    
    train_scaler, predict_scaler = MinMaxScaler(), MinMaxScaler()
    Y_train = train_scaler.fit_transform(Y_train)  
    Y_predict= predict_scaler.fit_transform(Y_predict)
    
    
    n = len(Y_train)
    n1 = len(Y_predict)
    
    X_train, Y_tr = np.zeros((n-d, d)), np.zeros((n-d, h))  # from d+1,...,n
    X_predict, Y_pr = np.zeros((n1, d)), np.zeros((n1, h))  # from n-d,...,n+n1-d
    
    for i in range(n-d-h+1):
        
        X_train[i, :] = Y_train[i:i+d].squeeze()
        Y_tr[i,:] = Y_train[i+d:i+d+h].reshape(-1)  # Reshape to match (2,)

        
    for i in range(n1):
        if i < d:
            X_predict[i, :] = np.r_[Y_train[n-d+i:].squeeze(), Y_predict[:i].squeeze()]
            Y_pr[i, :] = Y_predict[i:i+h].reshape(-1)
        else:
            X_predict[i, :] = Y_predict[i-d:i].squeeze()
            Y_pr[i, :] = Y_predict[i:i+h].reshape(-1)
            
    Y_train = Y_train[d:]
    
    return([X_train.astype(np.float32), X_predict.astype(np.float32), Y_tr.astype(np.float32), Y_pr.astype(np.float32), train_scaler, predict_scaler])


##############################################################################################################################################################


def coverage(intervals, target):
    """
    Determines whether intervals cover the target prediction
    considering each target horizon either separately or jointly.

    Args:
        intervals: shape [batch_size, 2, horizon, n_outputs]
        target: ground truth forecast values

    Returns:
        individual and joint coverage rates
    """

    lower, upper = intervals[:, 0], intervals[:, 1]
    # [batch, horizon, n_outputs]
    horizon_coverages = torch.logical_and(target >= lower, target <= upper)
    # [batch, horizon, n_outputs], [batch, n_outputs]
    return horizon_coverages, torch.all(horizon_coverages, dim=1)


def get_critical_scores(calibration_scores, q):
    """
    Computes critical calibration scores from scores in the calibration set.

    Args:
        calibration_scores: calibration scores for each example in the
            calibration set.
        q: target quantile for which to return the calibration score

    Returns:
        critical calibration scores for each target horizon
    """

    return torch.tensor(
            [
                torch.quantile(position_calibration_scores, q=q)
                for position_calibration_scores in calibration_scores
            ]
    ).T

##############################################################################################################################################################

class AuxiliaryForecaster(torch.nn.Module):
    """
    The auxiliary RNN issuing point predictions.

    Point predictions are used as baseline to which the (normalised)
    uncertainty intervals are added in the main CFRNN network.
    """

    def __init__(self, embedding_size, input_size=1, output_size=1, horizon = 1, path=None):
        """
        Initialises the auxiliary forecaster.

        Args:
            embedding_size: hyperparameter indicating the size of the latent
                RNN embeddings.
            input_size: dimensionality of the input time-series
            output_size: dimensionality of the forecast
            horizon: forecasting horizon
            rnn_mode: type of the underlying RNN network
            path: optional path where to save the auxiliary model to be used
                in the main CFRNN network
        """
        super(AuxiliaryForecaster, self).__init__()
        # input_size indicates the number of features in the time series
        # input_size=1 for univariate series.
        
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.path = path
        self.horizon = horizon
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_size, embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_size, embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_size, output_size*horizon)
        )

    def forward(self, x):
        
        out = self.mlp(x)
        out = out.view(-1, self.output_size*self.horizon)
    
        return out

    def fit(self, train_dataset, batch_size, epochs, lr):
        """
        Trains the auxiliary forecaster to the training dataset.

        Args:
            train_dataset: a dataset of type `torch.utils.data.Dataset`
            batch_size: batch size
            epochs: number of training epochs
            lr: learning rate
        """
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        self.train()
        for epoch in range(epochs):
            train_loss = 0.0

            for i, (inputs, targets) in enumerate(train_loader):
                optimizer.zero_grad()

                out = self(inputs)
                
                loss = criterion(out.float(), targets.float())
                loss.backward()

                train_loss += loss.item()

                optimizer.step()

            # mean_train_loss = train_loss / len(train_loader)
            # if epoch % 50 == 0:
            #     print("Epoch: {}\tTrain loss: {}".format(epoch+1, mean_train_loss))

        if self.path is not None:
            torch.save(self, self.path)

##############################################################################################################################################################

class CFRNN:
    """
    The basic CFRNN model as presented in Algorithm 1 of the accompanying paper.

    CFRNN training procedure entails training the underlying (auxiliary) model
    on the training dataset (which is implemented as part of
    `AuxiliaryForecaster`), and calibrating the predictions of the auxiliary
    forecaster against the calibration dataset.

    The calibration is done by computing the empirical distribution of
    nonconformity scores (implemented via the `nonconformity`
    function), via the `calibrate` method.

    The AuxiliaryForecaster can be fit to the dataset from scratch, or,
    if the model path is provided, the model is loaded directly, its training is
    skipped and only the calibration procedure is carried out.

    Additional methods are provided for returning predictions: on the test
    example, the point prediction is done by the underlying
    `AuxiliaryForecaster`, and the horizon-specific critical calibration scores
    (obtained from the calibration procedure) are added to the point forecast to
    obtain the resulting interval. The coverage can be further evaluated by
    comparing the uncertainty intervals to the ground truth forecasts,
    returning joint and independent coverages and getting the errors from the
    point prediction.
    """

    def __init__(
        self,
        embedding_size,
        input_size=1,
        output_size=1,
        horizon = 1,
        error_rate=0.05,
        auxiliary_forecaster_path=None,
        **kwargs
    ):
        """
        Args:
            embedding_size: size of the embedding of the underlying point
                forecaster
            input_size: dimensionality of observed time-series
            output_size: dimensionality of a forecast step
            horizon: forecasting horizon (number of steps into the future)
            error_rate: controls the error rate for the joint coverage in the
                estimated uncertainty intervals
            rnn_mode: type of the underlying AuxiliaryForecaster model
            auxiliary_forecaster_path: training of the underlying
                `AuxiliaryForecaster` can be skipped if the path for the
                already trained `AuxiliaryForecaster` is provided.
        """
        super(CFRNN, self).__init__()
        # input_size indicates the number of features in the time series
        # input_size=1 for univariate series.
        self.input_size = input_size
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.horizon = horizon
        self.requires_auxiliary_fit = True

        self.auxiliary_forecaster_path = auxiliary_forecaster_path
        if self.auxiliary_forecaster_path and os.path.isfile(self.auxiliary_forecaster_path):
            self.auxiliary_forecaster = torch.load(auxiliary_forecaster_path)
            for param in self.auxiliary_forecaster.parameters():
                param.requires_grad = False
            self.requires_auxiliary_fit = False
        else:
            self.auxiliary_forecaster = AuxiliaryForecaster(
                embedding_size, input_size, output_size,  horizon, auxiliary_forecaster_path
            )

        self.alpha = error_rate
        self.calibration_scores = None
        self.critical_calibration_scores = None
        self.corrected_critical_calibration_scores = None

    def nonconformity(self, output, calibration_example):
        """
        Measures the nonconformity between output and target time series.

        Args:
            output: the point prediction given by the auxiliary forecasting
                model
            calibration_example: the tuple consisting of calibration
                example's input sequence, ground truth forecast, and sequence
                length

        Returns:
            Average MAE loss for every step in the sequence.
        """
        # Average MAE loss for every step in the sequence.
        target = calibration_example[1]
        return torch.nn.functional.l1_loss(output, target, reduction="none")

    def calibrate(self, calibration_dataset: torch.utils.data.Dataset):
        """
        Computes the nonconformity scores for the calibration dataset.
        """
        calibration_loader = torch.utils.data.DataLoader(calibration_dataset, batch_size=1)
        n_calibration = len(calibration_dataset)
        calibration_scores = []

        with torch.set_grad_enabled(False):

            self.auxiliary_forecaster.eval()
            for calibration_example in calibration_loader:
                
                inputs, targets = calibration_example
                out = self.auxiliary_forecaster(inputs)
                score = self.nonconformity(out, calibration_example)
                # n_batches: [batch_size, horizon, output_size]
                calibration_scores.append(score)

        # [output_size, horizon, n_samples]
        self.calibration_scores = torch.vstack(calibration_scores).T

        # [horizon, output_size]
        q = min((n_calibration + 1.0) * (1 - self.alpha) / n_calibration, 1)
        corrected_q = min((n_calibration + 1.0) * (1 - self.alpha) / n_calibration, 1)

        self.critical_calibration_scores = get_critical_scores(calibration_scores=self.calibration_scores, q=q)

        # Bonferroni corrected calibration scores.
        # [horizon, output_size]
        self.corrected_critical_calibration_scores = get_critical_scores(
            calibration_scores=self.calibration_scores, q=corrected_q
        )

    def fit(
        self,
        train_dataset: torch.utils.data.Dataset,
        calibration_dataset: torch.utils.data.Dataset,
        epochs,
        lr,
        batch_size=32,
        **kwargs
    ):
        """
        Fits the CFRNN model.

        If the auxiliary forecaster is not trained, fits the underlying
        `AuxiliaryForecaster` on the training dataset using the batch size,
        learning rate and number of epochs provided. Otherwise, the auxiliary
        forecaster that has been loaded on initialisation is used. On fitting
        the underlying model, computes calibration scores for the calibration
        dataset.

        Args:
            train_dataset: training dataset on which the underlying
            forecasting model is trained
            calibration_dataset: calibration dataset used to compute the
            empirical nonconformity score distribution
            epochs: number of epochs for training the underlying forecaster
            lr: learning rate for training the underlying forecaster
            batch_size: batch size for training the underlying forecaster
        """

        if self.requires_auxiliary_fit:
            # Train the multi-horizon forecaster.
            self.auxiliary_forecaster.fit(train_dataset, batch_size, epochs, lr)

        # Collect calibration scores
        self.calibrate(calibration_dataset)

    def predict(self, x, corrected=True):
        """
        Forecasts the time series with conformal uncertainty intervals.

        Args:
            x: time-series to be forecasted
            state: initial state for the underlying auxiliary forecaster RNN
            corrected: whether to use Bonferroni-corrected calibration scores

        Returns:
            tensor with lower and upper forecast bounds; hidden RNN state
        """

        out = self.auxiliary_forecaster(x)

        if not corrected:
            # [batch_size, horizon, n_outputs]
            lower = out - self.critical_calibration_scores
            upper = out + self.critical_calibration_scores
        else:
            # [batch_size, horizon, n_outputs]
            lower = out - self.corrected_critical_calibration_scores
            upper = out + self.corrected_critical_calibration_scores

        # [batch_size, 2, horizon, n_outputs]
        return torch.stack((lower, upper), dim=1)

    def evaluate_coverage(self, test_dataset: torch.utils.data.Dataset, corrected=True):
        """
        Evaluates coverage of the examples in the test dataset.

        Args:
            test_dataset: test dataset
            corrected: whether to use the Bonferroni-corrected critical
            calibration scores
        Returns:
            independent and joint coverages, forecast uncertainty intervals
        """
        self.auxiliary_forecaster.eval()

        independent_coverages, joint_coverages, intervals = [], [], []
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

        for inputs, targets in test_loader:
            
            batch_intervals = self.predict(inputs, corrected=corrected)
            intervals.append(batch_intervals)
            independent_coverage, joint_coverage = coverage(batch_intervals, targets)
            independent_coverages.append(independent_coverage)
            joint_coverages.append(joint_coverage)

        # [n_samples, (1 | horizon), n_outputs] containing booleans
        independent_coverages = torch.cat(independent_coverages)
        joint_coverages = torch.cat(joint_coverages)

        # [n_samples, 2, horizon, n_outputs] containing lower and upper bounds
        intervals = torch.cat(intervals)

        return independent_coverages, joint_coverages, intervals

    def get_point_predictions_and_errors(self, test_dataset: torch.utils.data.Dataset, corrected=True):
        """
        Obtains point predictions of the examples in the test dataset.

        Obtained by running the Auxiliary forecaster and adding the
        calibrated uncertainty intervals.

        Args:
            test_dataset: test dataset
            corrected: whether to use Bonferroni-corrected calibration scores

        Returns:
            point predictions and their MAE compared to ground truth
        """
        self.auxiliary_forecaster.eval()

        point_predictions = []
        errors = []
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

        for inputs, targets in test_loader:
            point_prediction = self.auxiliary_forecaster(inputs)
            point_predictions.append(point_prediction)
            errors.append(torch.nn.functional.l1_loss(point_prediction, targets, reduction="none").squeeze())

        point_predictions = torch.cat(point_predictions)
        errors = torch.cat(errors)

        return point_predictions, errors
    
    
##############################################################################################################################################################

datasets = ['EURUSD', 'AUDUSD', 'GBPUSD', 'CNYUSD', 'CADUSD']
seeds = 5

for dataset_name in datasets:
    covp, iw = [], []
    
    print(f"\nDataset: {dataset_name}")
    data_dir = f'datasets/{dataset_name}.csv'
    # Reading the data
    
    if dataset_name == 'Solar_Atl_data':
        df = pd.read_csv(data_dir, skiprows=2)
        df.drop(columns=df.columns[0:5], inplace=True)
        df.drop(columns='Unnamed: 13', inplace=True)
        df = df[['DHI']]
        

    else:
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
    look_back, h = 20, 2
    
    X_train, X_val, y_train, y_val, _, _ = one_dimen_transform(train_data, val_data, look_back, h)
    _, X_test, _, y_test,_, test_scaler = one_dimen_transform(val_data, test_data, look_back, h)

    X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train)
    X_val, y_val = torch.Tensor(X_val), torch.Tensor(y_val)
    X_test, y_test = torch.Tensor(X_test), torch.Tensor(y_test)
    ###############################################################################

    train_dataset = TensorDataset(X_train, y_train)
    calibration_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test) 
    
    for s in range(seeds):
        model = CFRNN(embedding_size = 32, input_size = 20, output_size = 1, horizon=2, error_rate = 0.1)
        model.fit(train_dataset, calibration_dataset, epochs = 100, lr = 0.001, batch_size =32)
        
        independent_coverages, joint_coverages, intervals = model.evaluate_coverage(test_dataset, corrected = True)
        # mean_independent_coverage = torch.mean(torch.mean(independent_coverages.float(), dim=0), dim= 0)
        mean_joint_coverage = torch.mean(joint_coverages.float(), dim=0).item()*100
        
        interval_width = np.mean(np.mean((test_scaler.inverse_transform(intervals[:,1].detach().numpy()) - test_scaler.inverse_transform(intervals[:,0].detach().numpy())).squeeze(), axis=0), axis =0)
        
        
        output_loc = f"{dataset_name}//"
        
        if not os.path.exists(output_loc):
            os.makedirs(output_loc)
        
        with open(f'{dataset_name}/coverage_prob_{s}.pkl', 'wb') as file: pickle.dump(mean_joint_coverage, file)
        with open(f'{dataset_name}/interval_width_{s}.pkl', 'wb') as file: pickle.dump(interval_width, file)
        
        # with open(f'{dataset_name}/coverage_prob.pkl', 'rb') as file:
        #     mean_independent_coverage = pickle.load(file)
    
        # with open(f'{dataset_name}/interval_width.pkl', 'rb') as file:
        #     interval_width = pickle.load(file)
            
        # print(mean_joint_coverage, interval_width)
        covp.append(mean_joint_coverage)
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


