import numpy as np
from sklearn.preprocessing import MinMaxScaler

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
