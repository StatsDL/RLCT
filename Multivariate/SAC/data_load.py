import numpy as np
from sklearn.preprocessing import MinMaxScaler


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