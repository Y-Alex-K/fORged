import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

def preprocess(train_path, test_path):
    '''
    Input
    path to test dataset.
    train dataset is supposed to be under the same directory
    Output
    each data frame is converted a data frame which has "demand" and "without_seasonal" columns.
    (seasonal term is estimated based on the training set)
    '''
    #read csv files
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    #drop temporal data
    train = train.drop(['Unnamed: 0', 'Unnamed: 1'], axis=1)
    test = test.drop(['Unnamed: 0', 'Unnamed: 1'], axis=1)
    #rename column
    train.columns = ['demand']
    test.columns = ['demand']
    #estimate seasonal term
    result = seasonal_decompose(train['demand'], model='additive', freq=12)
    seasonal = np.array(result.seasonal).flatten()
    #add without_seasonal column
    train['without_seasonal'] = train['demand'] - seasonal[:len(train)]
    test['without_seasonal'] = test['demand'] - seasonal[:len(test)]
    return train, test, seasonal

def D_pred(train, test, seasonal, verbose=False):
    '''
    Input
    train: data that ARIMA fits on (pd.DataFrame)
    test: observed data of the prediction period (pd.DataFrame)
    Output
    rolling prediction of length len(test)
    '''
    train = np.array(train['without_seasonal'])
    test = np.array(test['without_seasonal'])
    history = [x for x in train]
    predictions = []
    for t in range(len(test)):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0][0]
        obs = test[t]
        history.append(obs)
        predictions.append(yhat)
        if verbose:
            print(t)
            print('predicted=%f, expected=%f' % (yhat, obs))
    prediction = predictions + seasonal[:len(test)]
    return predictions

def F_epsilon(obs, pred):
    '''
    Input
    obs: observed values (np.array or pd.Series)
    pred: predicted values
    Output
    distribution function of residuals
    '''
    obs = np.array(obs)
    pred = np.array(pred)
    epsilon = obs - pred
    epsilon.sort()
    return lambda x: (stats.percentileofscore(epsilon, x, kind='weak') / 100 + 0.5 - stats.percentileofscore(epsilon, 0, kind='weak') / 100) #centerinze

def find_Q(I, Dhat, F):
    '''
    one-time order quantity
    I: inventory level
    Dhat: prediction of
    F: distribution function
    '''
    Q = -100.0
    alpha = 0.01 #step size
    while True:
        a = F(I + Q - Dhat)
        a += F(I + Q - Dhat - 90)
        a -= 3 * (1 - F(I + Q - Dhat))
        if a > 0.0:
            return max(Q - alpha, 0)
        Q += alpha

def order_quantity(train, test, seasonal, F, I_0=73):
    '''
    compute the order quantity over the period of test dataset
    '''
    Q = []
    pred = D_pred(train, test, seasonal)
    I = I_0
    for i in range(len(test)):
        Dhat = pred[i]
        D = test['demand'].iloc[i]
        q = find_Q(I, Dhat, F)
        Q.append(q)
        I = max(I + q - D, 0)
    return np.array(Q)

def final(test, Q, I_0=73):
    '''
    beginning inventory, order quantity, ending inventory, holding cost, and
backorder cost. As a summary, the program should also output total cost, as well as, total and average holding costs,
and total and average backorder costs.
    '''
    cost = 0.0
    start_inventory = [I_0]
    end_inventory = []
    holding_cost = 0.0
    backorder_cost = 0.0
    I = I_0
    for i, d in enumerate(test['demand']):
        print(f'-----Summary of month {i+1}-----')
        print(f'beginning inventory level is {round(I, 2)}')
        D = test['demand'].iloc[i]
        q = Q[i]
        print(f'order quantity is {round(q, 2)}')
        h = max(I + q - D, 0) #holding cost for inventory <=90
        h += max(I + q - D - 90, 0) # holding cost for inventory > 90
        print(f'holding cost is {round(h, 2)}')
        b = 3 * max(D - I - q, 0) #backorder cost
        print(f'backorder cost is {round(b, 2)}')
        I = max(I + q - D, 0) #inventory level
        print(f'ending inventory level is {round(I, 2)}')
        start_inventory.append(I)
        end_inventory.append(I)
        holding_cost += h
        backorder_cost += b
        cost += h + b
    print('*****Overall summary*****')
    print(f'total cost is {round(cost, 2)}')
    print(f'total holding cost is {round(holding_cost, 2)}')
    print(f'average holding cost is {round(holding_cost/len(test), 2)}')
    print(f'total backorder cost is {round(backorder_cost, 2)}')
    print(f'average backorder cost is {round(backorder_cost/len(test), 2)}')

#preprocess the data
train, test, seasonal = preprocess(sys.argv[1], sys.argv[2])
#use last 2 years of train dataset to estimate the distribution of residuals
pred = D_pred(train[:len(train)-24], train[len(train)-24:], seasonal)
obs = train['demand'].iloc[len(train)-24:]
F = F_epsilon(obs, pred)
#compute order quantity
Q = order_quantity(train, test, seasonal, F)
#final output
final(test, Q)
