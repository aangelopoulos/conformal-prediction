import os, sys, inspect, itertools
sys.path.insert(1, os.path.join(sys.path[0], '../../'))
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest 
from tqdm import tqdm
from pathlib import Path
ABSPATH = Path(__file__).parent.absolute()

def get_data():
    os.makedirs(str(ABSPATH.parent) + '/data/meps/raw/', exist_ok=True)
    os.system('wget https://raw.githubusercontent.com/aangelopoulos/ltt/main/experiments/meps/data/meps_19_reg.csv -O ' + str(ABSPATH.parent) + '/data/meps/raw/meps_19_reg.csv')
    df = pd.read_csv(str(ABSPATH.parent) + '/data/meps/raw/meps_19_reg.csv')
    response_name = "UTILIZATION_reg"
    col_names = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', 'PERWT15F', 'REGION=1',
               'REGION=2', 'REGION=3', 'REGION=4', 'SEX=1', 'SEX=2', 'MARRY=1',
               'MARRY=2', 'MARRY=3', 'MARRY=4', 'MARRY=5', 'MARRY=6', 'MARRY=7',
               'MARRY=8', 'MARRY=9', 'MARRY=10', 'FTSTU=-1', 'FTSTU=1', 'FTSTU=2',
               'FTSTU=3', 'ACTDTY=1', 'ACTDTY=2', 'ACTDTY=3', 'ACTDTY=4',
               'HONRDC=1', 'HONRDC=2', 'HONRDC=3', 'HONRDC=4', 'RTHLTH=-1',
               'RTHLTH=1', 'RTHLTH=2', 'RTHLTH=3', 'RTHLTH=4', 'RTHLTH=5',
               'MNHLTH=-1', 'MNHLTH=1', 'MNHLTH=2', 'MNHLTH=3', 'MNHLTH=4',
               'MNHLTH=5', 'HIBPDX=-1', 'HIBPDX=1', 'HIBPDX=2', 'CHDDX=-1',
               'CHDDX=1', 'CHDDX=2', 'ANGIDX=-1', 'ANGIDX=1', 'ANGIDX=2',
               'MIDX=-1', 'MIDX=1', 'MIDX=2', 'OHRTDX=-1', 'OHRTDX=1', 'OHRTDX=2',
               'STRKDX=-1', 'STRKDX=1', 'STRKDX=2', 'EMPHDX=-1', 'EMPHDX=1',
               'EMPHDX=2', 'CHBRON=-1', 'CHBRON=1', 'CHBRON=2', 'CHOLDX=-1',
               'CHOLDX=1', 'CHOLDX=2', 'CANCERDX=-1', 'CANCERDX=1', 'CANCERDX=2',
               'DIABDX=-1', 'DIABDX=1', 'DIABDX=2', 'JTPAIN=-1', 'JTPAIN=1',
               'JTPAIN=2', 'ARTHDX=-1', 'ARTHDX=1', 'ARTHDX=2', 'ARTHTYPE=-1',
               'ARTHTYPE=1', 'ARTHTYPE=2', 'ARTHTYPE=3', 'ASTHDX=1', 'ASTHDX=2',
               'ADHDADDX=-1', 'ADHDADDX=1', 'ADHDADDX=2', 'PREGNT=-1', 'PREGNT=1',
               'PREGNT=2', 'WLKLIM=-1', 'WLKLIM=1', 'WLKLIM=2', 'ACTLIM=-1',
               'ACTLIM=1', 'ACTLIM=2', 'SOCLIM=-1', 'SOCLIM=1', 'SOCLIM=2',
               'COGLIM=-1', 'COGLIM=1', 'COGLIM=2', 'DFHEAR42=-1', 'DFHEAR42=1',
               'DFHEAR42=2', 'DFSEE42=-1', 'DFSEE42=1', 'DFSEE42=2',
               'ADSMOK42=-1', 'ADSMOK42=1', 'ADSMOK42=2', 'PHQ242=-1', 'PHQ242=0',
               'PHQ242=1', 'PHQ242=2', 'PHQ242=3', 'PHQ242=4', 'PHQ242=5',
               'PHQ242=6', 'EMPST=-1', 'EMPST=1', 'EMPST=2', 'EMPST=3', 'EMPST=4',
               'POVCAT=1', 'POVCAT=2', 'POVCAT=3', 'POVCAT=4', 'POVCAT=5',
               'INSCOV=1', 'INSCOV=2', 'INSCOV=3', 'RACE']

    y = df[response_name].values.astype(np.float32)
    X = df[col_names].values.astype(np.float32)

    return X, y

def shuffle_split(X,y):
    n_full = X.shape[0]
    perm = np.random.permutation(n_full)
    X = X[perm]
    y = y[perm]
    n = n_full//2
    return X[:n], X[n:], y[:n], y[n:]

def process_data(X_train, X_val, y_train, y_val):
	# zero mean and unit variance scaling 
	scalerX = StandardScaler()
	scalerX = scalerX.fit(X_train)
	X_train = scalerX.transform(X_train)
	X_val = scalerX.transform(X_val)

	# scale the response as it is highly skewed
	y_train = np.log(1.0 + y_train)
	y_val = np.log(1.0 + y_val)
	# reshape the response
	y_train = np.squeeze(np.asarray(y_train))
	y_val = np.squeeze(np.asarray(y_val))
	return X_train, X_val, y_train, y_val

def optimize_params_GBR(X_train, X_val, y_train, y_val, alpha=0.1):
    filename = './.cache/GBR_optim.pkl'
    try:
        optim_df = pd.read_pickle(filename)
    except:
        lrs = [0.01,]
        n_ests = [100,]
        subsamples = [1,]
        max_depths = [25,] 
        optim_df = pd.DataFrame(columns = ['lr','n_estimators', 'subsample', 'max_depth', 'cvg'])
        for lr, n_estimators, subsample, max_depth in tqdm(itertools.product(lrs, n_ests, subsamples, max_depths)):
            mean = GradientBoostingRegressor(random_state=0, learning_rate=lr, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
            upper = GradientBoostingRegressor(random_state=0, learning_rate=lr, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth, alpha=1-alpha/2, loss='quantile')
            lower = GradientBoostingRegressor(random_state=0, learning_rate=lr, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth, alpha=alpha/2, loss='quantile')
            mean.fit(X_train, y_train)
            upper.fit(X_train, y_train)
            lower.fit(X_train, y_train)
            pred_mean = mean.predict(X_train) 
            pred_upper = upper.predict(X_train)
            pred_lower = lower.predict(X_train)
            pred_upper = np.maximum(pred_upper, pred_lower + 1e-6)
            mse = ( (y_train - pred_mean)**2 ).mean()
            cvg = ( (y_train <= pred_upper) & (y_train >= pred_lower) ).mean()
            optim_dict = { 'lr' : lr,'n_estimators' : n_estimators, 'subsample' : subsample, 'max_depth' : max_depth, 'mse' : mse, 'cvg' : cvg }
            optim_df = optim_df.append(optim_dict, ignore_index=True)
            tqdm.write(str(optim_dict))
        os.makedirs('./.cache/', exist_ok=True)
        optim_df.to_pickle(filename)
    idx_quantiles = np.argmin(np.abs(optim_df['cvg']-(1-alpha)))
    idx_mean = np.argmin(optim_df['mse'])
    optim_df_quantiles = optim_df.loc[idx_quantiles]
    optim_df_mean = optim_df.loc[idx_mean]
    # GBR
    mean = GradientBoostingRegressor(random_state=0, learning_rate=optim_df_mean['lr'], n_estimators=int(optim_df_mean['n_estimators']), subsample=optim_df_mean['subsample'], max_depth=int(optim_df_mean['max_depth']), alpha=1-alpha/2, loss='quantile')
    upper = GradientBoostingRegressor(random_state=0, learning_rate=optim_df_quantiles['lr'], n_estimators=int(optim_df_quantiles['n_estimators']), subsample=optim_df_quantiles['subsample'], max_depth=int(optim_df_quantiles['max_depth']), alpha=1-alpha/2, loss='quantile')
    lower = GradientBoostingRegressor(random_state=0, learning_rate=optim_df_quantiles['lr'], n_estimators=int(optim_df_quantiles['n_estimators']), subsample=optim_df_quantiles['subsample'], max_depth=int(optim_df_quantiles['max_depth']), alpha=alpha/2, loss='quantile')
    mean.fit(X_train, y_train)
    upper.fit(X_train, y_train)
    lower.fit(X_train, y_train)
    pred_mean_train = mean.predict(X_train)
    pred_mean_val = mean.predict(X_val)
    pred_upper_val = upper.predict(X_val)
    pred_lower_val = lower.predict(X_val)
    return pred_mean_val, pred_upper_val, pred_lower_val

def save_model_outputs():
    X, y = get_data()
    X_train, X_val, y_train, y_val = process_data(*shuffle_split(X,y))
    pred_val, upper_val, lower_val = optimize_params_GBR(X_train, X_val, y_train, y_val)
    os.makedirs(str(ABSPATH.parent) + '/data/meps/', exist_ok=True)
    np.savez(str(ABSPATH.parent) + '/data/meps/meps-gbr.npz', pred=pred_val, upper=upper_val, lower=lower_val, X=X_val, y=y_val)

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    save_model_outputs()
