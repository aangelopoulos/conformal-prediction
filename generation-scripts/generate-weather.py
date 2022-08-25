import os 
import numpy as np
import pandas as pd
from pathlib import Path
import catboost
import pdb

ABSPATH = Path(__file__).parent.absolute()

def get_predictions(features_df, model):
    '''
    Calculates predictions on df features for specified model
    
    Return: array [num_samples x 2],
        where
            num_samples = number of rows in features_df
            2 = [mean, variance]
    
    '''
    return model.predict(features_df)


def get_all_predictions(features_df, models_list):
    '''
    Return: array [ensemble_size x num_samples x 2],
        where
            ensemble_size = number of models in models_list
            num_samples = number of rows in features_df
            2 = [mean, variance]
    '''
    all_preds = []
    for model in models_list:
        preds = np.asarray(get_predictions(features_df, model))
        all_preds.append(preds)
    return np.stack(all_preds, axis=0)

def load_models(utils_dir):
    if not os.path.exists(utils_dir + '/baseline-models/seed10.cbm'):
        os.makedirs(utils_dir, exist_ok=True)
        os.system('wget https://storage.yandexcloud.net/yandex-research/shifts/weather/baseline-models.tar -O ' + utils_dir + '/raw_models.tar')
        os.system('tar -xf ' + utils_dir + '/raw_models.tar -C ' + utils_dir)
    baseline_models = []

    # 10 models provided
    ensemble_size=10

    for ind in range(1, ensemble_size+1):
        model = catboost.CatBoostRegressor()
        model.load_model(f'{utils_dir}/baseline-models/seed{ind}.cbm')
        baseline_models.append(model)

    return baseline_models

if __name__ == "__main__":
    data_dir = '/home/aa/data/weather/'
    utils_dir = str(ABSPATH) + '/weather_utils/'
    save_dir = str(ABSPATH.parent) + '/data/weather/'
    if not os.path.exists(data_dir + '/canonical-partitioned-dataset/shifts_canonical_dev_in.csv'):
        os.makedirs(data_dir, exist_ok=True)
        os.system('wget https://storage.yandexcloud.net/yandex-research/shifts/weather/canonical-partitioned-dataset.tar -O ' + data_dir + 'raw.tar')
        os.system('tar -xf ' + data_dir + '/raw.tar -C ' + data_dir)
    df_dev_in = pd.read_csv(data_dir + '/canonical-partitioned-dataset/shifts_canonical_dev_in.csv')
    df_dev_out = pd.read_csv(data_dir + '/canonical-partitioned-dataset/shifts_canonical_dev_out.csv')
    df_dev = pd.concat([df_dev_in, df_dev_out])
    baseline_models = load_models(utils_dir)
    X_dev = df_dev.iloc[:,6:]
    all_preds = get_all_predictions(X_dev, baseline_models)
    np.savez(save_dir + '/weather-catboost.npz', preds=all_preds, temperatures=df_dev['fact_temperature'], times=df_dev['fact_time'])
