import os 
import numpy as np
import pandas as pd
from pathlib import Path
from detoxify import Detoxify
import pdb
from tqdm import tqdm

ABSPATH = Path(__file__).parent.absolute()

def get_all_predictions(model, input_list, step_size=200):
    toxicities = []
    print("Making toxicity predictions...")
    for i in tqdm(range(0,len(input_list),step_size)):
        toxicities += model.predict(input_list[i:min(i+step_size,len(input_list))])['toxicity']
    return toxicities

if __name__ == "__main__":
    utils_dir = str(ABSPATH) + '/toxic_text_utils/'
    save_dir = str(ABSPATH.parent) + '/data/toxic-text/'
    os.makedirs(save_dir, exist_ok=True)
    # Read data
    test_features = pd.read_csv(utils_dir + 'test.csv')
    test_labels = pd.read_csv(utils_dir + 'test_labels.csv')
    # Initialize model
    model = Detoxify('multilingual', device='cuda')
    preds = np.array(get_all_predictions(model, list(test_features['content'])))
    np.savez(save_dir + '/toxic-text-detoxify.npz', preds=preds, labels=test_labels['toxic'])
