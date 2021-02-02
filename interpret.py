# Interprets results of grid search
import sys
sys.path.append('/z/abwilf/Standard-Grid')
import standard_grid
import pickle
import os 
import sys
from utils import *
import pandas as pd

def get_df(path):
    if 'results' not in path:
        path = f'./results/{path}/csv_results.csv'
    df = pd.read_csv(path)
    df = df.rename(columns={k: k.replace('STDGRID_', '') for k in df.columns})
    
    for col in lfilter(lambda elt: 'accs' in elt or 'losses' in elt, df.columns):
        try:
            df[col] = df[col].map(lambda elt: ar(json.loads(elt.replace('nan', 'NaN'))))
        except:
            print(f'Destringify failed at {col}')
     
    return df

if __name__=="__main__":
    hash_out = sys.argv[1]
    grid=pickle.load(open(f'.{hash_out}.pkl',"rb"))
    csv_path=f"results/{hash_out}/csv_results.csv"
    grid.json_interpret("output/results.txt",csv_path)
    # df = pd.read_csv(csv_path)
    # df = df.rename(columns={k: k.replace('STDGRID_', '') for k in df.columns})
    # df['acc_delta'] = df['acc_delta'].map(lambda elt: npr(elt*100))
    # proc_csv_path = f"results/{hash_out}/csv_processed.csv"
    # df.sort_values(by=['acc_delta'], ascending=False).to_csv(proc_csv_path, index=False)


