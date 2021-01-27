import sys
sys.path.append('/z/abwilf/Standard-Grid')
import standard_grid
import pickle
import os 
import sys

if __name__=="__main__":
    hash_out = sys.argv[1]
    grid=pickle.load(open(f'.{hash_out}.pkl',"rb"))
    grid.get_status()
