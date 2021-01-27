import sys
sys.path.append('/z/abwilf/Standard-Grid')
import standard_grid
import pickle
import os 
import sys
import datetime

if __name__=="__main__":
    hash_out = sys.argv[1]
    print(f'Resetting start time for {hash_out}')
    grid_path = f'.{hash_out}.pkl'
    grid=pickle.load(open(grid_path, "rb"))
    grid.rt.start_time=datetime.datetime.now()
    grid.save(grid_path)
