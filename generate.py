# Used to generate grid search
SG_PATH = '/z/abwilf/Standard-Grid'
import sys
sys.path.append(SG_PATH)
import standard_grid
import pickle
import time
import os
from utils import *

if __name__=="__main__":
    hash_len = 5 # the number of characters in each hash.  if running lots of tests, may have collision if too few chars.  elif running few tests, can be nice to have smaller identifying strings
    # email_args= {
    #     'subject': 'Hello there',
    #     'text': 'hi',
    #     'to_addr': 'dummyblah123@gmail.com',
    #     'secrets_path': '/z/abwilf/dw/mailgun_secrets.json',
    # }
    # grid = standard_grid.Grid('./main.py','./results/', hash_len=hash_len, email_args=email_args)
    grid = standard_grid.Grid('./main.py','./results/', hash_len=hash_len)
    mkdirp('./results/')

    hp = {
        'overwrite_tensors': [0],
        'cross_utterance': [0,1],
        'modality': ['text', 'audio', 'text,audio'],
        'tensors_path': ['unique'],
    }

    for k,v in hp.items():
        grid.register(k,v)

    grid.generate_grid()
    grid.shuffle_grid()
    grid.generate_shell_instances(prefix='python ',postfix='')
    
    # Breaks the work across num_gpus GPUs, num_parallel jobs on each gpu.  device 0 is "2", device 1 is 0, device 2 is 1
    
    # devices you want to use in nvidia-smi naming
    gpus = [0,1,2]

    gpu_map = {1: 0, 2: 1, 0: 2} # translating to how cuda sees them
    gpus = lmap(lambda elt: gpu_map[elt], gpus)

    num_parallel = 1
    hash_out = grid.create_runner(num_runners=len(gpus),runners_prefix=['CUDA_VISIBLE_DEVICES=%d sh'%i for i in gpus],parallel=num_parallel)

    save_json(f'results/{hash_out}/hp.json', hp)
    print(f'''

hash='{hash_out}'

p start_time.py $hash; root=$(pwd); attempt='0'; cd $root/results/${{hash}}/central/attempt_${{attempt}}/; chmod +x main.sh; ./main.sh; cd $root; p status.py ${{hash}}; p interpret.py ${{hash}};

    ''')


