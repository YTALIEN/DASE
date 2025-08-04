import os
import pandas as pd
from time import time
from benchmark.SDDObench.SDDObench import *
from benchmark.SDDObench.bench_config import Config
from sampling import lhs
from configuration import initial_parameters
import DASE
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


config={} # define your own benchmark configuration
user_config= Config(**config)
metric_flag=True # print the metric results
MAX_RUN =20
INST=[1,2,3,4,5,6,7,8]
DRFS=[1,2,3,4,5]
d=10
config=initial_parameters(d=d)

for ins in INST:
    for drf in DRFS:
        for run in range(MAX_RUN):
            T=config['T']
            initial_peaks=config['initial_peaks']
            P=config['P']
            max_iter=config['max_iter']
            env_size=config['es']
            N=config['N'] 
            chunk_size=config['cs']

            lb, ub = get_bound(ins)
            X = lhs(env_size, d, lb, ub)
            prob_params = {
                'peak_info': None, 
                'delta_info': None, 
                'num_peaks': initial_peaks, 
                'T': T, 
                'P': P,

            }
            prob_params.update(num_instance=ins, df_type=drf, dim=d)
            env_archive, cur_window = None, None
            detector=None

            delt_info = []
            env_pbar=tqdm(
                range(T),
                total=T,
                desc=f'run:{run} F{ins}D{drf}',
                bar_format='{l_bar}{bar:20}{r_bar}'
            )
            
            for env in env_pbar:
                # initial environment
                prob_params.update(x=X,change_count=env)
                y,prob_params= sddobench(prob_params,user_config)
                delt_info.append(prob_params['delta_info'][1]) # The value of drift

                # np.random.seed(42)
                samples = np.hstack((X, y[:, np.newaxis]))
                perm_idx=np.random.permutation(env_size)
                perm_splits=np.array_split(perm_idx, N)
                # np.random.seed()

                if env==0:
                    env_archive = DASE.initial_archives(samples, lb, ub, config)
                for t in range(N):
                    samples_t=samples[perm_splits[t]]
                    time1=time()
                    iter_best,env_archive,cur_window,detector,drift_eval= DASE.run(
                                                                samples=samples_t,
                                                                lb=lb,ub=ub,
                                                                env_archive=env_archive,
                                                                config=config,
                                                                detector=detector,
                                                                cur_window=cur_window,)


                