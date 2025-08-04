import numpy as np


def initial_parameters(**kw):
    config = {
        # for SDDObench
        'T': 60, # max_env(default: 60)
        'P': 20,
        'initial_peaks': 8,
        # for problem definition
        'd': None,
        'N':None,

        # for algorithm
        'n': None,
        'cs': 50, # chunk size
        'es': 200, # environment size
        'cur_window_size':50,
        'beta': 10,

        'test_zone':100,
        
        'reuse_env':30,
        'warn_zone':25,
        
        # for RBFN
        'hidden_shape': None,

        # for optimal process
        'max_iter': 30,
        'mp': None, # mutation probability
        'eta_m': 15,
        'eta_c': 20,
        'cp': 0.9, # crossover probability
        'pop_size': 100,
        'F':0.5,
        'cr':0.9,

        # for archive_solutions
        'top_n_rate': 0.2,
        'top_n': None,
        'reuse_n_rate': 0.4,
        'reuse_n':None,

        'max_arc':30,
    }
    config.update(kw)
    
    config['d'] = 10 if config['d'] is None else config['d']
    config['mp'] = 1.0 / config['d']
    config['n'] = 6 * config['d']
    # config['l1_warn_zone']=3*config['beta']
    # config['l2_warn_zone']=2*config['beta']
    # config['l3_warn_zone']=config['beta']    


    if config['hidden_shape'] is None:
        config['hidden_shape'] = int(np.sqrt(config['n'])) 


    if config['top_n'] is None:
        config['top_n'] = int(config['pop_size'] * config['top_n_rate'])

    if config['reuse_n'] is None:
        config['reuse_n'] = int(config['pop_size'] * config['reuse_n_rate'])

    if config['N'] is None:
        config['N'] = int(config['es'] / config['cs'])

    return config
