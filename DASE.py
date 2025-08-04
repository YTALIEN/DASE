import DE
from sampling import lhs,rs
from archive import *
from detector import DriftDetect
from functools import partial
from typing import Union
from utils import remove_duplicate_data


def run(samples,
        lb,ub,
        env_archive,
        config:dict,
        detector=None,
        cur_window=None,
        true_drift=None,):

    window_size=config['cur_window_size']
    test_zone=config['test_zone']
    warn_zone=config['warn_zone']
    n,d_=samples.shape
    d=d_-1


    curenv=env_archive[-1]
    EW=curenv['EW']
    if cur_window is None:
        cur_samples=curenv['EW']['data'][-window_size:]
        metrics=calculate_metrics(cur_samples,curenv)
        CW=CurtimeWindow(data=cur_samples,metrics=metrics)
        
    else:
        CW=cur_window

    if detector is None:
        detector = DriftDetect(beta=config['beta'])


    if true_drift is None:
        new_env_flag=False
        for ti in range(n):
            samples_ti=samples[ti]
            new_env_flag,CW=is_new_env(CW,EW,curenv,detector,samples_ti)
            if new_env_flag:
                tr=ti # the delay for the chunk data
                break
    else:
        # true drift information is known
        new_env_flag=true_drift

    
    if new_env_flag:
        test_data=lhs(test_zone,d,lb,ub)
        if warn_zone<n:
            newenv=Env(data=samples)
        else:
            new_env_sample=remove_duplicate_data(np.vstack((detector.warn_window['data'],samples)))
            newenv = Env(data=new_env_sample)
            
        curenv.save_models(config=config)
        env_archive.append(curenv)

        curenv=make_new_env(newenv,env_archive,test_data,config)

        detector.reset()

    else:
        for ti in range(n):
            curenv['EW'].append(samples[ti])
            # print(curenv['EW'].update_flag)
            if curenv['EW'].update_flag:
                curenv.update_models(config=config)

    opt_solutions,iter_best=optimal_process(curenv,lb,ub,config)
    opt_res=opt_solutions[0]

    if (opt_res!=iter_best[-1]).any():
        iter_best[-1]=opt_res

    curenv.update_sols(opt_solutions)

    env_archive.append(curenv)

    # update the metrics for EW
    metrics_ew=calculate_metrics(curenv['EW']['data'],curenv)
    curenv['EW'].update(metrics_ew)
    env_archive.update(max_arc=config['max_arc'])

    return iter_best,env_archive,CW,detector,new_env_flag



def optimal_process(curenv,lb,ub,config):
    arc_solutions=curenv['solutions']
    pop_size=config['pop_size']
    max_iter=config['max_iter']
    d=config['d']
    F=config['F']
    cr=config['cr']
    pop=lhs(pop_size,d,lb,ub)
    if arc_solutions is not None:
        if len(arc_solutions)>0:
            pop=np.vstack((pop,arc_solutions))

    pop,iter_best=DE.current_to_best_1(pop,lb,ub,f=partial(ensemble_models,models=curenv['models'],weights=curenv['weights']),max_iter=max_iter,F=F,cr=cr)

    best_inds=pop[:config['top_n']]
    return best_inds,iter_best

def calculate_metrics(samples,curenv):
    models=curenv['models']
    weights=curenv['weights']
    samples=np.atleast_2d(samples)
    x,y=samples[:,:-1],samples[:,-1]
    y_pred=ensemble_models(x,models,weights)
    metrics=np.abs((y-y_pred)/y)
    return metrics


def is_new_env(CW,EW,curenv,detector,samples_i):
    '''
    detect for drift by CW
    '''
    CW.append(samples_i)
    CW.delete()
    metrics=calculate_metrics(CW['data'],curenv)
    CW.update(metrics)
    drift=detector.run(CW,EW)

    return drift,CW


def make_new_env(newenv,env_archive,test_data,config):
    samples=newenv['EW']['data']

    newenv["models"].append(construct_rbfn(samples=samples,hidden_shape=config['hidden_shape']))
    newenv['weights'].append(1)

    models_c=newenv['models'][0]
    
    newenv['weights'],newenv['models'],newenv['solutions']=weights_reuse_envs(env_archive,models_c,test_data,samples,config)

    return newenv


def weights_reuse_envs(env_archive,models_c,test_data,samples,config):
    
    '''
        weighted reuse the old envs in the archive
    '''
    max_reuse_env=config['reuse_env']
    pop_size=config['pop_size']
    top_n=config['top_n']

    models=[]
    weights=[]
    solutions=[]

    test_k=ensemble_models(test_data,models_c)
    samples_k_x=samples[:,:-1]
    samples_k_y=samples[:,-1]

    models_old=[]
    len_env=len(env_archive)

    mapping_distance=np.empty(len_env) 
    approx_error=np.empty(len_env)


    for i in range(len_env):
        env_i=env_archive[i]
        model_i=env_i['models']

        test_mdi=ensemble_models(test_data,model_i)
        mapping_distance[i]=np.mean((test_k-test_mdi)**2)

        test_aei=ensemble_models(samples_k_x,model_i)
        approx_error[i]=np.mean((test_aei-samples_k_y)**2)
    
    metrics=approx_error+mapping_distance
    env_sort=np.argsort(metrics)

    w_c=np.clip((1-len_env/max_reuse_env),0.5,1)
    w_r=1-w_c
    len_reuse=min(len_env,max_reuse_env)
    models_mtrs=mapping_distance[env_sort[:len_reuse]]
    models_mtrs_fixed=np.where(models_mtrs==0,1e-6,models_mtrs)
    recip_mtrs=np.reciprocal(models_mtrs_fixed)
    w_old=w_r*recip_mtrs/np.sum(recip_mtrs)
    k=0
    for i in env_sort[:len_reuse]:
        env_i=env_archive[i]
        models_old.append(env_i['models'][0])
        sol_count_i=min(int(pop_size*w_old[k]),top_n)
        solutions.extend(env_i['solutions'][:sol_count_i])
        k+=1

    models.append(models_c)
    weights.append(w_c)
    models.extend(models_old)
    weights.extend(w_old)

    return weights,models,solutions


def ensemble_models(x:np.ndarray,models,weights:list=None):
    y_pred=0
    if not isinstance(models,list):
        models_list= [models]
    else:
        models_list = models

    len_m=len(models_list)
    if weights is None:
        if len(models_list)!=1:
            raise Exception("Number of models should be only 1 without weights!")
        else:
            weights_list=[1]*len_m
    else:
        weights_list=weights

    for mi,wi in zip(models_list,weights_list):
        yi=mi.predict(x)
        yi=yi*wi
        y_pred+=yi
    
    return y_pred

def initial_archives(samples,
                    lb,ub,
                    config,):
    env_archive=Archive()
    newenv = Env(data=samples)

    model_0=construct_rbfn(samples,hidden_shape=config['hidden_shape'])


    newenv['models'].append(model_0)
    newenv['weights'].append(1)


    solutions,_=optimal_process(newenv,lb,ub,config)
    newenv.update_sols(solutions)

    # update the metrics for EW
    metrics_ew = calculate_metrics(newenv['EW']['data'], newenv)
    newenv['EW'].update(metrics_ew)


    env_archive.append(newenv)

    return env_archive
