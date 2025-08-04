import numpy as np
from copy import deepcopy
from rbfn import *
from utils import remove_duplicate_data


class Window:
    def __init__(self,data:np.ndarray=None,limit_size:bool=False,max_size:int=None):
        if data is not None:
            self.data=data
        else:
            self.data = None
        if limit_size is True:
            if max_size is None:
                raise ValueError("Window: max_size should be set when limit_size is True")
            else:
                self.max_size=max_size
        else:
            self.max_size=None

    def append(self, sample:np.ndarray):
        if self.data is None:
            self.data=np.empty_like(sample)
            self.data=np.atleast_2d(self.data)
        else:
            self.data=np.vstack((self.data,sample))
        if self.max_size is not None:
            if len(self)>self.max_size:
                new_data=np.delete(self.data,0,axis=0)
                self.data=new_data

    def delete(self,idx:int=None):
        if idx is None:
            new_data=np.delete(self.data,0,axis=0)
        else:
            # such as idx=-1
            new_data=np.delete(self.data,idx,axis=0)
        self.data=deepcopy(new_data)

    def clear(self):
        self.data=None
    
    def __len__(self):
        length=self.data.shape[0]
        return length

    def __getitem__(self, key):
        if key == 'data':
            return self.data
        else:
            raise KeyError(f"Invalid key: {key}")
    
    def __setitem__(self,key,value):
        
        if key=="data":
            if isinstance(value,np.ndarray):
                self.data=deepcopy(value)
            else:
                raise ValueError(f"Window: data type should be numpy.ndarray, but get {type(value)} ")
        else:
            raise KeyError(f"Invalid key of {key}")



class CurtimeWindow(Window):
    def __init__(self,data,metrics):
        super().__init__(data)
        self.metrics = metrics
        self.mean_list=[]
        self.std_list=[]
        self.mean,self.std,self.median=get_statics(self.metrics)



    def __getitem__(self, key):
        if key == 'data':
            return self.data
        elif key == 'metrics':
            return self.metrics
        elif key=='mean_list':
            return self.mean_list
        elif key=='std_list':
            return self.std_list
        elif key=='mean':
            return self.mean
        elif key=='std':
            return self.std
        else:
            raise KeyError(f"Invalid key: {key}")
    
    def __setitem__(self, key, value):
        if key=="data":
            if isinstance(value,np.ndarray):
                self.data=deepcopy(value)
            else:
                raise ValueError(f"Window: data type should be numpy.ndarray, but get {type(value)} ")
        elif key=="metrics":
            if isinstance(value,np.ndarray):
                self.data=deepcopy(value)
            else:
                raise ValueError(f"Window: metrics type should be numpy.ndarray, but get {type(value)} ")
        else:
            raise KeyError(f"Invalid key: {key}")


    def update(self,metrics):
        self.metrics=metrics
        self.mean, self.std, self.median = get_statics(self.metrics)
        self.mean_list.append(self.mean)
        self.std_list.append(self.std)

class EnvWindow(Window):
    def __init__(self,data=None,metrics=None,update_interval=None):
        super().__init__(data)
        self.update_flag=False
        if update_interval is None:
            self.update_interval=1
        else:
            self.update_interval=update_interval

        if metrics is not None:
            self.metrics=metrics
            self.mean, self.std, self.median = get_statics(self.metrics)
        else:
            self.metrics=None
            self.mean, self.std, self.median =None,None,None

        if data is not None:
            self.update_count=len(self)//self.update_interval
        else:
            self.update_count=0

    def append(self,sample:np.ndarray):
        super().append(sample)
        self.update_flag=self._is_update()

    def _is_update(self):
        if len(self)//self.update_interval>self.update_count:
            update=True
            self.update_count = len(self) // self.update_interval
        else:
            update=False
        return update


    def __getitem__(self, item):
        if item == 'data':
            return self.data
        elif item == 'metrics':
            return self.metrics
        elif item=='mean':
            return self.mean
        elif item=='std':
            return self.std
        else:
            raise KeyError(f"Invalid key: {item}")

    def update(self, metrics):
        self.metrics = metrics
        self.mean, self.std, self.median = get_statics(self.metrics)

class Env:
    def __init__(self,data=None):
        
        self.EW=EnvWindow(data)
        self.env_id=None
        self.models=[] # all the models in the current environment
        self.weights=[] # ensemble weight
        self.solutions=None # archive of the best solutions
        
    def __setitem__(self,key,value):
        if key=="models":
            self.models=value
        elif key=="weights":
            self.weights=value
        elif key=="solutions":
            self.solutions=value
        elif key == "EW":
            self.EW=value
        elif key == "env_id":
            self.env_id=value
        else:
            raise KeyError(f"Invalid key of {key}")
    
    def __getitem__(self,key):
        if key=="models":
            return self.models
        elif key=="weights":
            return self.weights
        elif key=="solutions":
            return self.solutions
        elif key == "EW":
            return self.EW
        elif key == "env_id":
            return self.env_id
        else:
            raise KeyError(f"Invalid key of {key}")

    def _new_cur_models(self,config):
        uni_data=remove_duplicate_data(self.EW['data'])
        self.EW['data']=uni_data
        new_model=construct_rbfn(samples=self.EW['data'],hidden_shape=config['hidden_shape'])
        # new_model=construct_gpr(self.EW['data'])
        return new_model

    def save_models(self,config=None):
        self.models.clear()
        self.weights.clear()

        new_model=self._new_cur_models(config=config)
        self.models.append(new_model)
        self.weights.append(1)

    def update_models(self,config=None):
        new_model=self._new_cur_models(config=config)
        self.models[0]=new_model

    def update_sols(self,solutions):
        self.solutions=deepcopy(solutions)
        
        
class Archive:
    def __init__(self):
        self.envs=[]
        self.env_num=0

    def delete(self,env_id=None):
        if env_id is None:
            self.envs.pop(0)
        else:
            self.envs.pop(env_id)


    def __len__(self):
        return self.env_num
    
    def __getitem__(self,env_id):
        return self.envs[env_id]
    
    def append(self,Env):
        if Env['env_id'] is None:
            Env['env_id']= self.env_num
            self.envs.append(Env)
            self.env_num+=1
        else:   
            for j in range(self.env_num):
                if self.envs[j].env_id==Env.env_id:
                    self.envs[j]=Env
                    break
    def update(self,max_arc):
        if len(self.envs) > max_arc:
            self.delete()
            self.env_num-=1
            for j in range(self.env_num):
                self.envs[j].env_id-=1



def get_statics(metrics):
    return np.mean(metrics),np.std(metrics),np.median(metrics)