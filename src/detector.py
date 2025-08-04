import numpy as np
from archive import Window


class DriftDetect:
    def __init__(self,beta):
        self.drift_statu=False
        self.drift_thres=None

        self.l1_warn_thres=None
        self.l1_warn_count = 0
        self.l2_warn_thres=None
        self.l2_warn_count=0
        self.l3_warn_thres=None
        self.l3_warn_count=0


        self.warn_l1_zone=3*beta
        self.warn_l2_zone=2*beta
        self.warn_l3_zone=beta

        self.warn_window=Window(limit_size=True,max_size=6*beta)


    def run(self,CW,EW):
        self.l2_warn_thres=EW['mean']+2*EW['std']
        self.l1_warn_thres=EW['mean']+EW['std']
        self.l3_warn_thres=EW['mean']+3*EW['std']
        # self.drift_thres=EW['mean']+3*EW['std']

        
        if CW['mean']+CW['std']>self.l1_warn_thres:
            self.l1_warn_count+=1
            self.warn_window.append(CW['data'][-1])
        if CW['mean'] + CW['std'] > self.l2_warn_thres:
            self.l2_warn_count += 1
            self.warn_window.append(CW['data'][-1])
        if CW['mean'] + CW['std'] > self.l3_warn_thres:
            self.l3_warn_count += 1
            self.warn_window.append(CW['data'][-1])

        if self.l1_warn_count>=self.warn_l1_zone:
            self.drift_statu=True
        elif self.l2_warn_count>=self.warn_l2_zone:
            self.drift_statu=True
        elif self.l3_warn_count>=self.warn_l3_zone:
            self.drift_statu=True

        return self.drift_statu


    def reset(self):

        self.drift_statu=False
        self.l1_warn_count=0
        self.l2_warn_count=0
        self.l3_warn_count=0
        self.warn_window.clear()


