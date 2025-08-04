import numpy as np
from sampling import rs
from rbfn import RBFN
from typing import Union,Callable



def current_to_best_1(pop,lb,ub,f,max_iter,F=0.5,cr=0.5,real_func=None):
    n, d = pop.shape

    if isinstance(lb,np.ndarray):
        lb=lb.reshape(1,-1)
    if isinstance(ub,np.ndarray):
        ub=ub.reshape(1,-1)
    
    y_pop=f(pop)
    iter_best=np.empty([max_iter,d])

    for i in range(max_iter):
        r1 = np.argmin(y_pop)
        idxes = select_m_different_indexes_randomly_np(n, 2, n)
        r2, r3 = idxes[:, 0], idxes[:, 1]
        v = pop + F * (pop[r1] - pop) + F * (pop[r2] - pop[r3])# variant vector

        j_r = np.random.randint(0, d, size=[n, d])
        r = np.random.rand(n, d)
        # children
        u = np.where((r < cr) | (j_r == np.arange(d)), v, pop)
        # using the random strategy to fix out of range
        u = np.where((lb <= u) & (u <= ub),u,rs(n, d, lb, ub))
        # using the bound strategy to fix out of range
        # u = np.clip(u, lower_bound, upper_bound)
        # update the population
        y_u = f(u).reshape(-1)
        res = (y_u < y_pop).reshape(-1)
        pop = np.where(res.reshape(-1, 1), u, pop)
        y_pop = np.where(res, y_u, y_pop)
        iter_best[i]=pop[np.argmin(y_pop)]


    pop=pop[np.argsort(y_pop),:]
    if real_func is not None:
        y_iter=real_func(iter_best)
        return pop,iter_best,y_iter
    else:
        return pop,iter_best



def select_m_different_indexes_randomly_np(
    n: int,
    m: int,
    times: int
) -> np.ndarray:
    '''
    Select m different indexes from [0, n), and it will be repeated `times` times.

    Args:
        n: length of indexes
        m: the number of index to be picked at one time, 1 <= m <= n
        times: the number of pick
    '''
    idxes = np.arange(n)
    ret = []
    for _ in range(times):
        np.random.shuffle(idxes)
        ret.append(idxes[0: m].copy())
    return np.array(ret)