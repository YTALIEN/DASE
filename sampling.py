import numpy as np
from typing import Union


def lhs(n: int,
        d: int,
        lb: Union[int, float, np.ndarray],
        ub: Union[int, float, np.ndarray]
        ) -> np.ndarray:
    """Latin hypercube sampling
       
    Args:
        n: The number of the sample data
        d: The number of the decision variables
        lb: A number or a vector, the lower bound of the decision variables
        ub: A number or a vector, the ub of the decision variables
    """
    if np.any(lb > ub):
        return None
   
    lb, ub = to2dColVec(lb), to2dColVec(ub)

    intervalSize = 1.0 / n
    
    samplePoints = np.empty([d, n])
    for i in range(n):
       
        samplePoints[:, i] = np.random.uniform(low=i * intervalSize, high=(i + 1) * intervalSize, size=d)
    # offset
    samplePoints = lb + samplePoints * (ub - lb)
    for i in range(d):
        np.random.shuffle(samplePoints[i])
    return samplePoints.T


def rs(n, d, lb, ub):
    """random sampling

    Args:
        n: The number of the sample data
        d: The number of the decision variables
        lb: A number or a vector, the lower bound of the decision variables
        ub: A number or a vector, the ub of the decision variables
    """
    if np.any(lb > ub):
        return None
    lb, ub = to2dColVec(lb), to2dColVec(ub)
    samplePoints = np.random.random([d, n])
    samplePoints = lb + samplePoints * (ub - lb)
    return samplePoints.T

def to2dColVec(x):
    """convert to column vector
    convert a number or 1-d vector to column vector
    """
    # if x is a number
    if np.size(x) == 1:
        return x
    # convert to 2-d column vector of type numpy.array
    return np.reshape(x, (np.size(x), -1))
