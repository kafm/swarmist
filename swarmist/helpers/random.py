from typing import Union, Dict, Optional, List, TypeVar, Tuple, Callable
from dataclasses import dataclass
import numpy as np
import scipy.stats as ss
from core import Bounds 

EPSILON = 1e-6

def uniform(low: float = 0.0, high: float = 1.0, size:int = None)->Union[float, List[float]]:
    return np.random.uniform(low=low, high=high, size=size)

#scale is the expected mean
def exponential(size:int = None, scale: float = 1.0)->Union[float, List[float]]:
    return np.random.exponential(scale=scale, size=size)

#scale is the expected mode
def rayleigh(size: int = None, scale: float = 1.0)->Union[float, List[float]]:
    return np.random.rayleigh(scale=scale,size=size)

def beta(size:int = None, alpha: float = 2.0, beta: float = 2.0)->Union[float, List[float]]:
    return  np.random.beta(alpha, beta, size=size)

def normal(mean:float = 0.0, sigma: float = 1.0, size: int = None)->Union[float, List[float]]:
    return np.random.normal(mean=mean, sigma=sigma, size=size)

def lognormal(mean:float = 0.0, sigma: float = 1.0, size: int = None)->Union[float, List[float]]:
    return np.random.lognormal(mean=mean, sigma=sigma, size=size)

@dataclass(frozen=True)
class BondedRandom(Bounds):
    def uniform(self, size: int = None)->Union[float, List[float]]:
        return uniform(low=self.min, high=self.max, size=size)
    
    def beta(self, size:int = None, alpha: float = 2.0, beta: float = 2.0)->Union[float, List[float]]:
        return beta(alpha, beta, size=size) * (self.max - self.min) + self.min
    
    #Using inverse transform sampling. 
    #Taken from  https://stackoverflow.com/questions/25141250/how-to-truncate-a-numpy-scipy-exponential-distribution-in-an-efficient-way
    def exponential(self, size:int = None, scale: float = None)->Union[float, List[float]]:
        if not scale:
            np.clip(exponential(size=size, scale=.5),0,1) * (self.max - self.min) + self.min
        return np.clip(exponential(size=size, scale=scale),self.min,self.max)
        # rnd_cdf = np.random.uniform(
        #     low=ss.expon.cdf(x=self.min, scale=scale),
        #     high=ss.expon.cdf(x=self.max, scale=scale),
        #     size=size
        # )
        # return ss.expon.ppf(q=rnd_cdf, scale=scale)
    
    def rayleigh(self, size: int = None, scale: float = 1.0)->Union[float, List[float]]:
        return np.clip(rayleigh(size=size, scale=scale),0,1) * (self.max - self.min) + self.min