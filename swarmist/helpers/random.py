from typing import Union, Dict, Optional, List, TypeVar, Tuple, Callable
from dataclasses import dataclass
import numpy as np
import scipy.stats as ss
from core import Bounds

def uniform(low: float = 0.0, high: float = 1.0, size:int = None)->Union[float, List[float]]:
    return np.random.uniform(low=low, high=high, size=size)

def beta(size:int = None, alpha: float = 2.0, beta: float = 2.0)->Union[float, List[float]]:
    return  np.random.beta(alpha, beta, size=size)

#scale is the expected mean
def exponential(size:int = None, scale: float = 1.0)->Union[float, List[float]]:
    return np.random.exponential(scale=scale, size=size)

#scale is the expected mode
def rayleigh(size: int = None, scale: float = 1.0)->Union[float, List[float]]:
    return np.random.rayleigh(scale=scale,size=size)

def normal(loc:float = 0.0, scale: float = 1.0, size: int = None)->Union[float, List[float]]:
    return np.random.normal(mean=loc, sigma=scale, size=size)

def lognormal(loc:float = 0.0, scale: float = 1.0, size: int = None)->Union[float, List[float]]:
    return np.random.lognormal(mean=loc, sigma=scale, size=size)

#k=shape
def weibull(shape: float = 1.0, scale: float = 1.0, size: int = None)->Union[float, List[float]]:
    return np.random.weibull(shape, scale=scale, size=size)

#loc=location
def cauchy(loc: float = 0.0, scale: float = 1.0, size: int = None)->Union[float, List[float]]:
    u = np.random.uniform(size=size)
    return loc + ( scale * np.tan(np.pi * (u-.5) ) )

def levy(loc: float = 0.0, scale: float = 1.0, size: int = None)->Union[float, List[float]]:
    u = np.random.uniform(size=size)
    z = ss.norm.ppf(1 - u/2)
    return loc + ( scale / ( 1/z ) **2 )

def skewnormal(shape: float = 0.0, loc: float = 0.0, scale: float = 1.0, size: int = None)->Union[float, List[float]]:
    return ss.skewnorm.rvs(a=shape, loc=loc, scale=scale, size=size)

@dataclass(frozen=True)
class BondedRandom(Bounds):
    def uniform(self, size: int = None)->Union[float, List[float]]:
        return uniform(low=self.min, high=self.max, size=size)
    
    def beta(self, size: int = None, alpha: float = 2.0, beta: float = 2.0)->Union[float, List[float]]:
        return beta(alpha, beta, size=size) * (self.max - self.min) + self.min
    
    def exponential(self, size: int = None, scale: float = 1.0)->Union[float, List[float]]:
        exp_val = np.clip(np.divide(-np.log(np.random.uniform(size=size)),scale),0,1)
        return self.min + exp_val * (self.max - self.min)
    
    def rayleigh(self, size: int = None, scale: float = 1.0)->Union[float, List[float]]:
        ray_val = np.clip(np.multiply(scale, np.sqrt(-2*np.log(np.random.uniform(size=size)))),0,1)
        return self.min + ray_val * (self.max - self.min)
    
    def weibull(self, size: int = None, k: float = 1.0, scale: float = 1.0)->Union[float, List[float]]:
        wei_val = np.clip(np.random.weibull(k, scale=scale, size=size),0,1)
        return self.min + wei_val * (self.max - self.min)
    
    def cauchy(self, size: int = None, loc: float = 0.0, scale: float = 1.0)->Union[float, List[float]]:
        c_val = np.clip(cauchy(loc=loc, scale=scale, size=size), 0, 1)
        return self.min + c_val * (self.max - self.min)
    
    def levy(self, size: int = None, loc: float = 0.0, scale: float = 1.0)->Union[float, List[float]]:
        l_val = np.clip(levy(loc=loc, scale=scale, size=size), 0, 1)
        return self.min + l_val * (self.max - self.min)
    
    def normal(self, loc:float = 0.0, scale: float = 1.0, size: int = None)->Union[float, List[float]]:
        return np.clip(normal(mean=loc, sigma=scale, size=size), self.min, self.max)

    def lognormal(self, loc:float = 0.0, scale: float = 1.0, size: int = None)->Union[float, List[float]]:
        return np.clip(lognormal(mean=loc, sigma=scale, size=size), self.min, self.max)
    
    def skewnormal(self, size: int, shape: float = 0.0, loc: float = 0.0, scale: float = 1.0)->Union[float, List[float]]:
        return np.clip(skewnormal(shape=shape, mean=loc, sigma=scale, size=size), self.min, self.max)
    
    
