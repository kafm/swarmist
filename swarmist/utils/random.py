from typing import Union, Dict, Optional, List, TypeVar, Tuple, Callable
from dataclasses import dataclass
import numpy as np
import scipy.stats as ss
from swarmist.core import Bounds

OneOrMoreFloat = float | List[float]

def rand(size: int = 1)->OneOrMoreFloat:
    return np.random.rand(size)

def uniform(low:float = 0.0, high:float = 1.0, size:int = None)->OneOrMoreFloat:
    return np.random.uniform(low=low, high=high, size=size)

def beta(alpha:float = 2.0, beta:float = 2.0, size:int = None)->OneOrMoreFloat:
    return  np.random.beta(alpha, beta, size=size)

#scale is the expected mean
def exponential(scale:float = 1.0, size:int = None)->OneOrMoreFloat:
    return np.random.exponential(scale=scale, size=size)

#scale is the expected mode
def rayleigh(scale:float = 1.0, size:int = None)->OneOrMoreFloat:
    return np.random.rayleigh(scale=scale,size=size)

def normal(loc:OneOrMoreFloat = 0.0, scale:OneOrMoreFloat = 1.0, size:int = None)->OneOrMoreFloat:
    return np.random.normal(loc=loc, scale=scale, size=size)

def lognormal(loc:OneOrMoreFloat = 0.0, scale:OneOrMoreFloat = 1.0, size:int = None)->OneOrMoreFloat:
    return np.random.lognormal(loc=loc, scale=scale, size=size)

#k=shape
def weibull(shape:float = 1.0, scale:float = 1.0, size: int = None)->OneOrMoreFloat:
    return np.random.weibull(shape, scale=scale, size=size)

#loc=location
def cauchy(loc:float = 0.0, scale:float = 1.0, size:int = None)->OneOrMoreFloat:
    u = np.random.uniform(size=size)
    return loc + ( scale * np.tan(np.pi * (u-.5) ) )

def levy(loc:float = 0.0, scale:float = 1.0, size:int = None)->OneOrMoreFloat:
    u = np.random.uniform(size=size)
    z = ss.norm.ppf(1 - u/2)
    return loc + ( scale / ( 1/z ) **2 )

def skewnormal(shape:OneOrMoreFloat = 0.0, loc:OneOrMoreFloat = 0.0, scale: float = 1.0, size: int = None)->OneOrMoreFloat:
    return ss.skewnorm.rvs(a=shape, loc=loc, scale=scale, size=size)

@dataclass(frozen=True)
class BondedRandom(Bounds):
    def uniform(self, size:int = None)->OneOrMoreFloat:
        return uniform(low=self.min, high=self.max, size=size)
    
    def beta(self, alpha: float = 2.0, beta:float = 2.0, size:int = None)->OneOrMoreFloat:
        return beta(alpha=alpha, beta=beta, size=size) * (self.max - self.min) + self.min
    
    def exponential(self, scale:float = 1.0, size:int = None)->OneOrMoreFloat:
        exp_val = np.clip(np.divide(-np.log(np.random.uniform(size=size)),scale),0,1)
        return self.min + exp_val * (self.max - self.min)
    
    def rayleigh(self, scale:float = 1.0, size:int = None)->OneOrMoreFloat:
        ray_val = np.clip(np.multiply(scale, np.sqrt(-2*np.log(np.random.uniform(size=size)))),0,1)
        return self.min + ray_val * (self.max - self.min)
    
    def weibull(self, shape:float = 1.0, scale:float = 1.0, size:int = None, )->OneOrMoreFloat:
        wei_val = np.clip(weibull(shape=shape, scale=scale, size=size), 0, 1)
        return self.min + wei_val * (self.max - self.min)
    
    def cauchy(self, loc:float = 0.0, scale:float = 1.0, size:int = None)->OneOrMoreFloat:
        c_val = np.clip(cauchy(loc=loc, scale=scale, size=size), 0, 1)
        return self.min + c_val * (self.max - self.min)
    
    def levy(self, loc:float = 0.0, scale:float = 1.0, size:int = None)->OneOrMoreFloat:
        l_val = np.clip(levy(loc=loc, scale=scale, size=size), 0, 1)
        return self.min + l_val * (self.max - self.min)
    
    def normal(self, loc:OneOrMoreFloat = 0.0, scale:OneOrMoreFloat = 1.0, size:int = None)->OneOrMoreFloat:
        return np.clip(normal(loc=loc, scale=scale, size=size), self.min, self.max)

    def lognormal(self, loc:OneOrMoreFloat = 0.0, scale:OneOrMoreFloat = 1.0, size:int = None)->OneOrMoreFloat:
        return np.clip(lognormal(loc=loc, scale=scale, size=size), self.min, self.max)
    
    def skewnormal(self, shape: float = 0.0, loc:OneOrMoreFloat = 0.0, scale:OneOrMoreFloat = 1.0, size:int = None)->OneOrMoreFloat:
        return np.clip(skewnormal(shape=shape, loc=loc, scale=scale, size=size), self.min, self.max)
    
    
