from typing import  List
from dataclasses import dataclass
import math
import numpy as np
import scipy.stats as ss

OneOrMoreFloat = float | List[float]

@dataclass(frozen=True)
class Random:
    size: int

    def rand(self)->List[float]:
        return np.random.rand(self.size)

    def uniform(self, low:float = 0.0, high:float = 1.0)->List[float]:
        return np.random.uniform(low=low, high=high, size=self.size)

    def beta(self, alpha:float = 2.0, beta:float = 2.0)->List[float]:
        return  np.random.beta(alpha, beta, size=self.size)
    
    #scale is the expected mean
    def exponential(self, scale:float = 1.0)->List[float]:
        return np.random.exponential(scale=scale, size=self.size)

    #scale is the expected mode
    def rayleigh(self, scale:float = 1.0)->List[float]:
        return np.random.rayleigh(scale=scale,size=self.size)

    def normal(self, loc:OneOrMoreFloat = 0.0, scale:OneOrMoreFloat = 1.0)->List[float]:
        return np.random.normal(loc=loc, scale=scale, size=self.size)

    def lognormal(self, loc:OneOrMoreFloat = 0.0, scale:OneOrMoreFloat = 1.0)->List[float]:
        return np.random.lognormal(loc=loc, scale=scale, size=self.size)

    #k=shape
    def weibull(self, shape:float = 1.0, scale:float = 1.0, size: int = None)->List[float]:
        return np.random.weibull(shape, scale=scale, size=self.size)

    #loc=location
    def cauchy(self, loc:float = 0.0, scale:float = 1.0)->List[float]:
        u = np.random.uniform(size=self.size)
        return loc + ( scale * np.tan(np.pi * (u-.5) ) )
    
    def skewnormal(self, shape:OneOrMoreFloat = 0.0, loc:OneOrMoreFloat = 0.0, scale: float = 1.0, size: int = None)->List[float]:
        return ss.skewnorm.rvs(a=shape, loc=loc, scale=scale, size=self.size)    


    def levy(self, loc:float = 0.0, scale:float = 1.0)->List[float]:
        u = np.random.uniform(size=self.size)
        z = ss.norm.ppf(1 - u/2)
        return loc + ( scale / ( 1/z ) **2 )

    def levy2(self, loc: float = 0.0, scale: float = 1.0, size: int = None)->List[float]:
        beta = loc if loc > 1 else 1
        gamma1 = math.gamma(1+beta)
        gamma2 = math.gamma((1+beta)/2)
        sigma = (gamma1*math.sin(math.pi*beta/2)/(gamma2*beta*2**((beta-1)/2))) ** (1/beta)
        u = np.random.normal(0,1, size=self.size) * sigma
        v = np.random.normal(0,1, size=self.size) 
        return u / abs(v) ** (1 / beta)

    
#TODO migrate to initializers
@dataclass(frozen=True)
class BondedRandom(Random):
    lbound: float
    ubound: float

    def uniform(self)->List[float]:
        return super().uniform(low=self.lbound, high=self.ubound)
    
    def beta(self, alpha: float = 2.0, beta:float = 2.0)->List[float]:
        return super().beta(alpha=alpha, beta=beta) * (self.ubound - self.lbound) + self.lbound
    
    def exponential(self, scale:float = 1.0)->List[float]:
        exp_val = np.clip(np.divide(-np.log(super().rand()),scale),0,1)
        return self.lbound + exp_val * (self.ubound - self.lbound)
    
    def rayleigh(self, scale:float = 1.0)->List[float]:
        ray_val = np.clip(np.multiply(scale, np.sqrt(-2*np.log(super().rand()))),0,1)
        return self.lbound + ray_val * (self.ubound - self.lbound)
    
    def weibull(self, shape:float = 1.0, scale:float = 1.0, )->List[float]:
        wei_val = np.clip(super().weibull(shape=shape, scale=scale), 0, 1)
        return self.lbound + wei_val * (self.ubound - self.lbound)
    
    def cauchy(self, loc:float = 0.0, scale:float = 1.0)->List[float]:
        c_val = np.clip(super().cauchy(loc=loc, scale=scale), 0, 1)
        return self.lbound + c_val * (self.ubound - self.lbound)
    
    def levy(self, loc:float = 0.0, scale:float = 1.0)->List[float]:
        l_val = np.clip(super().levy(loc=loc, scale=scale), 0, 1)
        return self.lbound + l_val * (self.ubound - self.lbound)
    
    def normal(self, loc:OneOrMoreFloat = 0.0, scale:OneOrMoreFloat = 1.0)->List[float]:
        return np.clip(super().normal(loc=loc, scale=scale), self.lbound, self.ubound)

    def lognormal(self, loc:OneOrMoreFloat = 0.0, scale:OneOrMoreFloat = 1.0)->List[float]:
        return np.clip(super().lognormal(loc=loc, scale=scale), self.lbound, self.ubound)
    
    def skewnormal(self, shape: float = 0.0, loc:OneOrMoreFloat = 0.0, scale:OneOrMoreFloat = 1.0)->List[float]:
        return np.clip(super().skewnormal(shape=shape, loc=loc, scale=scale), self.lbound, self.ubound)