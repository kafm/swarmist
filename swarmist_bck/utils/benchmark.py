from __future__ import annotations
from typing import Tuple
import numpy as np
from numba import njit
from swarmist_bck.core.dictionary import FitnessFunction, Bounds

def sphere(opposite:bool=False)->Tuple(FitnessFunction, Bounds):
    return (lambda x: njitSphere(x, opposite), Bounds(min=-5.12, max=5.12))
    
@njit(nogil=True)
def njitSphere(x, opposite):
    evl = sum(x ** 2)
    return - evl if opposite else evl

def ackley(opposite:bool=False)->Tuple(FitnessFunction, Bounds):
    return (lambda x: njitAckley(x, opposite), Bounds(min=-32.768, max=32.768))

@njit(nogil=True)
def njitAckley(x, opposite, a=20, b=0.2, c=2*np.pi):
    n = len(x)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum( np.cos( c * x) )
    term1 = - a * np.exp( -b * np.sqrt( sum1/n ) )
    term2 =  -np.exp(sum2/n)
    evl = term1 + term2 + a + np.exp(1)
    return - evl if opposite else evl

def griewank(opposite:bool=False)->Tuple(FitnessFunction, Bounds):
    return (lambda x: njitGriewank(x, opposite), Bounds(min=-600.0, max=600.0))

@njit(nogil=True)
def njitGriewank(x, opposite, fr=4000):
    n = len(x)
    ii = np.arange( 1., n+1 )
    evl = np.sum( x**2 )/fr - np.prod( np.cos( x / np.sqrt(ii) )) + 1
    return - evl if opposite else evl

def rastrigin(opposite:bool=False)->Tuple(FitnessFunction, Bounds):
    return (lambda x: njitRastrigin(x, opposite), Bounds(min=-5.12, max=5.12))
    
@njit(nogil=True)
def njitRastrigin(x, opposite):
    n = len(x)
    evl = 10 * n + np.sum( x**2 - 10 * np.cos( 2 * np.pi * x ))
    return - evl if opposite else evl

def schwefel(opposite:bool=False)->Tuple(FitnessFunction, Bounds):
    return (lambda x: njitSchwefel(x, opposite),Bounds(min=-500, max=500))

@njit(nogil=True)
def njitSchwefel(x, opposite):
    n = len(x)
    evl = 418.9829 * n - np.sum( x * np.sin( np.sqrt( np.abs( x ))))
    return - evl if opposite else evl

def rosenbrock(opposite:bool=False)->Tuple(FitnessFunction, Bounds):
    return (lambda x: njitRosenbrock(x, opposite),Bounds(min=-2.048, max=2.048))

@njit(nogil=True)
def njitRosenbrock(x, opposite):
    n = len(x)
    xi = x[:-1]
    xnext = x[1:]
    evl = np.sum(100*(xnext-xi**2)**2 + (xi-1)**2)
    return - evl if opposite else evl

def michalewicz(opposite:bool=False)->Tuple(FitnessFunction, Bounds):
    return (lambda x: njitMichalewicz(x, opposite),Bounds(min=0, max=np.pi))

@njit(nogil=True)
def njitMichalewicz(x, opposite, m=10):
    ii = np.arange(1,len(x)+1)
    evl = - sum(np.sin(x) * (np.sin(ii*x**2/np.pi))**(2*m))
    return - evl if opposite else evl

def deJong3(opposite:bool=False)->Tuple(FitnessFunction, Bounds):
    return (lambda x: njitDeJong3(x, opposite),Bounds(min=-5.12, max=5.12))

@njit(nogil=True)
def njitDeJong3(x, opposite):
    evl = np.sum(np.floor(x))
    return - evl if opposite else evl

def deJong5(opposite:bool=False)->Tuple(FitnessFunction, Bounds):
    return (lambda x: njitDeJong5(x, opposite),Bounds(min=-65.536, max=65.536))

@njit(nogil=True)
def njitDeJong5(x, opposite):
    x1 = x[0]
    x2 = x[1]
    a1 = np.array([-32, -16,  0,  16,  32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32])
    a2 = np.array([-32, -32, -32,-32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32])  
    ii = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
    evl = (0.002  + np.sum (1 / (ii + (x1 - a1) ** 6 + (x2 - a2) ** 6 ))) ** -1
    return - evl if opposite else evl

def easom(opposite:bool=False)->Tuple(FitnessFunction, Bounds):
    return (lambda x: njitEasom(x, opposite),Bounds(min=-100, max=100))


@njit(nogil=True)
def njitEasom(x, opposite):
    x1 = x[0]
    x2 = x[1]
    evl = -np.cos(x1) * np.cos(x2) * np.exp(-(x1 - np.pi) ** 2 - (x2 - np.pi) ** 2)
    return - evl if opposite else evl