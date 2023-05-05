#from helpers.random import BondedRandom, rayleigh, exponential
import time
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
#print(BondedRandom(min=0, max=20).rayleigh(size=20))



#print(exponential(scale=.5))

# for _ in range(50000):
#     rand = BondedRandom(min=0, max=20).rayleigh(scale=10)
#     if rand > 10:
#         print(rand)

def gen_truncexpon(scale, lower_bound=0.0, upper_bound=1.0, size=1):
    return ss.truncexpon(1,scale=scale, size=size)

def generate_exponential0(scale, lower_bound=0.0, upper_bound=1.0, size=1):
    exp_val = ss.expon.rvs(scale=scale, size=size)#np.clip(np.random.exponential(scale=scale, size=size),0,1)
    return lower_bound + exp_val * (upper_bound - lower_bound)

def generate_exponential_log(scale, lower_bound=0.0, upper_bound=1.0, size=1):
    exp_val = np.divide(-np.log(np.random.uniform(size=size)),scale)
    return lower_bound + exp_val * (upper_bound - lower_bound)

def generate_exponential(scale, lower_bound=0.0, upper_bound=1.0, size=1):
    # Calculate the cumulative distribution function (CDF) of the exponential distribution
    cdf = ss.expon.cdf(upper_bound - lower_bound, scale=scale)
    # Generate uniform-distributed random variables between 0 and 1
    print(f"cdf={cdf}")
    sample_unit = np.random.uniform(0, cdf, size)
    # Calculate the inverse of the CDF for the exponential distribution
    inv_cdf = ss.expon.ppf(sample_unit, scale=scale)
    # Scale the samples to the range between the lower and upper bounds
    sample = lower_bound + inv_cdf
    
    return sample

def trunc_exp_rv(scale, lower_bound=0.0, upper_bound=1.0, size=1):
    rnd_cdf = np.random.uniform(ss.expon.cdf(x=lower_bound, scale=scale),
                                ss.expon.cdf(x=upper_bound, scale=scale),
                                size=size)
    print(f"rnd_cdf={rnd_cdf}")
    return ss.expon.ppf(q=rnd_cdf, scale=scale)

#print(ss.expon.cdf(10, scale=1))
start_time = time.time()
print("============INVERSE TRANSFORMATION SAMPLING===================")
print(generate_exponential(scale=1.0, lower_bound=10.0, upper_bound=20.0, size=10))
print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
print("============INVERSE TRANSFORMATION SAMPLING(trunc_exp_rv)===================")
print(trunc_exp_rv(scale=1.0, lower_bound=10.0, upper_bound=20.0, size=10))
print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
print("============CANONICAL=========================================")
print(generate_exponential0(scale=1.0, lower_bound=10.0, upper_bound=20.0, size=10))
print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
# print("============truncexpon=========================================")
# #print(gen_truncexpon(scale=1.0, lower_bound=10.0, upper_bound=20.0, size=10))
# print(ss.truncexpon.rvs(1, size=10))
print("============INVERSE TRANSFORMATION SAMPLING (LOG)===================")
print(generate_exponential_log(scale=1.0, lower_bound=10, upper_bound=20, size=10))
print("--- %s seconds ---" % (time.time() - start_time))
print("===========NUMPY EXP===================")
print(np.random.exponential(scale=1.0, size=10))
print("--- %s seconds ---" % (time.time() - start_time))




# Lambda = 2.5 #expected mean of exponential distribution is lambda in Scipy's parameterization
# Size = 1000

# #ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
# #ax.set_xlim([x[0], x[-1]])
# def trunc_exp_rv(low, high, scale, size):
#     # rnd_cdf = np.random.uniform(ss.expon.cdf(x=low, scale=scale),
#     #                             ss.expon.cdf(x=high, scale=scale),
#     #                             size=size)
#     #rnd_cdf = np.random.uniform(low=low,
#     #                             high=high,
#     #                             size=size)
#     #return ss.expon.ppf(q=rnd_cdf, scale=scale)
#     return ss.expon.rvs(size=size, loc=0, scale=1)

# print(trunc_exp_rv(-10, 10, Lambda, Size))
# plt.hist(trunc_exp_rv(-10, 10, Lambda, Size))
# plt.xlim(-12, 12)
# plt.show()