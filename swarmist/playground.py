#from helpers.random import BondedRandom, rayleigh, exponential
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
#print(BondedRandom(min=0, max=20).rayleigh(size=20))



#print(exponential(scale=.5))

# for _ in range(50000):
#     rand = BondedRandom(min=0, max=20).rayleigh(scale=10)
#     if rand > 10:
#         print(rand)

def generate_exponential(scale, lower_bound=0.0, upper_bound=1.0, size=1):
    # Calculate the cumulative distribution function (CDF) of the exponential distribution
    cdf = ss.expon.cdf(upper_bound - lower_bound, scale=scale)
    
    # Generate uniform-distributed random variables between 0 and 1
    sample_unit = np.random.uniform(0, cdf, size)
    
    # Calculate the inverse of the CDF for the exponential distribution
    inv_cdf = ss.expon.ppf(sample_unit, scale=scale)
    
    # Scale the samples to the range between the lower and upper bounds
    sample = lower_bound + inv_cdf
    
    return sample
print(ss.expon.cdf(10, scale=10))
print(generate_exponential(scale=1.0, lower_bound=10.0, upper_bound=20.0, size=10))


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