from scipy.stats import norm, binom, poisson, laplace, chi2, weibull_min, weibull_max, cauchy, gamma, beta, uniform, bernoulli
import numpy as np
import os

os.makedirs("data_random", exist_ok=True)

distributions = [
    lambda size: norm.rvs(size=size), 
    lambda size: norm.rvs(loc=5, scale=2, size=size),
    lambda size: binom.rvs(10, 0.5, size=size),
    lambda size: binom.rvs(20, 0.25, size=size),
    lambda size: binom.rvs(100, 0.62, size=size),
    lambda size: chi2.rvs(5, size=size),
    lambda size: laplace.rvs(size=size),
    lambda size: poisson.rvs(5, size=size),
    lambda size: weibull_min.rvs(1, size=size),
    lambda size: weibull_max.rvs(1, size=size),
    lambda size: cauchy.rvs(size=size),
    lambda size: gamma.rvs(1, size=size),
    lambda size: beta.rvs(2, 5, size=size),
    lambda size: uniform.rvs(size=size),
    lambda size: bernoulli.rvs(0.5, size=size),
    lambda size: bernoulli.rvs(0.2, size=size),
    lambda size: bernoulli.rvs(0.7, size=size),
]

distribution = np.random.choice(distributions)
number_of_symbols = np.random.randint(1, 257)
file_size = int(np.random.lognormal(mean=np.log(2**24), sigma=3))
file_size = min(file_size, 2**26)

samples = distribution(file_size)
samples -= samples.min()             # shift to [0, max]
samples = samples / samples.max()    # normalize to [0, 1]
samples *= number_of_symbols         # scale to [0, number_of_symbols]
samples = samples.astype(np.uint8)

with open(f"data_random/random_1x{file_size}.bin", 'wb') as f:
    f.write(samples)

print(f"data_random/random_1x{file_size}.bin")
