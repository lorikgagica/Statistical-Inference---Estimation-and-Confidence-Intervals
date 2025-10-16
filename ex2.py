# Exercise 2: Conduct Sampling and Create a Report
# url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

import pandas as pd
from scipy.stats import norm
import numpy as np

# Load Iris dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

# Sampling
sample = df["sepal_length"].sample(30, random_state=42)

# Sample statistics
mean = sample.mean()
std = sample.std()
n = len(sample)

# Confidence Interval
z_value = norm.ppf(0.975)
margin_of_error = z_value * (std / np.sqrt(n))
ci = (mean - margin_of_error, mean + margin_of_error)

print("Sample Mean: ", mean)
print("95% Confidence Interval: ", ci)


# Additional Practice:
# A. Create confidence intervals for other statistics.
# B. Perform stratified sampling and compare intervals across strata.
# C. Visualize confidence intervals for multipe samples using Matplotlib.




