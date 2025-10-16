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
# Confidence interval for "petal_length"
sample_petal = df["petal_length"].sample(30, random_state=42)
mean_petal = sample_petal.mean()
std_petal = sample_petal.std()
n_petal = len(sample_petal)
z_value = norm.ppf(0.975)
margin_of_error_petal = z_value * (std_petal / np.sqrt(n_petal))
ci_petal = (mean_petal - margin_of_error_petal, mean_petal + margin_of_error_petal)
print("Petal Length Mean:", mean_petal)
print("95% CI for Petal Length:", ci_petal)


# B. Perform stratified sampling and compare intervals across strata.
# Stratified sampling by species
for species, group in df.groupby('species'):
    sample_stratum = group['sepal_length'].sample(15, random_state=42)
    mean_stratum = sample_stratum.mean()
    std_stratum = sample_stratum.std()
    n_stratum = len(sample_stratum)
    margin_stratum = z_value * (std_stratum / np.sqrt(n_stratum))
    ci_stratum = (mean_stratum - margin_stratum, mean_stratum + margin_stratum)
    print(f"{species} - Mean: {mean_stratum:.2f}, 95% CI: {ci_stratum}")
    
    
# C. Visualize confidence intervals for multipe samples using Matplotlib.
import matplotlib.pyplot as plt

sample_means = []
lower_cis = []
upper_cis = []
for i in range(10):
    sample = df["sepal_length"].sample(30, random_state=42 + i)
    mean_ = sample.mean()
    std_ = sample.std()
    n_ = len(sample)
    margin = z_value * (std_ / np.sqrt(n_))
    ci = (mean_ - margin, mean_ + margin)
    sample_means.append(mean_)
    lower_cis.append(ci[0])
    upper_cis.append(ci[1])

plt.errorbar(range(1, 11), sample_means, 
             yerr=[np.array(sample_means) - np.array(lower_cis), 
                   np.array(upper_cis) - np.array(sample_means)],
             fmt='o', capsize=5)
plt.title("Confidence Intervals for Sample Means (Sepal Length)")
plt.xlabel("Sample Number")
plt.ylabel("Sample Mean Sepal Length")
plt.grid(True)
plt.show()



