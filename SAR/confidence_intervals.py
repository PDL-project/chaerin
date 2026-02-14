import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

def get_CP_interval(outcomes):
    ci_low, ci_high = sm.stats.proportion_confint(
        sum(outcomes), len(outcomes), alpha=0.05, method="beta"
    )
    return np.mean(outcomes), ci_low, ci_high


def get_mean_std_interval(data):
    mean = np.mean(data)
    std_error = stats.sem(data)

    # Set the desired confidence level (e.g., 95%)
    confidence_level = 0.95

    # Calculate the margin of error
    margin_of_error = stats.t.ppf((1 + confidence_level) / 2, len(data) - 1) * std_error

    # Calculate the confidence interval
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error

    return mean, lower_bound, upper_bound


def get_ci_metrics(df):
    # get the mean transport rate and confidence intervals (Interquartile range)
    metrics = {}
    metrics["success rate"] = get_CP_interval(df["success rate"])
    metrics["transport rate"] = get_mean_std_interval(df["transport rate"])
    metrics["coverage"] = get_mean_std_interval(df["coverage"])
    metrics["balance"] = get_mean_std_interval(df["balance"])
    metrics["steps"] = get_mean_std_interval(df["steps"])
    return metrics
