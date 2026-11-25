"""Within-list rank percentiles for the SM-RS catalog features."""
from scipy.stats import rankdata


def percentile(values):
    return (rankdata(values, method="average") - 0.5) / len(values)
