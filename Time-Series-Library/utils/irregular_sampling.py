"""
Irregular Sampling Patterns for Time Series

Implements three canonical patterns of irregular sampling to simulate real-world scenarios:
- Pattern A: Uniform random missing (i.i.d.)
- Pattern B: Bursty/clustered missing (Markov chain)
- Pattern C: Adaptive/variance-driven sampling

Each pattern returns a boolean mask indicating which timestamps to keep.
"""

import numpy as np
import pandas as pd
from typing import Union, List


def apply_uniform_random_missing(df: pd.DataFrame, missing_rate: float, seed: int = None) -> np.ndarray:
    if not 0.0 <= missing_rate <= 1.0:
        raise ValueError(f"missing_rate must be between 0 and 1, got {missing_rate}")
    if df is None:
        raise ValueError("df cannot be None")

    if seed is not None:
        np.random.seed(seed)

    mask = np.random.binomial(1, 1 - missing_rate, size=len(df)).astype(bool)

    # Ensure at least one observation
    if not mask.any():
        mask[0] = True

    return mask


def apply_bursty_missing(df: pd.DataFrame,
                        p_miss_to_miss: float = 0.8,
                        p_obs_to_miss: float = 0.1,
                        seed: int = None) -> np.ndarray:
    
    if not 0.0 <= p_miss_to_miss <= 1.0:
        raise ValueError(f"p_miss_to_miss must be between 0 and 1, got {p_miss_to_miss}")
    if not 0.0 <= p_obs_to_miss <= 1.0:
        raise ValueError(f"p_obs_to_miss must be between 0 and 1, got {p_obs_to_miss}")
    if df is None:
        raise ValueError("df cannot be None")
    if seed is not None:
        np.random.seed(seed)

    n = len(df)
    state = 0  # Start in observed state
    mask = np.zeros(n, dtype=bool)

    for i in range(n):
        if state == 0:  # Currently observed
            state = np.random.binomial(1, p_obs_to_miss)
        else:  # Currently missing
            state = np.random.binomial(1, p_miss_to_miss)

        mask[i] = (state == 0)

    # Ensure at least one observation
    if not mask.any():
        mask[0] = True

    return mask


def apply_adaptive_missing(df: pd.DataFrame,
                          feature_cols: Union[List[str], None] = None,
                          window_size: int = 24,
                          target_retention: float = 0.3,
                          seed: int = None) -> np.ndarray:

    if not 0.0 < target_retention <= 1.0:
        raise ValueError(f"target_retention must be between 0 and 1, got {target_retention}")
    if window_size <= 0:
        raise ValueError(f"window_size must be greater than 0, got {window_size}")
    if df is None:
        raise ValueError("df cannot be None")
    if seed is not None:
        np.random.seed(seed)

    # Determine which columns to use for variance computation
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col.lower() != 'date']

    if len(feature_cols) == 0:
        raise ValueError("No feature columns found for variance computation")

    # Compute local variance
    features = df[feature_cols].values
    n = len(df)

    df_temp = pd.DataFrame(features)
    local_var = df_temp.rolling(window=window_size, center=True, min_periods=1).var()
    local_var = local_var.mean(axis=1).values

    # Handle NaN values
    local_var = np.nan_to_num(local_var, nan=np.nanmedian(local_var))

    # Normalize variance to [0, 1]
    var_min, var_max = local_var.min(), local_var.max()
    if var_max > var_min:
        local_var_norm = (local_var - var_min) / (var_max - var_min)
    else:
        local_var_norm = np.ones(n) * 0.5

    # Apply sigmoid transformation
    percentile_val = np.percentile(local_var_norm, 50)
    shifted_var = (local_var_norm - percentile_val) * 10
    probs = 1 / (1 + np.exp(-shifted_var))

    # Scale to hit target retention rate
    current_mean = probs.mean()
    if current_mean > 0:
        probs = probs * (target_retention / current_mean)
    else:
        probs = np.ones(n) * target_retention

    probs = np.clip(probs, 0.0, 1.0)

    # Sample based on probabilities
    mask = np.random.binomial(1, probs).astype(bool)

    # Ensure at least one observation
    if not mask.any():
        mask[np.argmax(local_var)] = True

    return mask


def get_pattern_statistics(mask: np.ndarray) -> dict:

    n_total = len(mask)
    n_observed = mask.sum()
    retention_rate = n_observed / n_total if n_total > 0 else 0.0

    observed_indices = np.where(mask)[0]

    if len(observed_indices) > 1:
        gaps = np.diff(observed_indices)
        avg_gap = gaps.mean()
        max_gap = gaps.max()
        gap_std = gaps.std()
    else:
        avg_gap = 0.0
        max_gap = 0.0
        gap_std = 0.0

    return {
        'retention_rate': retention_rate,
        'n_observed': n_observed,
        'n_total': n_total,
        'avg_gap': avg_gap,
        'max_gap': max_gap,
        'gap_std': gap_std,
    }
