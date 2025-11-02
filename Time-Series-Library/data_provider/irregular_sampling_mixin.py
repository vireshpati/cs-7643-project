"""
This module provides common functionality for applying irregular sampling patterns to time series datasets.
"""

import numpy as np
import pandas as pd
from utils.irregular_sampling import (
    apply_uniform_random_missing,
    apply_bursty_missing,
    apply_adaptive_missing
)


def apply_irregular_sampling_to_dataset(args, data_x, timestamps):

    irregular_sampling = getattr(args, 'irregular_sampling_pattern', 'none')

    if irregular_sampling == 'none':
        # No irregular sampling - all observations are kept
        return np.ones(len(data_x), dtype=bool)

    # Create temporary DataFrame for pattern application
    df_temp = pd.DataFrame(data_x)
    df_temp['date'] = pd.to_datetime(timestamps)

    if irregular_sampling == 'uniform':
        missing_rate = getattr(args, 'irregular_missing_rate', 0.3)
        seed = getattr(args, 'irregular_seed', 42)
        return apply_uniform_random_missing(df_temp, missing_rate, seed)

    elif irregular_sampling == 'bursty':
        p_miss_to_miss = getattr(args, 'irregular_p_miss_to_miss', 0.8)
        p_obs_to_miss = getattr(args, 'irregular_p_obs_to_miss', 0.1)
        seed = getattr(args, 'irregular_seed', 42)
        return apply_bursty_missing(df_temp, p_miss_to_miss, p_obs_to_miss, seed)

    elif irregular_sampling == 'adaptive':
        window_size = getattr(args, 'irregular_window_size', 24)
        target_retention = getattr(args, 'irregular_target_retention', 0.3)
        seed = getattr(args, 'irregular_seed', 42)
        # Get feature columns (exclude 'date')
        feature_cols = [col for col in df_temp.columns if col != 'date']
        return apply_adaptive_missing(df_temp, feature_cols, window_size, target_retention, seed)

    else:
        raise ValueError(f"Unknown irregular sampling pattern: {irregular_sampling}")


def get_irregular_item(data_x, data_y, data_stamp, timestamps, observation_mask,
                       s_begin, s_end, r_begin, r_end, irregular_sampling):

    if irregular_sampling != 'none':
        # Get observed indices within windows
        x_mask = observation_mask[s_begin:s_end]
        y_mask = observation_mask[r_begin:r_end]

        # Get observed data
        seq_x = data_x[s_begin:s_end][x_mask]
        seq_y = data_y[r_begin:r_end][y_mask]
        seq_x_mark = data_stamp[s_begin:s_end][x_mask]
        seq_y_mark = data_stamp[r_begin:r_end][y_mask]

        # Get timestamps for observed data
        x_timestamps = pd.to_datetime(timestamps[s_begin:s_end][x_mask])
        y_timestamps = pd.to_datetime(timestamps[r_begin:r_end][y_mask])

        # Compute time deltas (in hours from start of sequence)
        if len(x_timestamps) > 0:
            x_start = x_timestamps[0]
            seq_x_delta = (x_timestamps - x_start).total_seconds().values / 3600.0
        else:
            seq_x_delta = np.array([])

        if len(y_timestamps) > 0:
            y_start = y_timestamps[0]
            seq_y_delta = (y_timestamps - y_start).total_seconds().values / 3600.0
        else:
            seq_y_delta = np.array([])

        # Store actual lengths
        seq_x_len = len(seq_x)
        seq_y_len = len(seq_y)

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x_delta, seq_y_delta, seq_x_len, seq_y_len
    else:
        # Regular sampling - return original 4-tuple
        seq_x = data_x[s_begin:s_end]
        seq_y = data_y[r_begin:r_end]
        seq_x_mark = data_stamp[s_begin:s_end]
        seq_y_mark = data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark


class IrregularSamplingMixin:

    def setup_irregular_sampling(self, args, data_x, timestamps):

        return apply_irregular_sampling_to_dataset(args, data_x, timestamps)

    def get_item_with_irregular_sampling(self, data_x, data_y, data_stamp, timestamps,
                                        observation_mask, s_begin, s_end, r_begin, r_end,
                                        irregular_sampling):

        return get_irregular_item(data_x, data_y, data_stamp, timestamps, observation_mask,
                                 s_begin, s_end, r_begin, r_end, irregular_sampling)
