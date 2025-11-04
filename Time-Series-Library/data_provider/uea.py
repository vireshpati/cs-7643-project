import os
import numpy as np
import pandas as pd
import torch


def collate_fn(data, max_len=None):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
    """

    batch_size = len(data)
    features, labels = zip(*data)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)

    X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]

    targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                 max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep

    return X, targets, padding_masks


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(self, norm_type='standardization', mean=None, std=None, min_val=None, max_val=None):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform('mean')) / grouped.transform('std')

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform('min')
            return (df - min_vals) / (grouped.transform('max') - min_vals + np.finfo(float).eps)

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))


def interpolate_missing(y):
    """
    Replaces NaN values in pd.Series `y` using linear interpolation
    """
    if y.isna().any():
        y = y.interpolate(method='linear', limit_direction='both')
    return y


def subsample(y, limit=256, factor=2):
    """
    If a given Series is longer than `limit`, returns subsampled sequence by the specified integer factor
    """
    if len(y) > limit:
        return y[::factor].reset_index(drop=True)
    return y


def collate_fn_irregular(batch):
    """
    Collate function for irregular sampling with variable-length sequences.

    Args:
        batch: List of 8-tuples from dataset __getitem__:
            (seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x_delta, seq_y_delta, seq_x_len, seq_y_len)

    Returns:
        Tuple of padded tensors with padding masks:
        (seq_x_batch, seq_y_batch, seq_x_mark_batch, seq_y_mark_batch,
         seq_x_delta_batch, seq_y_delta_batch, x_padding_mask, y_padding_mask)
    """
    # Unpack batch
    seq_x_list, seq_y_list, seq_x_mark_list, seq_y_mark_list, \
        seq_x_delta_list, seq_y_delta_list, seq_x_len_list, seq_y_len_list = zip(*batch)

    batch_size = len(batch)

    # Find max lengths in batch
    max_x_len = max(seq_x_len_list)
    max_y_len = max(seq_y_len_list)

    # Get feature dimensions
    x_feat_dim = seq_x_list[0].shape[-1]
    y_feat_dim = seq_y_list[0].shape[-1]
    x_mark_dim = seq_x_mark_list[0].shape[-1]
    y_mark_dim = seq_y_mark_list[0].shape[-1]

    # Initialize padded tensors
    seq_x_batch = torch.zeros(batch_size, max_x_len, x_feat_dim)
    seq_y_batch = torch.zeros(batch_size, max_y_len, y_feat_dim)
    seq_x_mark_batch = torch.zeros(batch_size, max_x_len, x_mark_dim)
    seq_y_mark_batch = torch.zeros(batch_size, max_y_len, y_mark_dim)
    seq_x_delta_batch = torch.zeros(batch_size, max_x_len)
    seq_y_delta_batch = torch.zeros(batch_size, max_y_len)

    # Fill in actual data
    for i in range(batch_size):
        x_len = seq_x_len_list[i]
        y_len = seq_y_len_list[i]

        if x_len > 0:
            seq_x_batch[i, :x_len, :] = torch.from_numpy(seq_x_list[i])
            seq_x_mark_batch[i, :x_len, :] = torch.from_numpy(seq_x_mark_list[i])
            seq_x_delta_batch[i, :x_len] = torch.from_numpy(seq_x_delta_list[i])

        if y_len > 0:
            seq_y_batch[i, :y_len, :] = torch.from_numpy(seq_y_list[i])
            seq_y_mark_batch[i, :y_len, :] = torch.from_numpy(seq_y_mark_list[i])
            seq_y_delta_batch[i, :y_len] = torch.from_numpy(seq_y_delta_list[i])

    # Create padding masks (True = keep, False = padding)
    x_lengths = torch.tensor(seq_x_len_list, dtype=torch.long)
    y_lengths = torch.tensor(seq_y_len_list, dtype=torch.long)
    x_padding_mask = padding_mask(x_lengths, max_len=max_x_len)
    y_padding_mask = padding_mask(y_lengths, max_len=max_y_len)

    return (seq_x_batch, seq_y_batch, seq_x_mark_batch, seq_y_mark_batch,
            seq_x_delta_batch, seq_y_delta_batch, x_padding_mask, y_padding_mask)
