# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 11:22:26 2025

@author: nerij
"""

# noise_and_plot_functions.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

def add_noise_simple(arr, noise_level):
    """
    Add random Gaussian noise to an array using the specified noise_level as sigma.
    """
    return arr + np.random.normal(0, noise_level, size=arr.shape)

def create_noisy_shape(row, shape_x_cols, shape_y_cols, noise_level=0.1):
    """
    Fit a circular (periodic) spline through the 9 shape points from a row,
    sample 100 equally spaced points along the spline, add noise (using noise_level)
    to these 100 points, and then select 10 equidistant points (including the last
    point which equals the first) so that the plotted shape is closed.
    
    Returns:
        x_noisy: A 1D NumPy array of 10 noisy x values.
        y_noisy: A 1D NumPy array of 10 noisy y values.
    """
    # Extract original shape points as float arrays.
    x = row[shape_x_cols].to_numpy(dtype=float)
    y = row[shape_y_cols].to_numpy(dtype=float)
    # Fit a periodic spline.
    tck, _ = splprep([x, y], s=0, per=True)
    u_new = np.linspace(0, 1, 100)
    x_spline, y_spline = splev(u_new, tck)
    # Add noise to the spline sampled points.
    x_noisy_full = x_spline + np.random.normal(0, noise_level, size=x_spline.shape)
    y_noisy_full = y_spline + np.random.normal(0, noise_level, size=y_spline.shape)
    # Select 10 equidistant indices. For a periodic spline with per=True,
    # the point at u=0 and u=1 are identical.
    indices = np.linspace(0, 99, 10, endpoint=True).astype(int)
    return x_noisy_full[indices], y_noisy_full[indices]

def compute_means_pair(data, cols_x, cols_y, groups=5):
    """
    Split the DataFrame into ‘groups’ nearly equal parts and compute the column‐wise means.
    
    Returns:
        mean_x: A NumPy array of shape (groups, number_of_columns).
        mean_y: A NumPy array of shape (groups, number_of_columns).
    """
    splits = np.array_split(data, groups)
    mean_x = [grp[cols_x].mean().to_numpy() for grp in splits]
    mean_y = [grp[cols_y].mean().to_numpy() for grp in splits]
    return np.array(mean_x), np.array(mean_y)

def plot_comparison(ax, normal_df, noisy_df, cols_x, cols_y, title):
    """
    Plot curves from normal and noisy data onto the provided axis.
    Only the first normal and noisy lines get labels for a clean legend.
    """
    first_normal = True
    for _, row in normal_df.iterrows():
        x = row[cols_x].to_numpy(dtype=float)
        y = row[cols_y].to_numpy(dtype=float)
        if first_normal:
            ax.plot(x, y, marker='o', linestyle='-', color='blue', alpha=0.3, label='Normal')
            first_normal = False
        else:
            ax.plot(x, y, marker='o', linestyle='-', color='blue', alpha=0.3)
    first_noisy = True
    for _, row in noisy_df.iterrows():
        x = row[cols_x].to_numpy(dtype=float)
        y = row[cols_y].to_numpy(dtype=float)
        if first_noisy:
            ax.plot(x, y, marker='s', linestyle='--', color='red', alpha=0.3, label='Noisy')
            first_noisy = False
        else:
            ax.plot(x, y, marker='s', linestyle='--', color='red', alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel(cols_x[0].split('_')[0])
    ax.set_ylabel(cols_y[0].split('_')[0])
    ax.legend(loc='upper right')

def plot_means(ax, norm_mean_x, norm_mean_y, noisy_mean_x, noisy_mean_y, noisy_examples, cols_x, cols_y, title):
    """
    Plot the mean curves (normal and noisy) on the given axis along with overlaying example curves.
    Only the first instance of each type is labeled.
    """
    for i in range(norm_mean_x.shape[0]):
        ax.plot(norm_mean_x[i], norm_mean_y[i], marker='o', linestyle='-', color='navy', linewidth=3,
                label='Normal Mean' if i==0 else "")
    for i in range(noisy_mean_x.shape[0]):
        ax.plot(noisy_mean_x[i], noisy_mean_y[i], marker='s', linestyle='--', color='darkred', linewidth=3,
                label='Noisy Mean' if i==0 else "")
    for idx, row in noisy_examples.iterrows():
        x = row[cols_x].to_numpy(dtype=float)
        y = row[cols_y].to_numpy(dtype=float)
        ax.plot(x, y, marker='x', linestyle=':', color='orange', alpha=0.7,
                label='Noisy Example' if idx==0 else "")
    ax.set_title(title)
    ax.set_xlabel(cols_x[0].split('_')[0])
    ax.set_ylabel(cols_y[0].split('_')[0])
    ax.legend(loc='upper right')
