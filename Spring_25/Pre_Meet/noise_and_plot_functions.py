# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 10:25:09 2025

@author: nerij
"""

# noise_and_plot_functions.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

def add_noise_simple(arr, noise_level):
    """
    Add random Gaussian noise to an array using the specified noise_level (sigma).
    """
    return arr + np.random.normal(0, noise_level, size=arr.shape)

def create_noisy_shape(row, shape_x_cols, shape_y_cols, noise_level=0.1):
    """
    Fit a circular (periodic) spline through 9 shape points from a given row,
    sample 100 points along the spline, add noise to those points (using the given noise_level),
    and then select 9 equidistant points from the 100 samples as the new noisy shape.
    
    Returns:
        x_noisy: 1D NumPy array of 9 noisy x values.
        y_noisy: 1D NumPy array of 9 noisy y values.
    """
    x = row[shape_x_cols].to_numpy(dtype=float)
    y = row[shape_y_cols].to_numpy(dtype=float)
    tck, _ = splprep([x, y], s=0, per=True)
    u_new = np.linspace(0, 1, 100)
    x_spline, y_spline = splev(u_new, tck)
    x_noisy_full = x_spline + np.random.normal(0, noise_level, size=x_spline.shape)
    y_noisy_full = y_spline + np.random.normal(0, noise_level, size=y_spline.shape)
    indices = np.linspace(0, 99, 9, dtype=int)
    return x_noisy_full[indices], y_noisy_full[indices]

def compute_means_pair(data, cols_x, cols_y, groups=5):
    """
    Split the DataFrame into 'groups' equal parts and compute the mean (column-wise) of the specified columns.
    
    Returns:
        mean_x: NumPy array of shape (groups, len(cols_x))
        mean_y: NumPy array of shape (groups, len(cols_y))
    """
    splits = np.array_split(data, groups)
    mean_x = [grp[cols_x].mean().to_numpy() for grp in splits]
    mean_y = [grp[cols_y].mean().to_numpy() for grp in splits]
    return np.array(mean_x), np.array(mean_y)

def plot_comparison(ax, normal_df, noisy_df, cols_x, cols_y, title):
    """
    Plot all curves from normal and noisy data on a given axis.
    
    Normal curves are drawn in blue (with light opacity) and noisy curves in red.
    """
    for _, row in normal_df.iterrows():
        x = row[cols_x].to_numpy(dtype=float)
        y = row[cols_y].to_numpy(dtype=float)
        ax.plot(x, y, marker='o', linestyle='-', color='blue', alpha=0.3)
    for _, row in noisy_df.iterrows():
        x = row[cols_x].to_numpy(dtype=float)
        y = row[cols_y].to_numpy(dtype=float)
        ax.plot(x, y, marker='s', linestyle='--', color='red', alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel(cols_x[0].split('_')[0])
    ax.set_ylabel(cols_y[0].split('_')[0])
    ax.legend(['Normal', 'Noisy'], loc='upper right')

def plot_means(ax, norm_mean_x, norm_mean_y, noisy_mean_x, noisy_mean_y, noisy_examples, cols_x, cols_y, title):
    """
    Plot 5 mean curves (normal and noisy) on the given axis.
    
    Mean curves are drawn with thick lines (dark blue for normal, dark red for noisy) and
    overlaid noisy example curves are displayed in orange.
    """
    for i in range(norm_mean_x.shape[0]):
        ax.plot(norm_mean_x[i], norm_mean_y[i], marker='o', linestyle='-', color='navy', linewidth=3)
    for i in range(noisy_mean_x.shape[0]):
        ax.plot(noisy_mean_x[i], noisy_mean_y[i], marker='s', linestyle='--', color='darkred', linewidth=3)
    for _, row in noisy_examples.iterrows():
        x = row[cols_x].to_numpy(dtype=float)
        y = row[cols_y].to_numpy(dtype=float)
        ax.plot(x, y, marker='x', linestyle=':', color='orange', alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(cols_x[0].split('_')[0])
    ax.set_ylabel(cols_y[0].split('_')[0])
    ax.legend(['Normal Mean', 'Noisy Mean', 'Noisy Example'], loc='upper right')
