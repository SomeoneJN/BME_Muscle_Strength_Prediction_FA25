# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 23:05:51 2025

@author: nerij
"""

# noise_and_plot_functions.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev, CubicSpline

def add_noise_simple(arr, noise_level):
    """
    Add random Gaussian noise to an array using noise_level as sigma.
    """
    return arr + np.random.normal(0, noise_level, size=arr.shape)

def create_noisy_shape(row, shape_x_cols, shape_y_cols, noise_level=0.1):
    """
    Fit a periodic spline to the 9 shape points from the row,
    sample 100 equally spaced points along the spline, add noise (using noise_level)
    to these points, and then select 10 equidistant indices (with the first and last identical)
    so that the shape is closed.
    
    Returns:
        x_noisy: A 1D NumPy array of 10 sample x values.
        y_noisy: A 1D NumPy array of 10 sample y values.
    """
    x = row[shape_x_cols].to_numpy(dtype=float)
    y = row[shape_y_cols].to_numpy(dtype=float)
    tck, _ = splprep([x, y], s=0, per=True)
    u_new = np.linspace(0, 1, 100)
    x_spline, y_spline = splev(u_new, tck)
    x_noisy_full = x_spline + np.random.normal(0, noise_level, size=x_spline.shape)
    y_noisy_full = y_spline + np.random.normal(0, noise_level, size=y_spline.shape)
    indices = np.linspace(0, 99, 10, endpoint=True).astype(int)
    return x_noisy_full[indices], y_noisy_full[indices]

def get_smooth_curve(x_points, y_points, num_points=200):
    """
    Given a set of sample points (x_points, y_points), fit a periodic spline
    (assuming the points define a closed curve) and return dense smooth curves.
    
    Returns:
        x_dense, y_dense: Dense NumPy arrays representing the smooth curve.
    """
    tck, _ = splprep([x_points, y_points], s=0, per=True)
    u_dense = np.linspace(0, 1, num_points)
    x_dense, y_dense = splev(u_dense, tck)
    return x_dense, y_dense

def get_circular_spline_from_points(x_points, y_points, num_points=200):
    """
    Compute a circular spline from the 10 sample points as follows:
      1) Compute the center as the mean of x_points and y_points.
      2) Compute each point's angle (theta) relative to the center and its radius.
      3) Sort the angles and corresponding radii; append the first point (angle+2π) for periodicity.
      4) Fit a periodic cubic spline (using CubicSpline with bc_type='periodic')
         to interpolate the radii as a function of angle.
      5) Generate dense angles and compute the corresponding (x, y) coordinates.
      
    Returns:
        x_dense, y_dense, center_x, center_y
    """
    center_x = np.mean(x_points)
    center_y = np.mean(y_points)
    angles = np.arctan2(y_points - center_y, x_points - center_x)
    radii = np.sqrt((x_points - center_x)**2 + (y_points - center_y)**2)
    sort_idx = np.argsort(angles)
    sorted_angles = angles[sort_idx]
    sorted_radii = radii[sort_idx]
    sorted_angles = np.concatenate([sorted_angles, [sorted_angles[0] + 2*np.pi]])
    sorted_radii = np.concatenate([sorted_radii, [sorted_radii[0]]])
    cs = CubicSpline(sorted_angles, sorted_radii, bc_type='periodic')
    dense_angles = np.linspace(sorted_angles[0], sorted_angles[-1], num_points)
    dense_radii = cs(dense_angles)
    x_dense = center_x + dense_radii * np.cos(dense_angles)
    y_dense = center_y + dense_radii * np.sin(dense_angles)
    return x_dense, y_dense, center_x, center_y

def get_perfect_circle_line(x_points, y_points, num_points=200, use_median=False):
    """
    Compute a perfect circle from sample points by calculating the center and an average radius.
    Optionally, use the median for a robust estimate.
    
    Returns:
        x_circle, y_circle, center_x, center_y
    """
    center_x = np.median(x_points) if use_median else np.mean(x_points)
    center_y = np.median(y_points) if use_median else np.mean(y_points)
    radii = np.sqrt((x_points - center_x)**2 + (y_points - center_y)**2)
    r_avg = np.median(radii) if use_median else np.mean(radii)
    theta = np.linspace(0, 2*np.pi, num_points)
    x_circle = center_x + r_avg * np.cos(theta)
    y_circle = center_y + r_avg * np.sin(theta)
    return x_circle, y_circle, center_x, center_y

def plot_perfect_circle(ax, x_points, y_points, label='Perfect Circle', circle_color='purple'):
    """
    Compute a perfect circle from sample points by calculating the center and average radius,
    then plot that circle on the axis with markers at the sample points (without annotations).
    """
    center_x = np.mean(x_points)
    center_y = np.mean(y_points)
    radii = np.sqrt((x_points - center_x)**2 + (y_points - center_y)**2)
    r_avg = np.mean(radii)
    theta = np.linspace(0, 2*np.pi, 200)
    x_circle = center_x + r_avg * np.cos(theta)
    y_circle = center_y + r_avg * np.sin(theta)
    ax.plot(x_circle, y_circle, color=circle_color, linestyle='-', linewidth=2, label=label)
    ax.scatter(x_points, y_points, color=circle_color, marker='D', s=40)
    # (No annotations for coordinates)

def compute_means_pair(data, cols_x, cols_y, groups=5):
    """
    Split the DataFrame into 'groups' nearly equal parts and compute the column‐wise means
    of the specified columns.
    
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
    Plot curves from normal and noisy data on the provided axis.
    Labels are added only for the first normal and first noisy curve.
    """
    first_normal, first_noisy = True, True
    for _, row in normal_df.iterrows():
        x = row[cols_x].to_numpy(dtype=float)
        y = row[cols_y].to_numpy(dtype=float)
        if first_normal:
            ax.plot(x, y, marker='o', linestyle='-', color='blue', alpha=0.3, label='Normal')
            first_normal = False
        else:
            ax.plot(x, y, marker='o', linestyle='-', color='blue', alpha=0.3)
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
    Plot smooth mean curves (normal and noisy) on the axis based on the 10 sample points,
    and overlay example curves.
    For normal means, use a perfect circular spline and plot the sample points.
    For noisy means, use a perfect circular spline.
    Labels are added only once.
    """
    for i in range(norm_mean_x.shape[0]):
        x_dense, y_dense, _, _ = get_perfect_circle_line(norm_mean_x[i], norm_mean_y[i], num_points=200)
        ax.plot(x_dense, y_dense, color='navy', linewidth=3,
                label='Normal Mean' if i==0 else "")
        # Plot normal sample points
        ax.scatter(norm_mean_x[i], norm_mean_y[i], color='navy', marker='o', s=40)
    for i in range(noisy_mean_x.shape[0]):
        x_dense, y_dense, _, _ = get_perfect_circle_line(noisy_mean_x[i], noisy_mean_y[i], num_points=200)
        ax.plot(x_dense, y_dense, color='darkred', linestyle='--', linewidth=3,
                label='Noisy Mean' if i==0 else "")
    for idx, row in noisy_examples.iterrows():
        x_sample = row[cols_x].to_numpy(dtype=float)
        y_sample = row[cols_y].to_numpy(dtype=float)
        x_dense, y_dense, _, _ = get_perfect_circle_line(x_sample, y_sample, num_points=200)
        ax.plot(x_dense, y_dense, marker='x', linestyle=':', color='orange', alpha=0.7,
                label='Noisy Example' if idx==0 else "")
    ax.set_title(title)
    ax.set_xlabel(cols_x[0].split('_')[0])
    ax.set_ylabel(cols_y[0].split('_')[0])
    ax.legend(loc='upper right')
