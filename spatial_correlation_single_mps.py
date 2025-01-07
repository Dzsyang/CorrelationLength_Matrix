# !/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# https://github.com/Dzsyang/CorrelationLength_Matrix
#
# MIT License
#
# Copyright (c) 2025 Zixiang Lin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import numpy as np
from imresize import *


def compute_spatial_correlation_mps(h_mat, scale=1, max_distance=None, distance_bins=100):
    """
    Compute the spatial correlation C_s(R) for a single scalar matrix h_mat using PyTorch on MPS.

    Args:
        h_mat (np.ndarray): A 2D scalar matrix representing the field.
        scale (int): Rescaling factor to reduce the matrix size for faster computation.
        max_distance (float): Maximum distance to calculate correlation for.
                              Defaults to half the diagonal length of the matrix.
        distance_bins (int): Number of bins for grouping distances.

    Returns:
        distances (np.ndarray): Array of distances (bin centers).
        C_s (np.ndarray): Spatial correlation C_s(R) as a function of distance.
    """
    # Rescale the matrix
    h_mat_rescaled = imresize(h_mat, output_shape=(h_mat.shape[0] // scale,
                                                   h_mat.shape[1] // scale))
    x_size, y_size = h_mat_rescaled.shape

    # Transfer to MPS (GPU with PyTorch)
    device = torch.device("mps")  # Use Metal Performance Shaders
    h_mat_gpu = torch.tensor(h_mat_rescaled, device=device, dtype=torch.float32)
    x_indices, y_indices = torch.meshgrid(
        torch.arange(x_size, device=device), torch.arange(y_size, device=device), indexing='ij'
    )
    positions = torch.stack([x_indices.ravel(), y_indices.ravel()], dim=1)  # Shape: (num_points, 2)

    # Compute pairwise distances on GPU
    deltas = positions[:, None, :] - positions[None, :, :]  # Pairwise differences
    distances = torch.sqrt(torch.sum(deltas ** 2, dim=-1))  # Euclidean distances

    # Compute distance bins
    if max_distance is None:
        max_distance = (x_size ** 2 + y_size ** 2) ** 0.5 / 2  # Half the diagonal length
    bin_edges = torch.linspace(0, max_distance, distance_bins + 1, device=device)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_centers = torch.cat([torch.tensor([0.0], device=device), bin_centers])  # Include R=0

    # Scalar products and distance-based grouping
    h_flat = h_mat_gpu.flatten()
    scalar_products = h_flat[:, None] * h_flat[None, :]  # Pairwise products

    C_s = torch.zeros(len(bin_centers), device=device)
    counts = torch.zeros(len(bin_centers), device=device)

    # Handle the zero-distance bin explicitly
    C_s[0] = torch.sum(h_flat ** 2)  # Self-correlation at zero distance
    counts[0] = h_flat.numel()  # Total number of elements (N)

    # Compute for remaining bins
    for i in range(1, len(bin_edges)):
        r_min = bin_edges[i - 1]
        r_max = bin_edges[i]
        mask = (distances >= r_min) & (distances < r_max)

        # Accumulate scalar products and counts for the bin
        C_s[i] = torch.sum(scalar_products[mask])
        counts[i] = torch.sum(mask)

    # Normalize correlation
    C_s /= counts
    C_s /= torch.mean(h_flat ** 2)  # Normalize by the average squared magnitude

    # Transfer results back to CPU
    return bin_centers.cpu().numpy(), C_s.cpu().numpy()