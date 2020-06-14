#!/usr/bin/env python3
"""
Implementation of
Hyun Jun Shin, Jong Ju Jeon, and Il Kyu Eom "Color filter array pattern identification using variance of color difference image," Journal of Electronic Imaging 26(4), 043015 (7 August 2017). https://doi.org/10.1117/1.JEI.26.4.043015

Usage:
    python3 shin_variance.py (-j JPEG_QUALITY) (-b BLOCK_SIZE) (-o OUT) images
    --jpeg, -j
        If specified, must be a number between 0 and 100. Images will be processed with JPEG compression before detection.
        If unspecified, no JPEG compression is applied

    --block, -b
        The image is split in blocks of the specified size before being studied. Default size: 32

    --out -o
        Path to the output file. Default: out_choi.npz

    images
        All following arguments should be paths to images to analyse.

Output:
    Output will be written on the specified file. The file is in a npz NumPy format, and contains one 2D array named after each input image. Each pixel in an array represent the confidence that the block at this location is forged, with 0 representing high-confidence and 1 low-confidence (or high-confidence that the block is authentic).
"""

import argparse
import io
import os
import sys

import numpy as np
import PIL
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import jpeg_compress


def decompose_in_grids(img):
    """
    Decompose the 4 possible CFA positions of an image.
    :param img: np.ndarray, shape (2Y, 2X, 3)
    :return grids: np.ndarray, shape (4, Y, X, 3)
    """
    return np.asarray(
        [img[::2, ::2], img[::2, 1::2], img[1::2, ::2], img[1::2, 1::2]])


def remove_background(grids):
    """
    Background removal as performed in the original article.
    :param grids: np.ndarray, shape (4, Y, X, 3)
    :return grids (background removed): same shape
    """
    grids = grids - .25 * (grids + np.roll(grids, 1, axis=1) + np.roll(
        grids, 1, axis=2) + np.roll(grids, (1, 1), axis=(1, 2)))
    return grids


def compute_df(grids):
    """
    Difference of colours, red-green (d) and blue-green (f).
    :param grids: np.ndarray, shape (4, Y, X, 3)
    :return d, f: two np.ndarray, both of shape (4, Y, X, 3)
    Input shape: (4, Y, X, 3)
    Output shape: ((4, Y, X), (4, Y, X))
    """
    d, f = grids[:, :, :, 0] - grids[:, :, :, 1], grids[:, :, :,
                                                        2] - grids[:, :, :, 1]
    return d, f


def grids_to_blocks(grids, block_size=32):
    """
    Split a grid into blocks
    :param grids: np.ndarray, shape (4, block_size*Y+t, block_size*X+t)
    :param block_size: default 32, block size
    :return blocks: np.ndarray, shape (Y, X, 4, block_size, block_size)
    """
    _, Y, X = grids.shape
    Y -= Y % block_size
    X -= X % block_size
    grids = grids[:, :Y, :X]
    n_Y, n_X = Y // block_size, X // block_size
    blocks = np.asarray(
        np.split(np.asarray(np.split(grids, n_X, axis=2)), n_Y, axis=2))
    return blocks


def blocks_to_variance(blocks):
    """
    Compute the spatial variance of each block.
    :param blocks: np.ndarray, shape (Y, X, 4, block_size, block_size)
    :return var: np.ndarray, shape (Y, X, 4)
    """
    var = np.var(blocks, axis=(-1, -2))
    return var


def determine_candidates(var_d, var_f):
    """
    Find the mot likely grid
    :param var_d, var_f: two lists, tuples or np.ndarray, both of shape (4,)
    :return grid, confidence: grid is 0 for RG/GB, 1 for GR/BG, 2 for GB/RG, 3 for BG/GR. Confidence ranges from 0 (very confident) to 1 (not confident).
    Returns:
    """
    v_d_RGGB_BGGR = np.abs(var_d[0] - var_d[3])
    v_d_GRBG_GBRG = np.abs(var_d[1] - var_d[2])
    v_f_RGGB_BGGR = np.abs(var_f[0] - var_f[3])
    v_f_GRBG_GBRG = np.abs(var_f[1] - var_f[2])
    v_RGGB_BGGR = v_d_RGGB_BGGR + v_f_RGGB_BGGR
    v_GRBG_GBRG = v_d_GRBG_GBRG + v_d_GRBG_GBRG
    if v_RGGB_BGGR > v_GRBG_GBRG:  # Candidates are RGGB and BGGR
        v_RGGB = var_d[3] + var_f[
            0]  # Where neither R/B nor G are original in that grid. A lower value thus means a higher probability
        v_BGGR = var_d[0] + var_f[3]
        ratio_green = v_GRBG_GBRG / v_RGGB_BGGR if v_RGGB_BGGR > 0 else None
        if v_RGGB < v_BGGR:
            ratio_rb = v_RGGB / v_BGGR if v_BGGR > 0 else None
            grid = 0
        else:
            ratio_rb = v_BGGR / v_RGGB if v_BGGR > 0 else None
            grid = 3
    else:  # Candidates are GRBG and GBRG
        v_GRBG = var_d[2] + var_f[1]
        v_GBRG = var_d[1] + var_f[2]
        ratio_green = v_RGGB_BGGR / v_GRBG_GBRG if v_GRBG_GBRG > 0 else None
        if v_GRBG > v_GBRG:
            ratio_rb = v_GBRG / v_GRBG if v_GRBG > 0 else None
            grid = 1
        else:
            ratio_rb = v_GRBG / v_GBRG if v_GBRG > 0 else None
            grid = 2
    if ratio_green is None:
        if ratio_rb is None:
            confidence = 1.
        else:
            confidence = ratio_rb
    else:
        if ratio_rb is None:
            confidence = ratio_green
        else:
            confidence = min(ratio_green, ratio_rb)
    return grid, confidence


def find_forgeries(img, block_size=32):
    """
    Given an image, find forged regions.
    :param img: np.ndarray, shape (Y, X, 3)
    :param block_size: int, default:32. Size of the blocks in which forgeries are sought
    :return forged_confdence: np.ndarray, shape (Y//block_size, X//block_size). Confidence that each block is forged, from 0 (very confident) to 1 (not confident that the block is forged, or confident that it is authentic).
    """
    block_size //= 2  # we will be working at half-resolution after decomposing into grids, block size must account for this
    grids = decompose_in_grids(img)
    grids = remove_background(grids)[:, 2:-2, 2:-2]
    d, f = compute_df(grids)
    global_var_d, global_var_f = np.var(d,
                                        axis=(-1, -2)), np.var(f,
                                                               axis=(-1, -2))
    global_grid, _ = determine_candidates(global_var_d, global_var_f)
    blocks_d, blocks_f = grids_to_blocks(d, block_size), grids_to_blocks(
        f, block_size)
    var_d, var_f = blocks_to_variance(blocks_d), blocks_to_variance(blocks_f)
    n_Y, n_X, _ = var_d.shape
    forged_confidence = np.ones((n_Y, n_X))
    for y in range(n_Y):
        for x in range(n_X):
            grid, confidence = determine_candidates(var_d[y, x], var_f[y, x])
            if grid != global_grid:
                forged_confidence[y, x] = confidence
    return forged_confidence


def get_parser():
    parser = argparse.ArgumentParser(
        description=
        """Detect forgeries through CFA estimation using the variance of colour difference method. For more details see
    Hyun Jun Shin, Jong Ju Jeon, and Il Kyu Eom "Color filter array pattern identification using variance of color difference image," Journal of Electronic Imaging 26(4), 043015 (7 August 2017). https://doi.org/10.1117/1.JEI.26.4.043015"""
    )
    parser.add_argument(
        "-j",
        "--jpeg",
        type=int,
        default=None,
        help=
        "JPEG compression quality. Default: no compression is done before analysis."
    )
    parser.add_argument("-b",
                        "--block-size",
                        type=int,
                        default=32,
                        help="Block size. Default: 32.")
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default="out_shin.png",
        help="Path to output detected forgeries. Default: out_shin.png")
    parser.add_argument("input", type=str, help="Images to analyse.")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    out = args.out
    block_size = args.block_size
    quality = args.jpeg
    image_name = args.input
    confidences = {}
    img = plt.imread(image_name)
    Y_o, X_o, C = img.shape
    img = img[:Y_o, :X_o, :3]
    if quality is not None:
        img = jpeg_compress(img, quality)
    forged_confidence = find_forgeries(img, block_size)
    error_map = 1 - forged_confidence  # highest values (white) correspond to suspected forgeries
    # Resample the output to match the original image size
    error_map = np.repeat(np.repeat(error_map, block_size, axis=0),
                          block_size,
                          axis=1)
    output = np.zeros((Y_o, X_o))
    output[:error_map.shape[0], :error_map.shape[1]] = error_map
    plt.imsave(out, output)
