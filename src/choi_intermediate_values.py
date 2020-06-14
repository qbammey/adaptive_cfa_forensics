#!/usr/bin/env python3
"""
Implementation of
Choi, C., Choi, J., & Lee, H. (2011). CFA pattern identification of digital cameras using intermediate value counting. MM&Sec '11.


Usage:
    python3 choi_intermediate_values.py (-j JPEG_QUALITY) (-b BLOCK_SIZE) (-o OUT) images
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


def is_intermediate(a):
    """
    Returns a boolean array where each pixel is True if it is an intermediate value between its four direct neighbours, ie. if at least one of its neighbours has a lower value than it, and at least one has an higher value.
    :param a: np.ndarray, shape (Y, X, C) or (Y, X)
    :return intermediate: np.bool np.ndarray shape (Y-2, X-2, C) or (Y-4, X-4) depending on the shape of a. If several channels are provided, they are treated separately
    """
    has_lower = (a >= np.roll(a, 1, axis=0)) + (a >= np.roll(
        a, -1, axis=0)) + (a >= np.roll(a, 1, axis=1)) + (a >= np.roll(
            a, -1, axis=1))
    has_higher = (a <= np.roll(a, 1, axis=0)) + (a <= np.roll(
        a, -1, axis=0)) + (a <= np.roll(a, 1, axis=1)) + (a <= np.roll(
            a, -1, axis=1))
    is_intermediate_value = has_lower * has_higher
    return is_intermediate_value


def choose_green(a):
    """
    Given a boolean array of shape (Y, X), counts the number of ones in XGGX and GXXG. Returns the best grid, as well as the ratio between the number of intermediate value in the best and the second grid. A ratio closer to 0 means a higher confidence in the result.
    :param a: np.ndarray, shape (Y, X)
    :return best_grid, confidence
    """
    n_xggx = np.count_nonzero(a[::2, 1::2]) + np.count_nonzero(a[1::2, ::2])
    n_gxxg = np.count_nonzero(a[::2, ::2]) + np.count_nonzero(a[1::2, 1::2])
    if n_xggx < n_gxxg:  # more intermediate values in GXXG: XGGX is the best grid
        return 0, n_xggx / n_gxxg
    else:
        return 1, n_gxxg / n_xggx


def compare_two(a, b):
    """
    Given two boolean arrays, compare the two of them to find which one has the least non-zero values.
    :param a, b: np.ndarray, both of the same shape (Y, X)
    :return best_grid, confidence
    """
    n_a = np.count_nonzero(a)
    n_b = np.count_nonzero(b)
    if n_a < n_b:
        return 0, n_a / n_b
    else:
        try:
            return 1, n_b / n_a
        except ZeroDivisionError:
            return 0, 1.


def choose_grid(intermediate):
    """
    Given a boolean array of shape (Y, X, 3), indicating each intermediate value, selects the best of the four possible grid positions.
    :param intermediate: np.ndarray, shape (Y, X, 3)
    :return best_grid, confidence: best grid is 0 for RG/GB, 1 for GR/BG, 2 for GB/RG, 3 for BG/GR. Confidence ranges from 0 (very confident) to 1 (not confident).
    """
    green, confidence_g = choose_green(intermediate[:, :, 1])
    if green == 0:  # Possible grids are RGGB, BGGR
        r_RGGB = img[::2, ::2, 0]
        r_BGGR = img[1::2, 1::2, 0]
        b_RGGB = img[1::2, 1::2, 2]
        b_BGGR = img[::2, ::2, 2]
        best_r, confidence_r = compare_two(r_RGGB, r_BGGR)
        best_b, confidence_b = compare_two(b_RGGB, b_BGGR)
        if confidence_r < confidence_b:
            return (0 if best_r == 0 else 3), max(confidence_g, confidence_r)
        else:
            return (0 if best_b == 0 else 3), max(confidence_g, confidence_b)
    else:  # Possible grids are GRBG, GBRG
        r_GRBG = img[::2, 1::2, 0]
        r_GBRG = img[1::2, ::2, 0]
        b_GRBG = img[1::2, ::2, 2]
        b_GBRG = img[::2, 1::2, 2]
        best_r, confidence_r = compare_two(r_GRBG, r_GBRG)
        best_b, confidence_b = compare_two(b_GRBG, b_GBRG)
        if confidence_r < confidence_b:
            return (0 if best_r == 1 else 2), min(confidence_g, confidence_r)
        else:
            return (0 if best_b == 2 else 2), min(confidence_g, confidence_b)


def find_forgeries(img, block_size=32):
    """
    Given an image, find forged regions.
    :param img: np.ndarray, shape (Y, X, 3)
    :param block_size: int, default:32. Size of the blocks in which forgeries are sought
    :return forged_confdence: np.ndarray, shape (Y//block_size, X//block_size). Confidence that each block is forged, from 0 (very confident) to 1 (not confident that the block is forged, or confident that it is authentic).
    """
    Y, X, C = img.shape
    Y -= Y % 2
    X -= X % 2
    C = 3
    img = img[:Y, :X, :C]
    intermediate = is_intermediate(img)[4:-4, 4:-4]
    Y, X, C = intermediate.shape
    Y -= Y % block_size
    X -= X % block_size
    intermediate = intermediate[:Y, :X]
    global_grid, _ = choose_grid(intermediate)
    n_Y, n_X = Y // block_size, X // block_size  # number of blocks in each dimension
    blocks = np.asarray(
        np.split(np.asarray(np.split(intermediate, n_X, axis=1)), n_Y,
                 axis=1))  # shape (n_Y, n_X, block_size, block_size, 3)
    forged_confidence = np.ones((n_Y, n_X))
    for y in range(n_Y):
        for x in range(n_X):
            grid, confidence = choose_grid(blocks[y, x])
            if grid != global_grid:
                forged_confidence[y, x] = confidence
    return forged_confidence


def get_parser():
    parser = argparse.ArgumentParser(
        description=
        """Detect forgeries through CFA estimation using the intermediate values method. For more details see
    Choi, C., Choi, J., & Lee, H. (2011). CFA pattern identification of digital cameras using intermediate value counting. MM&Sec '11."""
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
        default="out_choi.png",
        help="Path to output detected forgeries. Default: out_choi.png")
    parser.add_argument("input", type=str, help="Image to analyse.")
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
    img = img[:Y_o - Y_o % 2, :X_o - X_o % 2, :3]
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
