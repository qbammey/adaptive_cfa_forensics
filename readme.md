Requirements:
python >= 3.6

PIL
numpy
tqdm
matplotlib
scikit-learn

In addition, the following files require pytorch with CUDA enabled:
detect_forgeries_interactive.py
detect_forgeries_multiple.py
train_model.py


# Usage:
## Train or retrain a network
train_model.py [-h] [-m MODEL] [-j JPEG] [-b BLOCK_SIZE] [-o OUT]
                      [-l LEARNING_RATE] [-a EPOCHS_AUXILIARY]
                      [-B EPOCHS_BLOCKWISE] [-s BATCH_SIZE]
                      input [input ...]
To use a pretrained network and retrain it on data, specify the pretrained model with -m.
All images are kept in GPU memory at the same time. As a consequence, training on a large database require more GPU memory.

## Detect forgeries on a single image with the proposed method, and plot the results interactively:
detect_forgeries_interactive.py [-h] [-m MODEL] [-j JPEG]
                                       [-b BLOCK_SIZE]
                                       input
The model can be specified with -m. By default, uses the pretrained model (not retrained on the database).

## Detect forgeries on multiple images with the proposed method, and store the results in one .npz file:
detect_forgeries_multiple.py [-h] [-m MODEL] [-j JPEG] [-b BLOCK_SIZE]
                                    [-o OUT]
                                    input [input ...]
The model can be specified with -m. By default, uses the pretrained model (not retrained on the database).

## Detect forgeries on multiple images with the intermediate values method, and store the results in one .npz file:
choi_intermediate_values.py [-h] [-j JPEG] [-b BLOCK_SIZE] [-o OUT]
                                   input [input ...]
This is an implementation of the method described in
Choi, C., Choi, J., & Lee, H. (2011). CFA pattern identification of digital cameras using intermediate value counting. MM&Sec'11.

## Detect forgeries on multiple images with the variance of colour difference method, and store the results in one .npz file:
shin_variance.py [-h] [-j JPEG] [-b BLOCK_SIZE] [-o OUT]
                        input [input ...]
This is an implementation of the method described in
Hyun Jun Shin, Jong Ju Jeon, and Il Kyu Eom "Color filter array pattern identification using variance of color difference image," Journal of Electronic Imaging 26(4), 043015 (7 August 2017). https://doi.org/10.1117/1.JEI.26.4.043015


# Provided content
The ground truth of the database is provided in the ground_truth/ folder, both with all forgeries marked as such and with only misaligned forgeries marked as such.

The pretrained model, as well as the adapted models for both the uncompressed and JPEG compressed with quality 95, are available in the models/ folder.
