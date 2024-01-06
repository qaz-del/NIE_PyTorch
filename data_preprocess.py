import argparse
import os
import h5py

import numpy
import cupy


import utility
import process


if __name__=="__main__":
    # mode argument
    parser = argparse.ArgumentParser()

    parser.add_argument('--dim', type=int, default=256, help='ELM weight size')
    parser.add_argument('--margin', type=float, default=0.4, help='Radius of sampling sphere')
    parser.add_argument('--normals', type=bool, default=True, help='Use normals for training')
    parser.add_argument('--sample_size', type=int, default=256,
                        help='Number of sampling points for distance field calculation')
    args = parser.parse_args()# mode argument

    DATA_DIR = './data/modelnet40_ply_hdf5_2048'  # your directory
    SAVE_DIR = './data/weight_data'  # your directory

    # check if data preparation is necessary
    data_list_train = utility.makeList(os.path.join(DATA_DIR, 'train_files.txt'))
    data_list_test = utility.makeList(os.path.join(DATA_DIR, 'test_files.txt'))

    # prepare samplings points
    basis_name = 'rands' + str(args.dim) + '.h5'
    point_name = 'points' + str(args.sample_size) + 'dim' + str(args.dim) + 'margin' + str(args.margin) + '.h5'
    utility.prepSamplePoints(args.sample_size, os.path.join(SAVE_DIR, point_name), os.path.join(SAVE_DIR, basis_name),
                             args.dim, args.margin)

    # prepare train data
    for num in range(len(data_list_train)):
        original_name = data_list_train[num]
        weight_name = data_list_train[num] + 'margin' + str(args.margin) + 'dim' + str(args.dim) + '.h5'

        # calculate distances
        print('---- train data ' + str(num) + ' ----')
        # convert them into ELM weights
        if not os.path.isfile(os.path.join(SAVE_DIR, weight_name)):
            process.cal_dis(os.path.join(SAVE_DIR, point_name), os.path.join(DATA_DIR, original_name), os.path.join(SAVE_DIR,weight_name), args.dim)
        print('---- train data ' + str(num) + 'finished ----')

    # prepare test data
    for num in range(len(data_list_train)):
        original_name = data_list_test[num]
        weight_name = data_list_test[num] + 'margin' + str(args.margin) + 'dim' + str(args.dim) + '.h5'

        # calculate distances
        print('---- test data ' + str(num) + ' ----')
        # convert them into ELM weights
        if not os.path.isfile(os.path.join(SAVE_DIR, weight_name)):
            process.cal_dis(os.path.join(SAVE_DIR, point_name), os.path.join(DATA_DIR, original_name), os.path.join(SAVE_DIR,weight_name), args.dim)
        print('---- test data ' + str(num) + 'finished ----')