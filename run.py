from __future__ import absolute_import
from __future__ import print_function
from autograd import grad

import os
import cPickle as pkl
import argparse
import numpy as np
import reg_utils as rg
import rl_utils as rl
import mixture_network as mn
import planner as pl


# import weights_util as util
# import pomdp_functions as pomdp


def get_parser():

    parser = argparse.ArgumentParser(description='run KDM method')
    parser.add_argument('--dataset', help='path to data files',
                        required=False, default='/public/toy/', type=str)
    parser.add_argument('--has_fence', help='if dataset is sequence based',
                        required=False, default=1, type=int)
    parser.add_argument('--out_dir', help='output directory',
                        required=False, default='/outputs/', type=str)
    parser.add_argument('--num_states', help='number of POMDP hidden states',
                        default=4, required=False, type=int)
    parser.add_argument('--num_actions', help='number of actions',
                        default=3, required=False, type=int)
    parser.add_argument('--num_rewards', help='number of rewards',
                        default=3, required=False, type=int)
    parser.add_argument('--dim_obs', help='dim of obs (if discrete)',
                        default=7, required=False, type=int)
    parser.add_argument('--branch_count', help='number of branches for FS',
                        default=1, required=False, type=int)
    parser.add_argument('--depth', help='search depth for FS', default=3, required=False, type=int)
    parser.add_argument('--kmeans', help='use k-means initialization',
                        default=True, required=False, type=bool)
    parser.add_argument('--discount_rate', help='discount factor',
                        default=0.98, required=False, type=float)
    parser.add_argument('--kernel', help='specify kernel function path', required=False, type=str)
    parser.add_argument('--seed', help='set random seed',
                        default=42, required=False, type=int)
    parser.add_argument('--flatten_weights', help='use flattened weights',
                        default=False, required=False, type=bool)
    return parser


def get_dataset_basename(dataset):

    # Get the data set name and path
    dataset_parts = dataset.split('/')[::-1]
    for i in range(len(dataset_parts)):
        if len(dataset_parts[i]) > 0:
            return dataset_parts[i]


def load_data(path, has_fence=True):

    # Load the data dictionaries with the corresponding fence-posts
    print("Loading data...")
    train_data = pkl.load(open(os.path.join(path, 'train_data.p'), "rb"))
    test_data = pkl.load(open(os.path.join(path, 'test_data.p'), "rb"))
    val_data = pkl.load(open(os.path.join(path, 'val_data.p'), "rb"))

    # Also load the data for the kernel based approach with long-term rewards
    train_ltrewards = pkl.load(open(os.path.join(path, 'train_ltr.p'), "rb"))
    test_ltrewards = pkl.load(open(os.path.join(path, 'test_ltr.p'), "rb"))
    val_ltrewards = pkl.load(open(os.path.join(path, 'val_ltr.p'), "rb"))

    if has_fence:

        # Get the train, test and val fence-posts.
        train_fcpt = pkl.load(open(os.path.join(path, 'train_fcpt.p'), "rb"))
        test_fcpt = pkl.load(open(os.path.join(path, 'test_fcpt.p'), "rb"))
        val_fcpt = pkl.load(open(os.path.join(path, 'val_fcpt.p'), "rb"))

        return train_fcpt, test_fcpt, val_fcpt, train_data, test_data, val_data, train_ltrewards, test_ltrewards, val_ltrewards
    else:

        return train_data, test_data, val_data, train_ltrewards, test_ltrewards, val_ltrewards


def save_kernel_pairs(pairs, train):

    # Save a set of kernel pairs to file
    if (train is True):
        f = open("outputs/all_kernel_pairs_train.p", "wb")
    else:
        f = open("outputs/all_kernel_pairs_test.p", "wb")
    pkl.dump(pairs, f)
    f.close()


def get_targets(train_data):

    # Get the one step ahead observations for each time step
    obs_data = train_data['obs_set']
    ids = train_data['dataindex_set']

    target_set = np.zeros(obs_data.shape)
    for k in range(obs_data.shape[0] - 1):
        if (ids[k] == ids[k + 1]):
            target_set[k, :] = obs_data[k + 1, :]
        else:
            target_set[k, :] = obs_data[k, :]

    return target_set


def main():
    parser = get_parser()
    args = vars(parser.parse_args())

    if (args['has_fence']):
        train_fcpt, test_fcpt, val_fcpt, train_data, test_data, val_data, train_ltrewards, test_ltrewards, val_ltrewards = load_data(path=args['dataset'], has_fence=bool(args['has_fence']))
    else:
        train_data, test_data, val_data, train_ltrewards, test_ltrewards, val_ltrewards = load_data(path=args['dataset'], has_fence=bool(args['has_fence']))

    print(train_data['action_set'].shape)

    # compute a set of kernel pairs over data set
    kernel_pairs = rg.compute_all_cross_kernel(train_data, train_data)

    # save the kernel pairs
    save_kernel_pairs(kernel_pairs, train=True)

    # get the quantile distances from the set of kernel pairs
    quantiles = rg.get_distances(train_data, kernel_pairs)

    # use the kernel to train a regression for predicting the next observation
    obs_pred, history_lengths = rg.kernel_regression(train_data, kernel_pairs, train_ltrewards)

    # learn a POMDP on the train data
    lbelief_set, pomdp_obs_pred, emission_mu, emission_std, ltrans = rl.run_pomdp(train_data, train_fcpt, args)

    # create inputs for mixing network
    inputs = np.hstack((obs_pred, pomdp_obs_pred, history_lengths, quantiles))

    # train a mixing network to predict the next observation; validate against true observation
    targets = get_targets(train_data)
    network, x, init, model_path = mn.train_mixture_network(inputs, targets, inputs.shape[1], targets.shape[1])

    # calculate kernel pairs for test data set
    cross_kernel_pairs = rg.compute_all_cross_kernel(test_data, train_data)

    # save the kernel pairs
    save_kernel_pairs(cross_kernel_pairs, train=False)

    # get the quantile distances for the test data from kernel pairs
    test_quantiles = rg.get_distances(test_data, cross_kernel_pairs)

    # use the kernel to predict the next observation
    test_obs_pred, test_history_lengths = rg.kernel_regression(test_data, cross_kernel_pairs, test_ltrewards)

    # run pomdp on test data and extract belief states
    lbelief_test_set, test_init_obs, emission_mu, emission_std, ltrans = rl.run_pomdp(test_data, test_fcpt, args)

    # run planning
    policy_list = pl.run_mixed_planner(lbelief_test_set, ltrans, test_data, test_fcpt, test_quantiles, test_obs_pred, test_init_obs, emission_mu, emission_std, network, x, init, model_path, args)

    return policy_list


if __name__ == "__main__":

    policy = main()

