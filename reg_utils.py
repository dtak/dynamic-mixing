from __future__ import absolute_import
from __future__ import print_function

from autograd import grad
import autograd.numpy.random as npr
import autograd.numpy as np


def get_history_length(data):

    # Extract the history length if not provided
    idx_data = data['dataindex_set']
    data_count = idx_data.shape[0]
    length_set = np.ones((data_count, 1))
    length = 1
    for idx in range(data_count):
        if (idx > 0):
            if (idx_data[idx] == idx_data[idx - 1]):
                length += 1
            else:
                length = 1
        length_set[idx, 0] = length
    return length_set


def get_distances(data, all_pairs):

    # Extracts the quantile distances between patients
    history = get_history_length(data) 
    q0_set = np.percentile(all_pairs, 5, axis=1, keepdims=True)
    q1_set = np.percentile(all_pairs, 10, axis=1, keepdims = True)
    q2_set = np.percentile(all_pairs, 15, axis=1, keepdims=True)
    quantiles = np.hstack((history, q0_set, q1_set, q2_set))
    return quantiles


def toy_kernel(obs1, obs2, act1, act2):

    # Computes the kernel value based on history - smaller -> more similar
    length_diff = np.abs(obs1.shape[0] - obs2.shape[0]) > 0
    type_diff = 0
    if (obs1.shape[0] > 1) and (obs2.shape[0] > 1):
        type_diff = np.abs(obs1[0, 0] - obs2[0, 0]) > 0
    action_diff = np.abs(act1[-1, 0] - act2[-1, 0]) > 0
    my_value = length_diff + type_diff + action_diff 
    return my_value


def compute_all_cross_kernel(set1, set2):

    # Computes a set of pairs of kernel values between two sets of data
    data_train_count = set1['dataindex_set'].shape[0]
    data_test_count = set2['dataindex_set'].shape[0]

    all_cross_kernel_set = np.zeros((data_train_count, data_test_count))
    start1 = 0
    for k in range(data_train_count):
        start2 = 0
        if (k > 0):
            if (set1['dataindex_set'][k - 1] != set1['dataindex_set'][k]):
                start1 = k
        for j in range(data_test_count):
            if (set2['dataindex_set'][j - 1] != set2['dataindex_set'][j]):
                start2 = j
            kernel_value = toy_kernel(set1['obs_set'][start1:k + 1, :],
                                      set2['obs_set'][start2:j + 1, :],
                                      set1['action_set'][start1:k + 1, :],
                                      set2['action_set'][start2:j + 1, :])
            all_cross_kernel_set[k, j] = kernel_value
    return all_cross_kernel_set


def calculate_loss(predictions, actual):

    loss = np.sum(np.abs(actual - predictions))
    return loss


def get_next_observation(obsind, idx, dt):

    if (obsind >= dt['dataindex_set'][-1]):
        obs = dt['obs_set'][obsind]
    elif (dt['dataindex_set'][obsind + 1] != idx):
        obs = dt['obs_set'][obsind]
    else:
        obs = dt['obs_set'][obsind + 1]
    return obs


def kernel_regression(train_data, kernel_pairs, ltrewards):

    obs_data = train_data['obs_set']
    dim_obs = obs_data.shape[1]
    data_count = obs_data.shape[0]
    k = 4
    pred_obs = np.zeros((data_count, dim_obs))
    pred_ltreward = np.zeros(data_count)
    for i in range(data_count):
        pairs_vec = kernel_pairs[i, :]
        NN = np.argsort(pairs_vec)[0:k]
        kernel_sum = np.sum(pairs_vec[NN])
        if (kernel_sum != 0):
            pred_ltreward[i] = np.sum(pairs_vec[0:k] * ltrewards[NN]) / kernel_sum
        else:
            pred_ltreward[i] = 0
        obs_ind = npr.choice(NN, 1)[0]
        pred_obs[i, :] = get_next_observation(obs_ind, train_data['dataindex_set'][i], train_data)
    loss = calculate_loss(pred_ltreward, ltrewards)

    history_lengths = get_history_length(train_data)


    return pred_obs, history_lengths
