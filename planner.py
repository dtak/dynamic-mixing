from __future__ import absolute_import
from __future__ import print_function

from autograd.scipy.misc import logsumexp
import autograd.numpy.random as npr
import autograd.numpy as np
import mixture_network as mn
import rl_utils as rl

def calculate_reward(dt, idx, obs, a):

    reward = 0
    if (idx == 0 or idx % 4 == 0):
        # first step
        if (a == 0):
            reward = -10
        else:
            reward = 5
    elif (idx % 4 == 1 & obs[0] > 0):
        if (a == 0):
            reward = 0
        elif (a == 1):
            reward = 5
        else:
            reward = -10
    elif (idx % 4 == 2 & obs[0] < 0):
        if (a == 0):
            reward = 0
        elif (a == 1):
            reward = -10
        else:
            reward == 5
    elif (idx % 4 == 3):
        reward = 0

    return reward


def get_action_value(curr_lbelief, ltrans, test_dt, test_fcpt, quantiles, kernel_pred, init_pred, emission_mu_set, emission_std_set, a, idx, network, x, init, model_path, depth, args):

    num_states = args['num_states']
    num_actions = args['num_actions']
    num_branches = args['branch_count']

    val = 0
    if (depth == 0):
        pomdp_obs = init_pred[idx]
        kernel_obs = kernel_pred[idx, :]
        inputs = np.hstack((kernel_obs, pomdp_obs, 1, quantiles[idx, :]))
        inputs = np.reshape(inputs, (1, inputs.shape[0]))
        obs = mn.predict(network, inputs, x, init, model_path)
        obs = obs.flatten()

        # Calculate the corresponding reward
        val = calculate_reward(test_dt, idx, obs, a)

    elif (depth > 0):
        branch_val = np.zeros(num_branches)
        for branch in range(num_branches):
            curr_lbelief = np.reshape(curr_lbelief, (1, num_states))

            # sample an observation
            pomdp_obs = rl.sample_observations(curr_lbelief, emission_mu_set, emission_std_set, test_dt, args)[0]
            
            # get kernel observation
            kernel_obs = kernel_pred[idx, :]
            
            history = test_dt['dataindex_set'][idx] % 4
            inputs = np.hstack((kernel_obs, pomdp_obs, history, quantiles[idx, :]))
            inputs = np.reshape(inputs, (1, inputs.shape[0]))
            
            obs = mn.predict(network, inputs, x, init, model_path)
            obs = np.reshape(obs, (1, args['dim_obs']))
            log_prob_x = rl.calc_log_proba_arr_for_x(obs, emission_mu_set, emission_std_set, num_states, a)
            obs = obs.flatten()
            val = calculate_reward(test_dt, idx, obs, a)

            # update belief according to a and log_prob_x
            curr_lbelief = rl.single_update_belief_log_probas(curr_lbelief, log_prob_x, ltrans, a)
            
            # calculate the forward_values
            forward_values = np.zeros(num_actions)

            for forward_a in range(0, num_actions):
                if (idx + 1 > test_dt['dataindex_set'].shape[0] - 1):
                    break
                elif (test_dt['dataindex_set'][idx + 1] != test_dt['dataindex_set'][idx]):  
                    break
                elif (test_dt['dataindex_set'][idx + 1] == test_dt['dataindex_set'][idx]):
                    forward_values[forward_a] = get_action_value(curr_lbelief, ltrans, test_dt, test_fcpt, quantiles, kernel_pred, init_pred, emission_mu_set, emission_std_set, forward_a, idx + 1, network, x, init, model_path, depth - 1, args) 

            branch_val[branch] = np.max(forward_values)

        val = val + args['discount_rate'] * np.sum(branch_val)

    return val


def run_mixed_planner(lbeliefs, ltrans, test_dt, test_fcpt, quantiles, kernel_obs_pred, init_obs_pred, emission_mu_set, emission_std_set, network, x, init, model_path, args):

    num_states = args['num_states']
    num_actions = args['num_actions']
    dim_obs = args['dim_obs']
    depth = args['depth']

    optimal_action_list = []
    policy = np.zeros((lbeliefs.shape[0], num_actions))

    for idx in xrange(12): # should be lbeliefs.shape[0]
        curr_lbelief = lbeliefs[idx, :]
        val_set = np.zeros((num_actions))

        # get forward value for each action by sampling
        for a in range(0, num_actions):
            val_set[a] = get_action_value(curr_lbelief, ltrans, test_dt, test_fcpt, quantiles, kernel_obs_pred, init_obs_pred, emission_mu_set, emission_std_set, a, idx, network, x, init, model_path, depth, args)

        optimal_action_list.append(np.argmax(val_set))

    print(optimal_action_list[0:12])

    return None, optimal_action_list
