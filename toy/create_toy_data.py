
'''
Creating a toy example to test the kernelised dynamical system code.
We have a system that evolves deterministically through four states:

state 1 -> state 2/state 3 -> state 4

state 4 is an absorbing state

A time-series belongs to one of two types, A or B. Both A and B have rare and common variants. There are three actions available at each stage, a, b, and c.  The rewards are as follows:

state 1, a = -10 , b = 5 , c = 0
state 2, a = 5 , b = -10 , c = 0
state 3, a = 5 if A, -10 if B; b = 5 if B, -10 if A; c = 0
state 3 persists for one step, so everyone follows 0,1/2,3

We observe the rewards (as observations, just made it easy for me)
as well as an observation that depends only on the time-series' type (the obs mean which determines whether we have A or B):

obs dim #1: either 0,1 or 3 if type A, 2 or 4,5 if type B.  Designed
so that in the training set, we see mostly 0,1 and 4,5; if a kernel
maps to the nearest it will fail on the observations because they
will be incorrectly mapped.

The output is the sequence of observations, actions, and rewards
stored in the fencepost way
'''

import numpy as np
import numpy.random as npr
import cPickle as pkl
import os


# create the data set
def create_toy_data(train_data, test_data):
    sequence_count = 250
    sequence_length = 4  # M = sequence_count * sequence_length

    # where to store things
    state_set = np.zeros((0, 1))
    obs_set = np.zeros((0, 1))
    action_set = np.zeros((0, 1))
    reward_set = np.zeros((0, 1))
    dataindex_set = np.zeros((0, 1))
    optimal_set = np.zeros((0, 1))

    # pre-decide the sequence types
    rare_count = 20
    # initialize all sequences to the same type eg A
    sequence_type = np.zeros(int(sequence_count)) + 1.0
    # make all second half of the sequences another type B
    sequence_type[int(sequence_count + 0.0) // int(2.0):] = -1.0

    if test_data is True:
        skip = 5
    else:
        skip = 1

    my_start = 0

    for rare_index in range(rare_count):
        my_end = my_start + skip
        sequence_type[int(my_start):int(my_end)] = rare_index + 2
        sequence_type[(-1 * int(my_end + 1)):(-1 * int(my_start + 1))] = -1 * int(rare_index + 2)
        my_start = my_end
    min_obs = np.min(sequence_type)

    obs_mean_list = []

    fenceposts = []
    # set values (the three-stage part is hard-coded!)
    count = 0
    fencecount = 0
    for sequence_iter in range(int(sequence_count)):
        count = count + 1
        obs_mean = sequence_type[sequence_iter]
        # Fill in the first dimension of the observation; since it is
        # dependent only on type, it does not change over time
        obs_sequence = obs_mean - min_obs + np.zeros((int(sequence_length), 1))
        # print(obs_sequence)

        # Fill in the actions, for now chosen randomly
        action_sequence = npr.choice(3, (int(sequence_length), 1))

        # Fill the rewards
        state_sequence = np.zeros((int(sequence_length), 1)) + 0.0
        reward_sequence = np.zeros((int(sequence_length), 1)) + 0.0
        optimal_sequence = np.zeros((int(sequence_length), 1))

        if (sequence_iter > 0):
            fenceposts.append(fencecount - 1)
        else:
            fenceposts.append(fencecount)

        for reward_iter in range(sequence_length):
            fencecount = fencecount + 1
            obs_mean_list.append(obs_mean)
            if reward_iter == 0:
                # at iteration 0, either action 2 or action 1 is optimal
                if action_sequence[int(reward_iter), 0] == 0:
                    reward_sequence[int(reward_iter), 0] = -10.0
                if action_sequence[int(reward_iter), 0] == 1:
                    reward_sequence[int(reward_iter), 0] = 5.0
                if action_sequence[int(reward_iter), 0] == 2:
                    reward_sequence[int(reward_iter), 0] = 5.0
                optimal_sequence[int(reward_iter), 0] = np.random.randint(1, 3)

            if reward_iter == 1 and obs_mean > 0:
                # at iteration 1, if type A then action 1 is optimal
                state_sequence[int(reward_iter), 0] = 1
                if action_sequence[int(reward_iter), 0] == 0:
                    reward_sequence[int(reward_iter), 0] = 0.0
                if action_sequence[int(reward_iter), 0] == 1:
                    reward_sequence[int(reward_iter), 0] = 5.0
                if action_sequence[int(reward_iter), 0] == 2:
                    reward_sequence[int(reward_iter), 0] = -10.0
                optimal_sequence[int(reward_iter), 0] = 1

            if reward_iter == 1 and obs_mean < 0:
                # at iteration 1, if type B then action 2 is optimal
                state_sequence[int(reward_iter), 0] = 2
                if action_sequence[int(reward_iter), 0] == 0:
                    reward_sequence[int(reward_iter), 0] = 0.0
                if action_sequence[int(reward_iter), 0] == 1:
                    reward_sequence[int(reward_iter), 0] = -10.0
                if action_sequence[int(reward_iter), 0] == 2:
                    reward_sequence[int(reward_iter), 0] = 5.0
                optimal_sequence[int(reward_iter), 0] = 2
            if reward_iter > 1:
                # at iteration 2 or more, there is no reward - absorbing state
                state_sequence[int(reward_iter), 0] = 3
                optimal_sequence[int(reward_iter), 0] = np.random.randint(0, 3)

            # So overall the optimal sequence should be 1/2, 1, .... if type A
            # OR 1/2, 2, .... if type B

        # Store
        state_set = np.vstack((state_set, state_sequence))  # M x 1
        obs_set = np.vstack((obs_set, obs_sequence))  # M x 1
        action_set = np.vstack((action_set, action_sequence))  # M x 1
        reward_set = np.vstack((reward_set, reward_sequence))
        dataindex_set = np.vstack(
            (dataindex_set, np.zeros((sequence_length, 1)) + sequence_iter))
        optimal_set = np.vstack((optimal_set, optimal_sequence))

    fenceposts.append(fencecount - 1)
    # Attach the rewards to the observations as additional
    # observations (multiple times, to make the POMDP "stick" to
    # explaining that signal and not the other obs signal (hopefully
    # something that the kernel will pick up)

    reward_value_set, reward_obs_set = np.unique(
        reward_set, return_inverse=True)
    reward_obs_set = np.reshape(reward_obs_set, (reward_obs_set.shape[0], 1))
    obs_set = np.hstack((obs_set, reward_obs_set, reward_obs_set,
                         reward_obs_set, reward_obs_set, reward_obs_set, reward_obs_set))
    optimal_set = optimal_set.flatten()

    data_set = {'state_set': state_set,
                'action_set': action_set,
                'reward_set': reward_set, 
                'obs_set': obs_set,
                'dataindex_set': dataindex_set,
                'obs_mean': obs_mean_list,
                'optimal_set': optimal_set,
                }

    # # write the dataset dictionary to file
    if (train_data is True):
        print("Creating Training Data")
        f = open("train_fcpt.p", "wb")
        pkl.dump(data_set, f)
        f.close()
        f = open("train_fcpt.p", "wb")
        pkl.dump(fenceposts, f)
        f.close()

    elif (test_data is True):
        print("Creating Test Data")
        f = open("test_data.p", "wb")
        pkl.dump(data_set, f)
        f.close()
        f = open("test_fcpt.p", "wb")
        pkl.dump(fenceposts, f)
        f.close()

    else:
        print("Creating Validation Data")
        f = open("val_data.p", "wb")
        pkl.dump(data_set, f)
        f.close()
        f = open("val_fcpt.p", "wb")
        pkl.dump(fenceposts, f)
        f.close()

def get_longterm_rewards(ids, rewards):

    totals = np.zeros(rewards.shape[0])
    i = 0
    while (i < rewards.shape[0] - 1):
        totals[i] = rewards[i]
        ind = i
        discount = 0.98
        while (ids[ind] == ids[ind + 1]):
            totals[i] = totals[i] + np.power(discount, i) * rewards[ind + 1]
            ind = ind + 1
            if (ind + 1 == rewards.shape[0]):
                break
        i = i + 1
    totals[rewards.shape[0] - 1] = rewards[rewards.shape[0] - 1]
    return totals


if __name__ == "__main__":

    # Load the data and calculate the long term rewards
    train_data = pkl.load(open('train_data.p', "rb"))
    test_data = pkl.load(open('test_data.p', "rb"))
    val_data = pkl.load(open('val_data.p', "rb"))

    train_ids = train_data['dataindex_set']
    test_ids = test_data['dataindex_set']
    val_ids = val_data['dataindex_set']

    train_rewards = train_data['reward_set']
    test_rewards = test_data['reward_set']
    val_rewards = val_data['reward_set']

    # Calculate the long term rewards for each set and save as ltr files
    train_ltr = get_longterm_rewards(train_ids, train_rewards)
    test_ltr = get_longterm_rewards(test_ids, test_rewards)
    val_ltr = get_longterm_rewards(val_ids, val_rewards)

    f = open("train_ltr.p", "wb")
    pkl.dump(train_ltr, f)
    f.close()

    f = open("test_ltr.p", "wb")
    pkl.dump(test_ltr, f)
    f.close()

    f = open("val_ltr.p", "wb")
    pkl.dump(val_ltr, f)
    f.close()
