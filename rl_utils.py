from __future__ import absolute_import
from __future__ import print_function

from autograd.scipy.misc import logsumexp
import autograd.numpy.random as npr
import autograd.numpy as np


def sigmoid(x):
    return 0.5 * (np.tanh(x) + 1)


def reverse_sigmoid(x):
    return safe_log(1 - x) - safe_log(x)


def safe_log(x, minval=1e-100):
    ''' assuming upper bound is 1 '''
    return np.log(np.maximum(x, minval))


def initialise_emission_weights(data, args):

    x_ND = data['obs_set']
    K = args['num_states']
    A = args['num_actions']
    D = x_ND.shape[1]
    seed = 0
    prng = np.random.RandomState(int(seed))
    min_mu = -1.0
    max_mu = 1.0
    min_stddev = 0.1
    max_stddev = 1.0

    mu_ADK = []
    stddev_ADK = []
    for a in range(A):
        mu_DK = prng.uniform(low=min_mu, high=max_mu, size=(D, K))
        stddev_DK = prng.uniform(
            low=min_stddev,
            high=max_stddev,
            size=(D, K))
        mu_ADK.append(mu_DK)
        stddev_ADK.append(stddev_DK)
    return mu_ADK, stddev_ADK


def initialise_emission_weights_KMeans(data, args):

    from sklearn.cluster import KMeans

    x_ND = data['obs_set']
    actions = data['action_set']
    D = x_ND.shape[1]
    K = args['num_states']
    A = args['num_actions']
    seed = 0

    mu_ADK = []
    stddev_ADK = []
    mu_DK, stddev_DK = np.zeros((D, K)), np.zeros((D, K))
    kmeans = KMeans(n_clusters=K, random_state=seed).fit((x_ND))

    for a in range(A):
        for k in range(K):
            x_ND_k = x_ND[np.where((kmeans.labels_ == k) & (actions == a))[0]]
            mu_DK[:, k] = np.mean(x_ND_k, axis=0)
            stddev_DK[:, k] = np.maximum(np.std(x_ND_k, axis=0), .1)
        mu_ADK.append(mu_DK)
        stddev_ADK.append(stddev_DK)
    return mu_ADK, stddev_ADK


def initialise_model_weights(data, args):

    K = args['num_states']
    A = args['num_actions']
    seed = 42

    prng = npr.RandomState(int(seed))
    init_start_K = prng.uniform(low=1.0, high=1.1, size=K)
    init_start_K /= init_start_K.sum()
    init_trans_KAK = prng.uniform(low=1.0, high=1.1, size = (K, A, K))
    init_trans_KAK /= init_trans_KAK.sum(axis=-1, keepdims=True)
    if (args['kmeans'] is False):
        emission_mu, emission_sigma = initialise_emission_weights(data, args)
    else:
        emission_mu, emission_sigma = initialise_emission_weights_KMeans(data, args)

    return init_start_K, init_trans_KAK, emission_mu, emission_sigma


def calc_log_proba_arr_for_x(x_n_TD, emission_mu, emission_std, num_states, a_n_T ):

    T = x_n_TD.shape[0]
    D, K = emission_mu[0].shape 
    for t in xrange(T):
        log_proba_list = list()

        if (T > 1):
            x_N_D = x_n_TD[t, :]
        else:
            x_N_D = x_n_TD[t]

        for k in xrange(K):
            if (T > 1):
                a = a_n_T[t] # extract relevant action
            else:
                a = a_n_T
            stddev_DK = emission_std[int(a)]
            mu_DK = emission_mu[int(a)]
            var_k_D = np.square(stddev_DK[:, k])
            arr_k_D = x_N_D - mu_DK[:, k]
            arr_k_D = arr_k_D / var_k_D
            mahal_dist_k_N = np.sum(arr_k_D, axis=0)
            log_proba_list.append( - 0.5 * D * np.log(2 * np.pi) - 0.5 * mahal_dist_k_N - np.sum(np.log(stddev_DK[:, k])))
    log_proba_TK = np.vstack(log_proba_list).T
    return log_proba_TK


def sample_observations(belief_set_TK, emission_mu_list, emission_std_list, data, args):

    T = belief_set_TK.shape[0]
    action_set = data['action_set']
    obs_dim = args['dim_obs']

    obs_set = []
    for t in range(T):
        belief_state_K = belief_set_TK[t,:]
        belief_state_K = np.exp(belief_state_K)
        state = np.argmax(belief_state_K)
        action = action_set[t]
        emission_mu = emission_mu_list[int(action)][:, state]
        emission_std = emission_std_list[int(action)][:, state]
        obs = np.zeros(obs_dim)
        for d in range(obs_dim):
            obs[d] = np.random.normal(emission_mu[d], emission_std[d], 1)
        obs_set.append(obs.astype(int))

    return obs_set


def single_update_belief_log_probas( 
    prev_belief_log_proba_K, curr_data_log_proba_K, ltrans, a):
    trans_log_proba_KK = ltrans[:, int(a), :]
    curr_belief_log_proba_K = logsumexp(trans_log_proba_KK + prev_belief_log_proba_K)
    curr_belief_log_proba_K = curr_belief_log_proba_K + curr_data_log_proba_K
    log_norm_const = logsumexp(curr_belief_log_proba_K)
    cur_belief_log_proba_K = curr_belief_log_proba_K - log_norm_const

    return curr_belief_log_proba_K


def update_belief_log_probas(
    prev_belief_log_proba_K,
    curr_data_log_proba_K,
    ltrans, a,
    return_norm_const=1
):

    trans_log_proba_KK = ltrans[:, int(a), :]
    cur_belief_log_proba_K = logsumexp(
        trans_log_proba_KK + prev_belief_log_proba_K[:, np.newaxis],
        axis=0)
    cur_belief_log_proba_K = cur_belief_log_proba_K \
        + curr_data_log_proba_K
    # Normalize in log space
    log_norm_const = logsumexp(cur_belief_log_proba_K)
    cur_belief_log_proba_K = cur_belief_log_proba_K - log_norm_const
    if return_norm_const:
        return cur_belief_log_proba_K, log_norm_const
    else:
        return cur_belief_log_proba_K


def calc_log_proba_for_one_seq(x_n_TD, a_n_T, lpi, ltrans, emission_mu, emission_std):

    n_timesteps = x_n_TD.shape[0]
    n_states = lpi.shape[0]
    belief_log_proba_TK = np.zeros((n_timesteps, n_states))

    # Compute log proba array
    x_n_log_proba_TK = calc_log_proba_arr_for_x(x_n_TD, emission_mu, emission_std, n_states, a_n_T) 

    x_n_log_proba_TK = x_n_log_proba_TK.flatten()

    # Initialise fwd belief vector at t = 0
    curr_belief_log_proba_K = lpi + x_n_log_proba_TK[0]
    curr_x_log_proba = logsumexp(curr_belief_log_proba_K)
    curr_belief_log_proba_K = curr_belief_log_proba_K - curr_x_log_proba
    belief_log_proba_TK[0, :] = curr_belief_log_proba_K

    log_proba_x = curr_x_log_proba

    for t in range(1, n_timesteps):
        # Update the beliefs over time
        curr_belief_log_proba_K, curr_x_log_proba = update_belief_log_probas(
            curr_belief_log_proba_K,
            x_n_log_proba_TK[t], 
            ltrans, a_n_T[t])

        belief_log_proba_TK[t, :] = curr_belief_log_proba_K
        log_proba_x += curr_x_log_proba

    return log_proba_x, belief_log_proba_TK


def calc_log_proba_for_many_sequences(lpi, ltrans, emission_mu, emission_std, data, fcpt, args):

    x_MD = data['obs_set']
    dataind = data['dataindex_set']
    a_M = data['action_set']

    n_seqs = len(fcpt) - 1
    n_dims = args['dim_obs']
    n_states = args['num_states']

    belief_log_prob_MK = np.zeros((x_MD.shape[0], n_states))
    n_i_track =0

    log_proba_x = 0
    fcpt_base = fcpt[0]

    for n in xrange(n_seqs):
        if n == 0:
            start_n = fcpt[n] - fcpt_base
        else:
            start_n = fcpt[n] + 1 - fcpt_base
        stop_n = fcpt[n + 1] + 1 - fcpt_base

        # use the sequence data corresponding to one sequence
        x_n_TD = x_MD[start_n:stop_n, :]  
        a_n_T = a_M[start_n:stop_n, :]  

        # calculate the beliefs across the time steps of one sequence
        log_proba_x_n, belief_log_prob_TK = calc_log_proba_for_one_seq(x_n_TD, a_n_T, lpi, ltrans, emission_mu, emission_std)

        log_proba_x = log_proba_x + log_proba_x_n

        seq_n_time = belief_log_prob_TK.shape[0]
        # append the beliefs to the set of beliefs
        belief_log_prob_MK[n_i_track:n_i_track + seq_n_time] = belief_log_prob_TK
        n_i_track += seq_n_time

    return log_proba_x, belief_log_prob_MK


def build_pomdp(pi, trans, emission_mu, emission_std, data, fcpt, args):

    lpi = pi - logsumexp(pi, axis=0)
    ltrans = trans - logsumexp(trans, axis=-1, keepdims = True)

    ll = 0
    lbelief_state_set_TK = None

    # collect the complete set of beliefs over all sequences 
    ll, lbelief_state_set_TK = calc_log_proba_for_many_sequences(lpi, ltrans, emission_mu, emission_std, data, fcpt, args)
    return lbelief_state_set_TK, ll


def run_pomdp(data, fcpt, args):

    num_states = args['num_states']
    num_actions = args['num_actions']
    dim_obs = args['dim_obs']

    # initialise model weights
    pi, trans, emission_mu, emission_std = initialise_model_weights(data, args)

    # build the model
    lbelief_set, log_likelihood = build_pomdp(pi, trans, emission_mu, emission_std, data, fcpt, args)

    # given a set of beliefs, sample the next observations
    obs_set = sample_observations(lbelief_set, emission_mu, emission_std, data, args)

    return lbelief_set, obs_set, emission_mu, emission_std, trans
