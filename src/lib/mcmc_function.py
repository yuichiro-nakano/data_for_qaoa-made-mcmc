import os
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"

import numpy as np
import scipy
import torch
#import lib.ising_model as ising
#import lib.made as made
import ising_model as ising
import made as made

# spin
def spin_to_number(spin):
    spin = np.flipud(spin)
    
    spin_binstr = '0b'
    for i in range(spin.shape[0]):
        if spin[i] == 1:
            spin_binstr += '0'
        else:
            spin_binstr += '1'
            
    spin_number = int(spin_binstr, 0)
    
    return spin_number


def number_to_spin(number, n_spin):
    spin = format(number,"b").zfill(n_spin)
    spin_list = np.array([1 if spin[i]=='0' else -1 for i in range(n_spin)])
    spin_list = np.flipud(spin_list)
    
    return spin_list

"""
def spin_measurement(state):
    n_qubits = state.get_qubit_count()
    state_index = state.sampling(1)[0]
    spin_list = number_to_spin(state_index, n_qubits)
    
    return spin_list
"""


# metropolis-hastings
def boltzmann_metropolis(spin, proposal_spin, instance, beta, rng=None):
	if rng == None:
		rng = np.random.default_rng()

	energy_diff = -1.0 * beta * (ising.spin_energy(proposal_spin,instance) - ising.spin_energy(spin,instance))

	# avoid overflowing value of np.exp()
	if energy_diff > 500:
		acceptance = 1.0
	elif energy_diff < -500:
		acceptance = 0.0
	else:
		acceptance = np.exp(np.minimum(energy_diff, 0.0))

	# accept/reject propose
	if acceptance >= rng.uniform(0,1):
		count = 1
		return proposal_spin, acceptance, count
	else:
		count = 0
		return spin, acceptance, count
    
def boltzmann_metropolis_hastings(spin, proposal_spin, proposal_log_prob, reverse_proposal_log_prob, instance, beta, rng=None):
	if rng == None:
		rng = np.random.default_rng()
     
	i = spin_to_number(spin)
	j = spin_to_number(proposal_spin)
    
	energy_diff = -1.0 * beta * (ising.spin_energy(proposal_spin,instance) - ising.spin_energy(spin,instance))
	diff = energy_diff + reverse_proposal_log_prob - proposal_log_prob
	diff = np.exp(diff)
	acceptance = np.minimum(diff, 1.0)
    
	# accept/reject propose
	if acceptance >= rng.uniform(0,1):
		count = 1
		return proposal_spin, acceptance, count
	else:
		count = 0
		return spin, acceptance, count


# proposal
def single_spin_flip(spin, flip_index):
    proposal_spin = spin.copy()
    
    if spin[flip_index] == 1:
        proposal_spin[flip_index] = -1
    elif spin[flip_index] == -1:
        proposal_spin[flip_index] = 1
        
    return proposal_spin
   
# mcmc 
def ssf_update_mcmc(init_spin, instance, beta, n_iter, rng=None):
	if rng == None:
		rng = np.random.default_rng()
    
	n_spin = init_spin.shape[0]
	state_history = np.zeros((n_iter+1, n_spin))
	state_history[0] = init_spin
	acceptance_history = np.zeros(n_iter)

	accept_state = init_spin

	for i in range(n_iter):
		# make a porposal
		flip_index = rng.integers(0, n_spin)
		proposal_state = single_spin_flip(accept_state, flip_index)

		# accept or reject the proposal
		next_state, acceptance, flag = boltzmann_metropolis(accept_state, proposal_state, instance, beta, rng)

		# update the accept_state
		state_history[i+1] = next_state
		acceptance_history[i] = acceptance
		if flag:
			accept_state = next_state
	
	return state_history, acceptance_history
    
def uniform_update_mcmc(init_spin, instance, beta, n_iter, rng=None):
	if rng == None:
		rng = np.random.default_rng()
    
    # preparation
	n_spin = init_spin.shape[0]
	state_history = np.zeros((n_iter+1, n_spin))
	state_history[0] = init_spin
	acceptance_history = np.zeros(n_iter)

	accept_state = init_spin

	for i in range(n_iter):
		# make a porposal
		proposal_state = number_to_spin(rng.integers(0, 2**n_spin), n_spin)

		# accept or reject the proposal
		next_state, acceptance, flag = boltzmann_metropolis(accept_state, proposal_state, instance, beta, rng)

		# update the accept_state
		state_history[i+1] = next_state
		acceptance_history[i] = acceptance
		if flag:
			accept_state = next_state
	
	return state_history, acceptance_history

def neural_update_mcmc(init_spin, instance, model, proposal_history, log_prob_history, beta, n_iter, rng=None):
	if rng == None:
		rng = np.random.default_rng()
    
    # preparation
	n_spin = init_spin.shape[0]
	state_history = np.zeros((n_iter+1, n_spin))
	state_history[0] = init_spin
	acceptance_history = np.zeros(n_iter)

	current_state = init_spin
	current_log_prob = made.compute_log_prob(model, made.spin_to_binary(init_spin))

	for i in range(n_iter):
		# accept or reject the proposal
		next_state, acceptance, flag = boltzmann_metropolis_hastings(current_state, proposal_history[i], log_prob_history[i], current_log_prob, instance, beta, rng)

		# update the accept_state
		state_history[i+1] = next_state
		acceptance_history[i] = acceptance
		if flag:
			current_state = proposal_history[i]
			current_log_prob = log_prob_history[i]
	
	return state_history, acceptance_history

# utils
def calc_boltzmann_mh_acceptance(energy_vector, proposal_mat, beta):
	n_state = energy_vector.shape[0]
	
	# transition from i to j
	energy_i = np.tile(energy_vector, (n_state,1))
	energy_j = energy_i.T
	reverse_proposal_mat = proposal_mat.T

	proposal_diff = np.log(reverse_proposal_mat) - np.log(proposal_mat)
	proposal_diff_ex = np.where(np.isnan(proposal_diff)==True, 0.0, proposal_diff) # exception handling when Q=Q'=0

	likelihood = -1.0 * beta * (energy_j - energy_i) + proposal_diff_ex
	likelihood = np.exp(likelihood)
	
	ones_mat = np.ones((n_state, n_state))
	acceptance_mat = np.minimum(ones_mat, likelihood)
	
	return acceptance_mat

def calc_boltzmann_metropolis_acceptance(energy_vector, beta):
	n_state = energy_vector.shape[0]
	
	# transition from i to j
	energy_i = np.tile(energy_vector, (n_state,1))
	energy_j = energy_i.T
	
	likelihood = -1.0 * beta * (energy_j - energy_i)
	likelihood = np.exp(likelihood)
	
	ones_mat = np.ones((n_state, n_state))
	acceptance_mat = np.minimum(ones_mat, likelihood)
	
	return acceptance_mat

def spectral_gap(mat):
    P_eigvals = scipy.linalg.eigvals(mat)
    P_eigvals_sort = np.sort(np.abs(P_eigvals))
    gap = 1 - P_eigvals_sort[-2]
    
    return gap