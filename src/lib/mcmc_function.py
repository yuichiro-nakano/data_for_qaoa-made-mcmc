import os
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"

import numpy as np
import scipy
import lib.ising_model as ising

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
def boltzmann_metropolis(spin, proposal_spin, instance, beta, seed=None):
    rng = np.random.default_rng(seed)
    
    energy_diff = -1.0 * beta * (ising.spin_energy(proposal_spin,instance) - ising.spin_energy(spin,instance))
    
    # avoid overflowing value of np.exp()
    if energy_diff > 500:
        acceptance = 1.0
    elif energy_diff < -500:
        acceptance = 0.0
    else:
        acceptance = float(min(1, np.exp(energy_diff)))
    
    # accept/reject propose
    if acceptance >= rng.uniform(0,1):
        count = 1
        return proposal_spin, acceptance, count
    else:
        count = 0
        return spin, acceptance, count
    
def boltzmann_metropolis_hastings(spin, proposal_spin, proposal_mat, instance, beta, seed=None):
    rng = np.random.default_rng(seed)
    
    i = spin_to_number(spin)
    j = spin_to_number(proposal_spin)
    
    energy_diff = -1.0 * beta * (ising.spin_energy(proposal_spin,instance) - ising.spin_energy(spin,instance))
    
    # avoid dividing zero when Q(i->j)=0
    if proposal_mat[j,i] < 1e-15:
        acceptance = 1.0
    
    # avoid overflowing an imput of np.exp()
    if energy_diff > 500:
        acceptance = 1.0
    elif energy_diff < -500:
        acceptance = 0.0
    else:
        diff = np.exp(energy_diff) * proposal_mat[i,j] / proposal_mat[j,i]
        acceptance = np.minimum(1, diff) 
    
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
def ssf_update(spin, instance, beta, seed):
    rng = np.random.default_rng(seed)
    n_spin = spin.shape[0]
    
    # make a porposal
    flip_index = rng.integers(0, n_spin)
    proposal_spin = single_spin_flip(spin, flip_index)
    
    # accept or reject the proposal
    return boltzmann_metropolis(spin, proposal_spin, instance, beta, seed)
    
def uniform_update(spin, instance, beta, seed):
    rng = np.random.default_rng(seed)
    n_spin = spin.shape[0]
    
    # make a porposal
    proposal_spin = number_to_spin(rng.integers(0, n_spin), n_spin)
    
    # accept or reject the proposal
    return boltzmann_metropolis(spin, proposal_spin, instance, beta, seed)


# utils
def calc_boltzmann_mh_acceptance(energy_vector, proposal_mat, beta):
	n_state = energy_vector.shape[0]
	
	# transition from i to j
	energy_i = np.tile(energy_vector, (n_state,1))
	energy_j = energy_i.T
	reverse_proposal_mat = proposal_mat.T
	
	likelihood = -1.0 * beta * (energy_j - energy_i) + np.log(reverse_proposal_mat) - np.log(proposal_mat)
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