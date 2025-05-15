import os
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"

from qulacs import QuantumState, Observable, QuantumCircuit, ParametricQuantumCircuit
from qulacs.state import inner_product
import numpy as np
import sys

import lib.ising_model as ising
import lib.mcmc_function as mcmc

class Qe_MCMC_circuit:
	def __init__(self, n_qubits, instance):
		self.n_qubits = n_qubits
		self.instance = instance
		self.ListOfJ, self.index_list, self.ListOfh = instance.get_parameter()
		self.alpha = np.sqrt(self.n_qubits / (np.sum(self.ListOfJ**2) + np.sum(self.ListOfh**2)))

		# generate circuit
		timestep = 0.8
		gamma = 0.45
		time = 12
		num_layers = int(time/timestep)
		prob_param = (1 - gamma) * self.alpha		

		circuit = ParametricQuantumCircuit(self.n_qubits)
		for i in range(num_layers-1):
			# mixer layer
			for j in range(circuit.get_qubit_count()):
				circuit.add_parametric_multi_Pauli_rotation_gate([j], [1], -2.0*gamma*timestep) # exp(-i * gamma * H_mix * time_step), H_mix = \sum X_j		
			# cost layer
			for j in range(circuit.get_qubit_count()):
				circuit.add_parametric_multi_Pauli_rotation_gate([j], [3], 2.0*prob_param*self.ListOfh[j]*timestep)
			for j,k in self.index_list:
				circuit.add_parametric_multi_Pauli_rotation_gate([j,k], [3,3], 2.0*prob_param*self.ListOfJ[j,k]*timestep) # exp(-i * (1-gamma) * alpha * H_prob * time_step), H_prob = -\sum J_jk Z_j Z_k - \sum h_j Z_j		
		# mixer layer
		for j in range(circuit.get_qubit_count()):
				circuit.add_parametric_multi_Pauli_rotation_gate([j], [1], -2.0*gamma*timestep)
                       
		self.circuit = circuit
		self.para = [gamma, time]
            
        
	def update_quantum_state(self, state):
		self.circuit.update_quantum_state(state)
		return

	def get_qulacs_circuit(self):
		return self.circuit

	def get_qulacs_parameter(self):
		qcirc_para = []
		for i in range(self.circuit.get_parameter_count()):
			qcirc_para.append(self.circuit.get_parameter(i))	
		return qcirc_para

	def get_parameter_count(self):
		return len(self.para)

	def get_qubit_count(self):
		return self.n_qubits
    
# util
def spin_measurement(state, rand_seed=None):
	n_qubits = state.get_qubit_count()

	if rand_seed == None:
		state_index = state.sampling(1)[0]
	else:
		state_index = state.sampling(1, rand_seed)[0]
	
	spin_list = mcmc.number_to_spin(state_index, n_qubits)

	return spin_list

# mcmc
def qe_mcmc_update(spin, ansatz, instance, beta, rng=None, rand_seed=None):
	if rng == None:
		rng = np.random.default_rng()

	n_spin = spin.shape[0]

	# make a porposal
	init_state_idx = mcmc.spin_to_number(spin)
	init_state = QuantumState(n_spin)
	init_state.set_computational_basis(init_state_idx)
	ansatz.update_quantum_state(init_state)
	proposal_spin = spin_measurement(init_state, rand_seed)

	# accept or reject the proposal
	return mcmc.boltzmann_metropolis(spin, proposal_spin, instance, beta, rng)