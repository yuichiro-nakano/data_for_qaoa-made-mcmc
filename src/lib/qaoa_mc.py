import os
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"

from qulacs import QuantumState, Observable, QuantumCircuit, ParametricQuantumCircuit
from qulacs.state import inner_product
import numpy as np
import sys

import lib.ising_model as ising
import lib.mcmc_function as mcmc

class QAOA_circuit:
    def __init__(self, n_qubits, instance, depth):
        if depth % 2 != 0:
            print("depth must be even.")
            sys.exit()
        
        self.n_qubits = n_qubits
        self.depth = depth
        self.instance = instance
        self.ListOfJ, self.index_list, self.ListOfh = instance.get_parameter()
        self.alpha = np.sqrt(self.n_qubits / (np.sum(self.ListOfJ**2) + np.sum(self.ListOfh**2)))
        
        # generate circuit
        circuit = ParametricQuantumCircuit(self.n_qubits)
        para = [0.0 for i in range(self.depth)]
        
        for i in range(2*self.depth):
            if i < self.depth:
                if i % 2 == 0:
                    # RX
                    for j in range(circuit.get_qubit_count()):
                        circuit.add_parametric_RX_gate(j, 0.0) 
                else:
                    # RZ
                    for j in range(circuit.get_qubit_count()):
                        circuit.add_parametric_RZ_gate(j, 0.0)
                    # RZZ
                    for j,k in self.index_list:
                        circuit.add_parametric_multi_Pauli_rotation_gate([j,k], [3,3], 0.0)
                        
            else:
                if i % 2 == 0:
                    # RZZ
                    for j,k in self.index_list:
                        circuit.add_parametric_multi_Pauli_rotation_gate([j,k], [3,3], 0.0)
                    # RZ
                    for j in range(circuit.get_qubit_count()):
                        circuit.add_parametric_RZ_gate(j, 0.0)
                        
                else:
                    # RX
                    for j in range(circuit.get_qubit_count()):
                        circuit.add_parametric_RX_gate(j, 0.0)
                       
        self.circuit = circuit
        self.para = para
            
        
    def update_quantum_state(self, state):
        self.circuit.update_quantum_state(state)
        return
    
    def get_parameter(self):
        return self.para
    
    def set_parameter(self, para):
        l = 0

        for i in range(2*self.depth):
            # 回路の左半分か右半分で場合分け
            if i < self.depth:
                # パラメータ番号の対応
                index = i
                
                # 回路にパラメータ代入
                if index % 2 == 0: # mixer hamiltonian
                    for j in range(self.n_qubits):
                        self.circuit.set_parameter(l, -para[index])
                        l += 1

                else: # problem hamiltonian
                    # RZ
                    for j in range(self.n_qubits):
                        self.circuit.set_parameter(l, -self.alpha*self.ListOfh[j]*para[index])
                        l += 1
                    
                    # RZZ
                    for j, k in self.index_list:
                        self.circuit.set_parameter(l, -self.alpha*self.ListOfJ[j,k]*para[index])
                        l += 1
                
            else:
                # パラメータ番号の対応
                index = 2*self.depth - 1 - i
            
                # 回路にパラメータ代入
                if index % 2 == 0: # mixer hamiltonian
                    for j in range(self.n_qubits):
                        self.circuit.set_parameter(l, -para[index])
                        l += 1

                else: # problem hamiltonian
                    # RZZ
                    for j, k in self.index_list:
                        self.circuit.set_parameter(l, -self.alpha*self.ListOfJ[j,k]*para[index])
                        l += 1
                        
                    # RZ
                    for j in range(self.n_qubits):
                        self.circuit.set_parameter(l, -self.alpha*self.ListOfh[j]*para[index])
                        l += 1
                    
        self.para = para
        
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
def qaoa_mc_update(spin, ansatz, instance, beta, rng=None, rand_seed=None):
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

# optimize
def cost_function(ansatz, instance, beta, **kwargs):
	n_qubit = ansatz.get_qubit_count()
    
	if 'mode' in kwargs:
		if kwargs['mode'] == 'exact':
			proposal_mat = np.zeros((2**n_qubit, 2**n_qubit))

			init_state = QuantumState(n_qubit)
			output_state = QuantumState(n_qubit)
			for i in range(2**n_qubit):
				init_state.set_computational_basis(i)
				ansatz.update_quantum_state(init_state)

				for j in range(2**n_qubit):
					output_state.set_computational_basis(j)
					proposal_mat[j,i] = (np.abs(inner_product(output_state, init_state)))**2

			energy_vector = np.array([ising.spin_energy(ising.number_to_spin(i, n_qubit), instance) for i in range(2**n_qubit)])
			acceptance_mat = mcmc.calc_boltzmann_metropolis_acceptance(energy_vector, beta)

			target_dist = ising.spin_boltzmann_distribution(instance, beta)
			cost = np.sum(target_dist * proposal_mat * acceptance_mat)

		elif kwargs['mode'] == 'mcmc':
			if 'rng' in kwargs:
				rng = kwargs['rng']
			else:
				rng = np.random.default_rng()
	
			if 'rand_seed' in kwargs:
				rand_seed = kwargs['rand_seed']
			else:
				rand_seed = None

			if 'n_iter' in kwargs:
				n_iter = kwargs['n_iter']
			else:
				print("n_iter is none.")
				sys.exit()

			cost_sample = np.zeros(n_iter)
			init_state = mcmc.number_to_spin(rng.integers(0, 2**n_qubit), n_qubit)
			for i in range(n_iter):
				if i == 0:
					spin = init_state
				else:
					spin, cost_sample[i] = qaoa_mc_update(spin, ansatz, instance, beta, rng, rand_seed)[0:2]
	
			cost = np.mean(cost_sample)
   
		else:
			print("This mode is not undefined.")
			sys.exit()
	
	else:
		print("Please select mode.")
		sys.exit()

	return cost