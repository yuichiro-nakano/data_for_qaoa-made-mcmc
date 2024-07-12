import os
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"

from qulacs import QuantumState, Observable, QuantumCircuit, ParametricQuantumCircuit
from qulacs.state import inner_product
import numpy as np
import ising_model as ising

# ansatz
class QAOA_ansatz:
    def __init__(self, prob_hamiltonian, mixer_hamiltonian, n_layers):
        self.n_qubits = prob_hamiltonian.get_qubit_count()
        self.prob_hamiltonian = prob_hamiltonian
        self.mixer_hamiltonian = mixer_hamiltonian
        self.n_layers = n_layers
        
        circuit = ParametricQuantumCircuit(self.n_qubits)
        para = [0. for i in range(2*self.n_layers)]
        
        # get pauli data
        prob_index_list = []
        prob_pauli_id_list = []
        prob_coef = []
        for i in range(prob_hamiltonian.get_term_count()):
            pauli = prob_hamiltonian.get_term(i)
            prob_index_list.append(pauli.get_index_list())
            prob_pauli_id_list.append(pauli.get_pauli_id_list())
            prob_coef.append(pauli.get_coef().real)
        
        mix_index_list = []
        mix_pauli_id_list = []
        mix_coef = []
        for i in range(mixer_hamiltonian.get_term_count()):
            pauli = mixer_hamiltonian.get_term(i)
            mix_index_list.append(pauli.get_index_list())
            mix_pauli_id_list.append(pauli.get_pauli_id_list())
            mix_coef.append(pauli.get_coef().real)
            
        self.prob_coef = prob_coef
        self.mix_coef = mix_coef
        
        # generate a parametric quantum circuit
        for i in range(self.n_layers):
            n_prob_terms = self.prob_hamiltonian.get_term_count()
            n_mix_terms = self.mixer_hamiltonian.get_term_count()
            
            for j in range(n_prob_terms):
                circuit.add_parametric_multi_Pauli_rotation_gate(prob_index_list[j], prob_pauli_id_list[j], prob_coef[j]*para[2*i])
            for j in range(n_mix_terms):
                circuit.add_parametric_multi_Pauli_rotation_gate(mix_index_list[j], mix_pauli_id_list[j], mix_coef[j]*para[2*i+1])
            
        self.circuit = circuit
        self.para = para
    
    def get_parameter(self):
        return self.para
    
    def get_qulacs_circuit(self):
        return self.circuit
    
    def get_parameter_count(self):
        return len(self.para)
    
    def get_qubit_count(self):
        return self.n_qubits
    
    def set_parameter(self, para):
        n_prob_terms = self.prob_hamiltonian.get_term_count()
        n_mix_terms = self.mixer_hamiltonian.get_term_count()
        
        idx = 0
        for i in range(self.n_layers):
            for j in range(n_prob_terms):
                #print(idx)
                self.circuit.set_parameter(idx, self.prob_coef[j]*para[2*i])
                idx += 1
                
            for j in range(n_mix_terms):
                #print(idx)
                self.circuit.set_parameter(idx, self.mix_coef[j]*para[2*i+1])
                idx += 1
            
        self.para = para
    
    def update_quantum_state(self, state):
        self.circuit.update_quantum_state(state)
        return
    
def generate_X_mixer(n_qubits):
    mixer = Observable(n_qubits)
    for k in range(n_qubits):
        mixer.add_operator(1.0,"X {0}".format(k)) 
    return mixer

# QAOA util
def cost_QAOA(prob_hamiltonian, ansatz, para):
    n_qubits = ansatz.get_qubit_count()
    state = QuantumState(n_qubits)
    
    # prepare an initial state
    state = QuantumState(n_qubits)
    pre_circuit = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        pre_circuit.add_H_gate(i)
    pre_circuit.update_quantum_state(state)
    
    # apply QAOA ansatz
    ansatz.set_parameter(para)
    ansatz.update_quantum_state(state)
    cost_val = prob_hamiltonian.get_expectation_value(state)
    
    return cost_val

def sampling_QAOA(ansatz, para, n_samples, rand_seed=None):
	n_qubits = ansatz.get_qubit_count()
	state = QuantumState(n_qubits)

	# prepare an initial state
	state = QuantumState(n_qubits)
	pre_circuit = QuantumCircuit(n_qubits)
	for i in range(n_qubits):
		pre_circuit.add_H_gate(i)
	pre_circuit.update_quantum_state(state)

	# apply QAOA ansatz
	ansatz.set_parameter(para)
	ansatz.update_quantum_state(state)
    
	if rand_seed == None:
		return state.sampling(n_samples)
	else:
		return state.sampling(n_samples, rand_seed)

def distribution_QAOA(ansatz, para):
    n_qubits = ansatz.get_qubit_count()
    state = QuantumState(n_qubits)
    
    # prepare an initial state
    pre_circuit = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        pre_circuit.add_H_gate(i)
    pre_circuit.update_quantum_state(state)
    
    # apply QAOA ansatz
    ansatz.set_parameter(para)
    ansatz.update_quantum_state(state)
    
    # calculate the output distribution
    probs = np.abs(state.get_vector())**2
    return probs

# util
def spin_to_number(spin_list):
    spin_list = np.flipud(spin_list)
    
    spin_binstr = '0b'
    for i in range(spin_list.shape[0]):
        if spin_list[i] == 1:
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

def binary_to_number(binary_list):
    binary_list = np.flipud(binary_list)
    
    bin_str = "0b"
    for i in range(binary_list.shape[0]):
        bin_str += str(int(binary_list[i]))
        
    bin_number = int(bin_str, 0)
    
    return bin_number

def number_to_binary(number, n_dim):
    bin = format(number,"b").zfill(n_dim)
    bin_list = np.array([0 if bin[i]=='0' else 1 for i in range(n_dim)])
    bin_list = np.flipud(bin_list)
    
    return bin_list