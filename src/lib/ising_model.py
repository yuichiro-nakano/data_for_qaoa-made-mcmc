
import os
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"

from qulacs import QuantumState, Observable, QuantumCircuit, ParametricQuantumCircuit
from qulacs.state import inner_product
import numpy as np
import networkx as nx
import sys

# model
class Ising_model:
	def __init__(self, n_qubits, rng=None, **kwargs):
		if rng == None:
			rng = np.random.default_rng()

		self.n_qubits = n_qubits
		self.rng = rng

		# random parameter (J,hが与えられていない場合)
		if 'type' in kwargs:
			if kwargs['type'] == '1D':
				J, index_list = self.generate_1D_spin_glass()
				h = np.array([rng.standard_normal() for i in range(self.n_qubits)])

			elif kwargs['type'] == '2D':
				J, index_list = self.generate_2D_spin_glass()
				h = np.array([rng.standard_normal() for i in range(self.n_qubits)])

			elif kwargs['type'] == 'SK':
				J, index_list = self.generate_2D_spin_glass()
				J /= np.sqrt(self.n_qubits)
				h = np.zeros(self.n_qubits)

			else:
				print("\"{0}\" is not defined.".format(kwargs['type']))
				sys.exit()

		# fixed parameter (J,hが与えられている場合)
		else:
			if 'ListOfInt' not in kwargs:
				print("please enter the type of model.")
				sys.exit()
	
			else:
				index_list = kwargs['ListOfInt']

				if 'ListOfJ' not in kwargs:
					J = np.zeros((self.n_qubits, self.n_qubits))
					for i,j in index_list:
						J[i,j] = rng.standard_normal()
				else:
					J = kwargs['ListOfJ']

				if 'ListOfh' not in kwargs:
					h = np.array([rng.standard_normal() for i in range(self.n_qubits)])
				else:
					h = kwargs['ListOfh']

		self.J = J
		self.J_index_list = index_list
		self.h = h
        
                
	def get_parameter(self):
		return self.J, self.J_index_list, self.h
    
	def get_qubit_count(self):
		return self.n_qubits
    
	def get_hamiltonian(self):
		hamiltonian = Observable(self.n_qubits)
    
		# Z-term
		for i in range(self.n_qubits):
			hamiltonian.add_operator(self.h[i], "Z {0}".format(i))

		# ZZ-term
		for i,j in self.J_index_list:
			hamiltonian.add_operator(self.J[i,j], "Z {0} Z {1}".format(i, j))
            
		return hamiltonian  
        
        
	def generate_1D_spin_glass(self):
		J = np.zeros((self.n_qubits, self.n_qubits))
		index_list = []

		for i in range(self.n_qubits-1):
			J[i,i+1] = self.rng.standard_normal()
			index_list.append([i,i+1])

		return J, index_list
    
    
	def generate_2D_spin_glass(self):
		J = np.zeros((self.n_qubits, self.n_qubits))
		index_list = []

		for i in range(self.n_qubits):
			for j in range(self.n_qubits):
				if i < j:
					J[i,j] = self.rng.standard_normal()
					index_list.append([i,j])

		return J, index_list


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


# energy
def spin_energy(spin, instance):
    n_spin = instance.get_qubit_count()
    J, index_list, h = instance.get_parameter()
    
    ising_energy = 0
    for i in range(n_spin):
        ising_energy += h[i] * spin[i]
        for j in range(n_spin):
            ising_energy += J[i,j] * spin[i] * spin[j]
            
    return ising_energy

def min_exact_spin_energy(instance):
    n_spin = instance.get_qubit_count()
    
    min_energy = 1e10
    for i in range(2**n_spin):
        spin = number_to_spin(i, n_spin)
        min_energy = np.min([min_energy, spin_energy(spin, instance)])
        
    return min_energy

# boltzmann distribution
def boltzmann_average_magnetization(n_spin, boltzmann_dist, beta):
    magnetization = np.array([np.sum(number_to_spin(i, n_spin)) for i in range(2**n_spin)]) / n_spin
    return np.sum(boltzmann_dist * magnetization)

def spin_boltzmann_distribution(instance, beta):
    n_spin = instance.get_qubit_count()
    boltzmann_distribution = np.array([-1.0 * beta * spin_energy(number_to_spin(i, n_spin), instance) for i in range(2**n_spin)])
    boltzmann_distribution = np.exp(boltzmann_distribution)
    boltzmann_distribution /= np.sum(boltzmann_distribution)
    
    return boltzmann_distribution

