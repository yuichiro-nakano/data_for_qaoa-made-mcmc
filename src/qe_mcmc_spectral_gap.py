import numpy as np
import scipy
import networkx as nx
import time
import datetime
import os
import random
import sys
import pathlib
import pickle
import logging
from qulacs import ParametricQuantumCircuit, QuantumState, Observable, QuantumCircuit, PauliOperator, GeneralQuantumOperator
from qulacs.state import inner_product

import lib.made as made
from lib.made import MADE
import lib.mcmc_function as mcmc
import lib.ising_model as ising
from lib.ising_model import Ising_model
import lib.QAOA_function as qaoa
from lib.QAOA_function import QAOA_ansatz
import lib.qe_mcmc as qe_mcmc
from lib.qe_mcmc import Qe_MCMC_circuit

sys.modules['ising_model'] = ising # specify my module to load pickles of the instance set

os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"

def main():
	start_time = time.time()
	now = datetime.datetime.now()
	datename = now.strftime('%Y-%m%d-%H%M-%S')

	logging.basicConfig(level=logging.DEBUG,
	                    format="%(message)s",
	                    datefmt="[%X]",
	                    handlers=[logging.FileHandler(filename="../log/{0}_log.txt".format(datename))])
	logger = logging.getLogger(__name__)

	# import instance sets
	fname_in = pathlib.Path(source_dir_name).joinpath('{0}_sites_instance_set.pickle'.format(n_spin))
	with open(str(fname_in), 'rb') as f:
		instance_set = pickle.load(f)
    
	# run calculation for each temperature
	n_beta = len(beta_list)
	n_instance = len(instance_set)
	gap_data = np.zeros((n_beta, 2, n_instance))

	l = 0
	for beta in beta_list:
    	# calcurate the spectral gap of instances
		for k in range(n_instance):
			instance = instance_set[k]

			# qe-mcmc
			qe_mcmc_circuit = Qe_MCMC_circuit(n_spin, instance)
            
            # get proposal matrix
			qe_mcmc_Q = np.zeros((2**n_spin, 2**n_spin))

			init_state = QuantumState(n_spin)
			output_state = QuantumState(n_spin)
			for i in range(2**n_spin):
				init_state.set_computational_basis(i)
				qe_mcmc_circuit.update_quantum_state(init_state)
				for j in range(2**n_spin):
					output_state.set_computational_basis(j)
					qe_mcmc_Q[j,i] = (np.abs(inner_product(output_state, init_state)))**2 # qe-mcmc				

            # calculate spectral gap
			energy = np.array([ising.spin_energy(ising.number_to_spin(i, n_spin), instance) for i in range(2**n_spin)])
			metropolis_A = mcmc.calc_boltzmann_metropolis_acceptance(energy, beta)

			qe_mcmc_P = qe_mcmc_Q * metropolis_A
			np.fill_diagonal(qe_mcmc_P, 0)
			for i in range(2**n_spin):
				qe_mcmc_P[i,i] = 1 - np.sum(qe_mcmc_P[:,i])

			gap_data[l,:,k] = mcmc.spectral_gap(qe_mcmc_P)
            
		l += 1
    
	calc_time = time.time() - start_time
    
    # export results
	sub_folder_name = "{0}_sites".format(n_spin)
	sub_folder_path = pathlib.Path(result_dir_name).joinpath(sub_folder_name)
	if not os.path.exists(str(sub_folder_path)):
		os.makedirs(str(sub_folder_path))
    
	for l in range(len(beta_list)):
		fname_out = pathlib.Path(result_dir_name).joinpath(sub_folder_name, 'beta_{0}_qe_mcmc.npy'.format('{:.0e}'.format(beta_list[l])))
		np.save(str(fname_out), gap_data[l])
        
	path_config = sub_folder_path.joinpath(datename+'_runtime_qe_mcmc.txt')
	with open(str(path_config), mode='w') as f:
		f.write("beta : {0}\n".format(beta_list))
		f.write("calculation time [s] : {0}\n".format(calc_time))
        
if __name__ == '__main__':
    # seed
    seed = 1454
    rng = np.random.default_rng(seed)
    random.seed(seed)
    
    # instance
    source_dir_name = '../data/instance_set_2024-0614-1705-31'
    n_spin = 4
    beta_list = [1e-1, 1e0, 2e0, 5e0, 1e1]
    
    # return
    result_dir_name = '../result/spectral_gap'
    
    main()