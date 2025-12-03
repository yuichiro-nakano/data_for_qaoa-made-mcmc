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
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import lib.made as made
from lib.made import MADE
import lib.mcmc_function as mcmc
import lib.ising_model as ising
from lib.ising_model import Ising_model
import lib.QAOA_function as qaoa
from lib.QAOA_function import QAOA_ansatz
sys.modules['ising_model'] = ising # specify my module to load pickles of the instance set

os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"

def main():
	global qaoa_opt_para
	start_time = time.time()

	# import instance
	fname_in = pathlib.Path(source_dir_name).joinpath('{0}_sites_instance_01.pickle'.format(n_spin))
	with open(str(fname_in), 'rb') as f:
		instance = pickle.load(f)

	# optimize QAOA ansatz
	prob_hamiltonian = instance.get_hamiltonian()
	mixer_hamiltonian = qaoa.generate_X_mixer(n_spin)
	qaoa_ansatz = QAOA_ansatz(prob_hamiltonian, mixer_hamiltonian, n_layers)

	def qaoa_cost(para):
		return qaoa.cost_QAOA(prob_hamiltonian, qaoa_ansatz, para)

	if qaoa_init_para == None:
		qaoa_para = [rng.uniform(0, 2*np.pi) for i in range(2*n_layers)]
	else:
		qaoa_para = qaoa_init_para
	if qaoa_opt_para == None:
		qaoa_opt = scipy.optimize.minimize(qaoa_cost, qaoa_para, method=qaoa_method, options=qaoa_options).x
		qaoa_opt_para = qaoa_opt.x

	qaoa_opt = scipy.optimize.minimize(qaoa_cost, qaoa_para, method=qaoa_method, options=qaoa_options)
	check_01_time = time.time()

	# sampling from QAOA distribution
	qaoa_opt_data_idx = qaoa.sampling_QAOA(qaoa_ansatz, qaoa_opt_para, n_samples) # optimize parameter
	qaoa_fix_data_idx = qaoa.sampling_QAOA(qaoa_ansatz, qaoa_init_para, n_samples) # fixed angle
	check_02_time = time.time()

	qaoa_opt_data_nd = np.array([qaoa.number_to_binary(qaoa_opt_data_idx[i], n_spin) for i in range(len(qaoa_opt_data_idx))], dtype='int32')
	qaoa_fix_data_nd = np.array([qaoa.number_to_binary(qaoa_fix_data_idx[i], n_spin) for i in range(len(qaoa_fix_data_idx))], dtype='int32')

	# export results
	sub_folder_name = "{0}_sites_dataset".format(n_spin)
	sub_folder_path = pathlib.Path(result_dir_name).joinpath(sub_folder_name)
	if not os.path.exists(str(sub_folder_path)):
		os.makedirs(str(sub_folder_path))

	fname_out_0 = pathlib.Path(result_dir_name).joinpath(sub_folder_name, 'opt_qaoa_data.npy')
	np.save(str(fname_out_0), qaoa_opt_data_nd)
	fname_out_1 = pathlib.Path(result_dir_name).joinpath(sub_folder_name, 'fix_qaoa_data.npy')
	np.save(str(fname_out_1), qaoa_fix_data_nd)
        
	path_config = sub_folder_path.joinpath(datename+'_runtime.txt')
	with open(str(path_config), mode='w') as f:
		f.write("optimize time [s] : {0}\n".format(check_01_time-start_time))
		f.write("sampling time (opt&fix) [s] : {0}\n".format(check_02_time-check_01_time))
		f.write("QAOA opt parameter : {0}\n".format(qaoa_opt.x))

if __name__ == '__main__':
	# seed
	seed = 1454
	rng = np.random.default_rng(seed)
	random.seed(seed)
	generator = torch.Generator().manual_seed(seed)

	# instance
	source_dir_name = '../data'
	n_spin = 25

	# QAOA
	n_layers = 5
	qaoa_init_para = [0.2705, -0.5899, 0.4803, -0.4492, 0.5074, -0.3559, 0.5646, -0.2643, 0.6397, -0.1291] #文献におけるSKmodelに対するQAOA(p=5)の固定角
	qaoa_opt_para = [0.51567793,-1.19054919, 0.90884207, -0.9015116, 0.95787243, -0.69219576, 1.05554861, -0.48736768, 1.12796827, -0.28885553] # 25-qubits instance_01
	qaoa_method = "BFGS"
	qaoa_options = {"disp": False, "maxiter": 200, "gtol": 1e-6}
	n_samples = 12500
 
	# return
	result_dir_name = '../result/qaoa_dataset'

	# logger
	now = datetime.datetime.now()
	datename = now.strftime('%Y-%m%d-%H%M-%S')
	logging.basicConfig(level=logging.DEBUG,
	                    format="%(message)s",
	                    datefmt="[%X]",
	                    handlers=[logging.FileHandler(filename="../log/{0}_log.txt".format(datename))])
	logger = logging.getLogger(__name__)

	main()