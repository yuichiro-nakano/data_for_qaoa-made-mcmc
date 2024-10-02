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
import lib.qaoa_mc as qaoa_mc
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

			qaoa_opt = scipy.optimize.minimize(qaoa_cost, qaoa_para, method=qaoa_method, options=qaoa_options)

            # sampling from QAOA distribution
			qaoa_opt_data_idx = qaoa.sampling_QAOA(qaoa_ansatz, qaoa_opt.x, n_train+n_test, seed) # optimize parameter
			qaoa_fix_data_idx = qaoa.sampling_QAOA(qaoa_ansatz, qaoa_init_para, n_train+n_test, seed) # fixed angle

			qaoa_opt_data_nd = np.array([qaoa.number_to_binary(qaoa_opt_data_idx[i], n_spin) for i in range(len(qaoa_opt_data_idx))], dtype='float32') # MADEモデルのコードが重み行列をfloat32で記述しているため、データもfloat32に明示的に指定する！
			qaoa_fix_data_nd = np.array([qaoa.number_to_binary(qaoa_fix_data_idx[i], n_spin) for i in range(len(qaoa_fix_data_idx))], dtype='float32')

			generator.manual_seed(seed)
			qaoa_opt_data = torch.from_numpy(qaoa_opt_data_nd).to(dtype=torch.float32)
			qaoa_opt_traindata, qaoa_opt_testdata = torch.utils.data.random_split(dataset=qaoa_opt_data, lengths=[n_train, n_test], generator=generator)
			qaoa_fix_data = torch.from_numpy(qaoa_fix_data_nd).to(dtype=torch.float32)
			qaoa_fix_traindata, qaoa_fix_testdata = torch.utils.data.random_split(dataset=qaoa_fix_data, lengths=[n_train, n_test], generator=generator)

			qaoa_opt_testset = torch.utils.data.DataLoader(qaoa_opt_testdata, batch_size=batchsize, shuffle=False)
			qaoa_opt_trainset = torch.utils.data.DataLoader(qaoa_opt_traindata, batch_size=batchsize, shuffle=False)
			qaoa_fix_testset = torch.utils.data.DataLoader(qaoa_fix_testdata, batch_size=batchsize, shuffle=False)
			qaoa_fix_trainset = torch.utils.data.DataLoader(qaoa_fix_traindata, batch_size=batchsize, shuffle=False)

			# learn MADE by QAOA samples
			hidden_list = [hidden_size for i in range(hidden_layers)]
			model_qaoa_opt = MADE(n_spin, hidden_list, n_spin, num_masks=1, natural_ordering=True, seed=seed)
			model_qaoa_fix = MADE(n_spin, hidden_list, n_spin, num_masks=1, natural_ordering=True, seed=seed)

			opt_qaoa_opt = torch.optim.Adam(model_qaoa_opt.parameters(), lr=lr, weight_decay=1e-4)
			scheduler_qaoa_opt = torch.optim.lr_scheduler.StepLR(opt_qaoa_opt, step_size=45, gamma=0.1)
			opt_qaoa_fix = torch.optim.Adam(model_qaoa_fix.parameters(), lr=lr, weight_decay=1e-4)
			scheduler_qaoa_fix = torch.optim.lr_scheduler.StepLR(opt_qaoa_fix, step_size=45, gamma=0.1)

			made.run_train(model_qaoa_opt, qaoa_opt_trainset, qaoa_opt_testset, n_epochs, opt_qaoa_opt, scheduler_qaoa_opt, seed)
			made.run_train(model_qaoa_fix, qaoa_fix_trainset, qaoa_fix_testset, n_epochs, opt_qaoa_fix ,scheduler_qaoa_fix, seed)

            # get QAOA-MADE proposal
			all_inputs = np.array([made.number_to_binary(i, n_spin) for i in range(2**n_spin)])
			qaoa_opt_pred_dist = made.compute_log_prob(model_qaoa_opt, all_inputs)
			qaoa_opt_pred_dist = np.exp(qaoa_opt_pred_dist)
			qaoa_fix_pred_dist = made.compute_log_prob(model_qaoa_fix, all_inputs)
			qaoa_fix_pred_dist = np.exp(qaoa_fix_pred_dist)
            
            # get proposal matrix
			qaoa_opt_made_Q = np.zeros((2**n_spin, 2**n_spin))
			qaoa_fix_made_Q = np.zeros((2**n_spin, 2**n_spin))

			init_state = QuantumState(n_spin)
			output_state = QuantumState(n_spin)
			for i in range(2**n_spin):
				spin = mcmc.number_to_spin(i, n_spin)
				init_state.set_computational_basis(i)
				for j in range(2**n_spin):
					output_state.set_computational_basis(j)
					qaoa_opt_made_Q[j,i] = qaoa_opt_pred_dist[j] # MADE(QAOA + opt)
					qaoa_fix_made_Q[j,i] = qaoa_fix_pred_dist[j] # MADE(QAOA + fixed angle)

            # calculate spectral gap
			energy = np.array([ising.spin_energy(ising.number_to_spin(i, n_spin), instance) for i in range(2**n_spin)])

            # MADE(QAOA)
			qaoa_opt_made_P = np.zeros((2**n_spin, 2**n_spin))
			qaoa_fix_made_P = np.zeros((2**n_spin, 2**n_spin))

			qaoa_opt_made_A = mcmc.calc_boltzmann_mh_acceptance(energy, qaoa_opt_made_Q, beta)
			qaoa_fix_made_A = mcmc.calc_boltzmann_mh_acceptance(energy, qaoa_fix_made_Q, beta)
			metropolis_A = mcmc.calc_boltzmann_metropolis_acceptance(energy, beta)

			qaoa_opt_made_P = qaoa_opt_made_Q * qaoa_opt_made_A
			qaoa_fix_made_P = qaoa_fix_made_Q * qaoa_fix_made_A

			np.fill_diagonal(qaoa_opt_made_P, 0)
			np.fill_diagonal(qaoa_fix_made_P, 0)

			for i in range(2**n_spin):
				qaoa_opt_made_P[i,i] = 1 - np.sum(qaoa_opt_made_P[:,i])
				qaoa_fix_made_P[i,i] = 1 - np.sum(qaoa_fix_made_P[:,i])

			gap_data[l,:,k] = np.array([mcmc.spectral_gap(qaoa_opt_made_P), mcmc.spectral_gap(qaoa_fix_made_P)]).T
            
		l += 1
    
	calc_time = time.time() - start_time
    
    # export results
	sub_folder_name = "{0}_sites".format(n_spin)
	sub_folder_path = pathlib.Path(result_dir_name).joinpath(sub_folder_name)
	if not os.path.exists(str(sub_folder_path)):
		os.makedirs(str(sub_folder_path))
    
	for l in range(len(beta_list)):
		fname_out = pathlib.Path(result_dir_name).joinpath(sub_folder_name, 'beta_{0}_add_data.npy'.format('{:.0e}'.format(beta_list[l])))
		np.save(str(fname_out), gap_data[l])
        
	path_config = sub_folder_path.joinpath(datename+'_runtime_add_data.txt')
	with open(str(path_config), mode='w') as f:
		f.write("beta : {0}\n".format(beta_list))
		f.write("calculation time [s] : {0}\n".format(calc_time))
        
if __name__ == '__main__':
    # seed
    seed = 1454
    rng = np.random.default_rng(seed)
    random.seed(seed)
    generator = torch.Generator()
    torch.manual_seed(seed)
    
    # instance
    source_dir_name = '../data/instance_set_2024-0614-1705-31'
    n_spin = 4
    #beta_list = [1e1]
    beta_list = [1e-1, 1e0, 2e0, 5e0, 1e1]
    
    # QAOA
    n_layers = 5
    qaoa_init_para = [0.2705, -0.5899, 0.4803, -0.4492, 0.5074, -0.3559, 0.5646, -0.2643, 0.6397, -0.1291] #文献におけるSKmodelに対するQAOA(p=5)の固定角
    #qaoa_init_para = [0.2528, 0.6004, 0.4531, 0.4670, 0.4750, 0.3880, 0.5146, 0.3176, 0.5650, 0.2325, 0.6392, 0.1291] #文献におけるSKmodelに対するQAOA(p=6)の固定角
    qaoa_method = "BFGS"
    qaoa_options = {"disp": False, "maxiter": 200, "gtol": 1e-6}
    
    # MADE
    n_train = 1000
    n_test = int(n_spin * 0.25)
    hidden_size = int(2 * n_spin)
    hidden_layers = 2
    batchsize = 8
    lr = 0.005
    n_epochs = 30
    
    # return
    result_dir_name = '../result/spectral_gap'
    
    main()