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

	# import instance
	fname_in = pathlib.Path(source_dir_name).joinpath('{0}_sites_instance.pickle'.format(n_spin))
	with open(str(fname_in), 'rb') as f:
		instance = pickle.load(f)

	# main
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

	check_01_time = time.time()

    # sampling from QAOA distribution
	qaoa_opt_data_idx = qaoa.sampling_QAOA(qaoa_ansatz, qaoa_opt.x, n_train+n_test) # optimize parameter
	qaoa_fix_data_idx = qaoa.sampling_QAOA(qaoa_ansatz, qaoa_init_para, n_train+n_test) # fixed angle

	qaoa_opt_data_nd = np.array([qaoa.number_to_binary(qaoa_opt_data_idx[i], n_spin) for i in range(len(qaoa_opt_data_idx))], dtype='float32') # MADEモデルのコードが重み行列をfloat32で記述しているため、データもfloat32に明示的に指定する！
	qaoa_fix_data_nd = np.array([qaoa.number_to_binary(qaoa_fix_data_idx[i], n_spin) for i in range(len(qaoa_fix_data_idx))], dtype='float32')

	qaoa_opt_data = torch.from_numpy(qaoa_opt_data_nd).to(dtype=torch.float32)
	qaoa_opt_traindata, qaoa_opt_testdata = torch.utils.data.random_split(dataset=qaoa_opt_data, lengths=[n_train, n_test], generator=generator)
	qaoa_fix_data = torch.from_numpy(qaoa_fix_data_nd).to(dtype=torch.float32)
	qaoa_fix_traindata, qaoa_fix_testdata = torch.utils.data.random_split(dataset=qaoa_fix_data, lengths=[n_train, n_test], generator=generator)

	qaoa_opt_testset = torch.utils.data.DataLoader(qaoa_opt_testdata, batch_size=batchsize, shuffle=False)
	qaoa_opt_trainset = torch.utils.data.DataLoader(qaoa_opt_traindata, batch_size=batchsize, shuffle=True)
	qaoa_fix_testset = torch.utils.data.DataLoader(qaoa_fix_testdata, batch_size=batchsize, shuffle=False)
	qaoa_fix_trainset = torch.utils.data.DataLoader(qaoa_fix_traindata, batch_size=batchsize, shuffle=True)

	# learn MADE by QAOA samples
	hidden_list = [hidden_size for i in range(hidden_layers)]

	model_qaoa_opt = MADE(n_spin, hidden_list, n_spin, num_masks=1, natural_ordering=True)
	model_qaoa_fix = MADE(n_spin, hidden_list, n_spin, num_masks=1, natural_ordering=True)

	opt_qaoa_opt = torch.optim.Adam(model_qaoa_opt.parameters(), lr=lr, weight_decay=1e-4)
	scheduler_qaoa_opt = torch.optim.lr_scheduler.StepLR(opt_qaoa_opt, step_size=45, gamma=0.1)
	opt_qaoa_fix = torch.optim.Adam(model_qaoa_fix.parameters(), lr=lr, weight_decay=1e-4)
	scheduler_qaoa_fix = torch.optim.lr_scheduler.StepLR(opt_qaoa_fix, step_size=45, gamma=0.1)

	made.run_train(model_qaoa_opt, qaoa_opt_trainset, qaoa_opt_testset, n_epochs, opt_qaoa_opt, scheduler_qaoa_opt, seed)
	made.run_train(model_qaoa_fix, qaoa_fix_trainset, qaoa_fix_testset, n_epochs, opt_qaoa_fix ,scheduler_qaoa_fix, seed)

	# sampling to models and compute the probability of these outputs
	opt_qaoa_made_outputs_nd = made.predict(model_qaoa_opt, n_step)
	opt_qaoa_made_outputs_spin = np.array([made.binary_to_spin(opt_qaoa_made_outputs_nd[i]) for i in range(opt_qaoa_made_outputs_nd.shape[0])])
	fix_qaoa_made_outputs_nd = made.predict(model_qaoa_fix, n_step)
	fix_qaoa_made_outputs_spin = np.array([made.binary_to_spin(fix_qaoa_made_outputs_nd[i]) for i in range(fix_qaoa_made_outputs_nd.shape[0])])

	opt_qaoa_made_log_prob = made.compute_log_prob(model_qaoa_opt, opt_qaoa_made_outputs_nd)
	fix_qaoa_made_log_prob = made.compute_log_prob(model_qaoa_fix, fix_qaoa_made_outputs_nd)
 
	check_02_time = time.time()

	# mcmc simulation
	opt_qaoa_made_result = np.zeros((n_chain, n_step+1, n_spin))
	fix_qaoa_made_result = np.zeros((n_chain, n_step+1, n_spin))
	uniform_result = np.zeros((n_chain, n_step+1, n_spin))
	ssf_result = np.zeros((n_chain, n_step+1, n_spin))

	for k in range(n_chain):
		init_spin = ising.number_to_spin(rng.integers(0, 2**n_spin), n_spin)

		opt_qaoa_made_result[k] = mcmc.neural_update_mcmc(init_spin, instance, model_qaoa_opt, opt_qaoa_made_outputs_spin, opt_qaoa_made_log_prob, beta, n_step, rng)[0]
		fix_qaoa_made_result[k] = mcmc.neural_update_mcmc(init_spin, instance, model_qaoa_fix, fix_qaoa_made_outputs_spin, fix_qaoa_made_log_prob, beta, n_step, rng)[0]
		uniform_result[k] = mcmc.uniform_update_mcmc(init_spin, instance, beta, n_step, rng)[0]
		ssf_result[k] = mcmc.ssf_update_mcmc(init_spin, instance, beta, n_step, rng)[0]

	end_time = time.time()

	# export results
	sub_folder_name = "{0}_sites_result".format(n_spin)
	sub_folder_path = pathlib.Path(result_dir_name).joinpath(sub_folder_name)
	if not os.path.exists(str(sub_folder_path)):
		os.makedirs(str(sub_folder_path))

	fname_out_0 = pathlib.Path(result_dir_name).joinpath(sub_folder_name, 'opt_qaoa_made_result.npy')
	np.save(str(fname_out_0), opt_qaoa_made_result)
	fname_out_1 = pathlib.Path(result_dir_name).joinpath(sub_folder_name, 'fix_qaoa_made_result.npy')
	np.save(str(fname_out_1), fix_qaoa_made_result)
	fname_out_2 = pathlib.Path(result_dir_name).joinpath(sub_folder_name, 'uniform_result.npy')
	np.save(str(fname_out_2), uniform_result)
	fname_out_3 = pathlib.Path(result_dir_name).joinpath(sub_folder_name, 'ssf_result.npy')
	np.save(str(fname_out_3), ssf_result)
        
	path_config = sub_folder_path.joinpath(datename+'_runtime.txt')
	with open(str(path_config), mode='w') as f:
		f.write("total time [s] : {0}\n".format(end_time-start_time))
		f.write("======\n")
		f.write("QAOA optimization [s] : {0}\n".format(check_01_time-start_time))
		f.write("MADE training & sampling [s] : {0}\n".format(check_02_time-check_01_time))
		f.write("MCMC simulation [s] : {0}\n".format(end_time-check_02_time))
		f.write("======\n")
		f.write("beta : {0}\n".format(beta))
		f.write("======\n")
		f.write("n_train : {0}\n".format(n_train))
		f.write("n_test : {0}\n".format(n_test))
		f.write("lr : {0}\n".format(lr))
		f.write("batchsize : {0}\n".format(batchsize))
		f.write("n_epochs : {0}\n".format(n_epochs))

if __name__ == '__main__':
	# seed
	seed = 1454
	rng = np.random.default_rng(seed)
	random.seed(seed)
	generator = torch.Generator().manual_seed(seed)

	# instance
	source_dir_name = '../data'
	n_spin = 15
	beta = 1.0

	# QAOA
	n_layers = 5
	qaoa_init_para = [0.2705, -0.5899, 0.4803, -0.4492, 0.5074, -0.3559, 0.5646, -0.2643, 0.6397, -0.1291] #文献におけるSKmodelに対するQAOA(p=5)の固定角
	qaoa_method = "BFGS"
	qaoa_options = {"disp": False, "maxiter": 200, "gtol": 1e-6}

	# MADE
	n_train = 1000
	n_test = int(n_train * 0.25)
	hidden_size = int(2 * n_spin)
	hidden_layers = 2
	batchsize = 8
	lr = 0.005
	n_epochs = 30

	# mcmc
	n_chain = 10
	n_step = 10000

	# return
	result_dir_name = '../result/mcmc_simulation'

	# logger
	now = datetime.datetime.now()
	datename = now.strftime('%Y-%m%d-%H%M-%S')
	logging.basicConfig(level=logging.DEBUG,
	                    format="%(message)s",
	                    datefmt="[%X]",
	                    handlers=[logging.FileHandler(filename="../log/{0}_log.txt".format(datename))])
	logger = logging.getLogger(__name__)

	main()