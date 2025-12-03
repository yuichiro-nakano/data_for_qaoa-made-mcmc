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
	start_time = time.time()

	# import instance & dataset
	fname_in = pathlib.Path(source_dir_name).joinpath('{0}_sites_instance_01.pickle'.format(n_spin))
	with open(str(fname_in), 'rb') as f:
		instance = pickle.load(f)

	qaoa_opt_data_nd = np.load("../result/qaoa_dataset/{0}_sites_dataset/opt_qaoa_data.npy".format(n_spin))
	qaoa_fix_data_nd = np.load("../result/qaoa_dataset/{0}_sites_dataset/fix_qaoa_data.npy".format(n_spin))

	# main
	opt_qaoa_made_result = np.zeros((len(n_train_list), n_subset, n_step+1, n_spin), dtype=np.int8)
	fix_qaoa_made_result = np.zeros((len(n_train_list), n_subset, n_step+1, n_spin), dtype=np.int8)
	opt_qaoa_made_update_log = np.zeros((len(n_train_list), n_subset, n_step), dtype=bool)
	fix_qaoa_made_update_log = np.zeros((len(n_train_list), n_subset, n_step), dtype=bool)
	opt_qaoa_made_acceptance_history = np.zeros((len(n_train_list), n_subset, n_step))
	fix_qaoa_made_acceptance_history = np.zeros((len(n_train_list), n_subset, n_step))

	learning_time_result = np.zeros((len(n_train_list), n_subset, 2))
	pred_time_result = np.zeros((len(n_train_list), n_subset, 2))

	for i in range(len(n_train_list)):
	    # train set
		n_train = n_train_list[i]

		for j in range(n_subset):
			subset_idx = rng.integers(low=0, high=qaoa_opt_data_nd.shape[0], size=int(n_train+np.floor(0.25*n_train)))

			qaoa_opt_data = torch.from_numpy(qaoa_opt_data_nd[subset_idx]).to(dtype=torch.float32)
			qaoa_fix_data = torch.from_numpy(qaoa_fix_data_nd[subset_idx]).to(dtype=torch.float32)

			qaoa_opt_traindata, qaoa_opt_testdata = torch.utils.data.random_split(dataset=qaoa_opt_data, lengths=[n_train, int(np.floor(0.25*n_train))])
			qaoa_fix_traindata, qaoa_fix_testdata = torch.utils.data.random_split(dataset=qaoa_fix_data, lengths=[n_train, int(np.floor(0.25*n_train))])

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

			start_learning_time_opt = time.time()
			made.run_train(model_qaoa_opt, qaoa_opt_trainset, qaoa_opt_testset, n_epochs, opt_qaoa_opt, scheduler_qaoa_opt, seed)
			end_learning_time_opt = time.time()
			start_learning_time_fix = time.time()
			made.run_train(model_qaoa_fix, qaoa_fix_trainset, qaoa_fix_testset, n_epochs, opt_qaoa_fix ,scheduler_qaoa_fix, seed)
			end_learning_time_fix = time.time()

			learning_time_result[i,j,0], learning_time_result[i,j,1] = end_learning_time_opt-start_learning_time_opt, end_learning_time_fix-start_learning_time_fix

			# sampling to models and compute the probability of these outputs
			start_pred_time_opt = time.time()
			opt_qaoa_made_outputs_nd = made.predict(model_qaoa_opt, n_step)
			opt_qaoa_made_log_prob = made.compute_log_prob(model_qaoa_opt, opt_qaoa_made_outputs_nd)
			end_pred_time_opt = time.time()
			opt_qaoa_made_outputs_spin = np.array([made.binary_to_spin(opt_qaoa_made_outputs_nd[i]) for i in range(opt_qaoa_made_outputs_nd.shape[0])])

			start_pred_time_fix = time.time()
			fix_qaoa_made_outputs_nd = made.predict(model_qaoa_fix, n_step)
			fix_qaoa_made_log_prob = made.compute_log_prob(model_qaoa_fix, fix_qaoa_made_outputs_nd)
			end_pred_time_fix = time.time()
			fix_qaoa_made_outputs_spin = np.array([made.binary_to_spin(fix_qaoa_made_outputs_nd[i]) for i in range(fix_qaoa_made_outputs_nd.shape[0])])

			pred_time_result[i,j,0], pred_time_result[i,j,1] = end_pred_time_opt-start_pred_time_opt, end_pred_time_fix-start_pred_time_fix

	  		# mcmc simulation
			init_spin = ising.number_to_spin(rng.integers(0, 2**n_spin), n_spin)
			opt_qaoa_made_result[i,j], opt_qaoa_made_acceptance_history[i,j], opt_qaoa_made_update_log[i,j] = mcmc.neural_update_mcmc(init_spin, instance, model_qaoa_opt, opt_qaoa_made_outputs_spin, opt_qaoa_made_log_prob, beta, n_step, rng)
			fix_qaoa_made_result[i,j], fix_qaoa_made_acceptance_history[i,j], fix_qaoa_made_update_log[i,j] = mcmc.neural_update_mcmc(init_spin, instance, model_qaoa_fix, fix_qaoa_made_outputs_spin, fix_qaoa_made_log_prob, beta, n_step, rng)

	end_time = time.time()

 	# export results
	sub_folder_name = "{0}_sites_result".format(n_spin)
	sub_folder_path = pathlib.Path(result_dir_name).joinpath(sub_folder_name)
	if not os.path.exists(str(sub_folder_path)):
		os.makedirs(str(sub_folder_path))

	for i in range(len(n_train_list)):
		subsub_folder_name = "{0}_samples".format(n_train_list[i])
		subsub_folder_path = sub_folder_path.joinpath(subsub_folder_name)
		if not os.path.exists(str(subsub_folder_path)):
			os.makedirs(str(subsub_folder_path))
    
		fname_out_0 = sub_folder_path.joinpath(subsub_folder_name, 'opt_qaoa_made_result.npy')
		np.save(str(fname_out_0), opt_qaoa_made_result[i])
		fname_out_1 = sub_folder_path.joinpath(subsub_folder_name, 'fix_qaoa_made_result.npy')
		np.save(str(fname_out_1), fix_qaoa_made_result[i])

		fname_out_2 = sub_folder_path.joinpath(subsub_folder_name, 'opt_qaoa_made_acceptance_history.npy')
		np.save(str(fname_out_2), opt_qaoa_made_acceptance_history[i])
		fname_out_3 = sub_folder_path.joinpath(subsub_folder_name, 'fix_qaoa_made_acceptance_history.npy')
		np.save(str(fname_out_3), fix_qaoa_made_acceptance_history[i])

		fname_out_4 = sub_folder_path.joinpath(subsub_folder_name, 'opt_qaoa_made_update_log.npy')
		np.save(str(fname_out_4), opt_qaoa_made_update_log[i])
		fname_out_5 = sub_folder_path.joinpath(subsub_folder_name, 'fix_qaoa_made_update_log.npy')
		np.save(str(fname_out_5), fix_qaoa_made_update_log[i])

		fname_out_6 = sub_folder_path.joinpath(subsub_folder_name, 'learning_time.npy')
		np.save(str(fname_out_6), learning_time_result[i])
		fname_out_7 = sub_folder_path.joinpath(subsub_folder_name, 'predict_time.npy')
		np.save(str(fname_out_7), pred_time_result[i])
        
	path_config = sub_folder_path.joinpath(datename+'_runtime.txt')
	with open(str(path_config), mode='w') as f:
		f.write("total time [s] : {0}\n".format(end_time-start_time))
		f.write("======\n")
		f.write("beta : {0}\n".format(beta))
		f.write("n_step : {0}\n".format(n_step))
		f.write("======\n")
		f.write("training data size : {0}\n".format(n_train_list))
		f.write("hidden_size : {0}\n".format(hidden_size))
		f.write("hidden_layers : {0}\n".format(hidden_layers))
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
	n_spin = 25
	beta = 5.0

	# MADE
	n_train_list = [5,10,15,20,25,30,40,50,100,200,500,1000,5000,10000]
	n_subset = 10
	hidden_size = int(2 * n_spin)
	hidden_layers = 2
	batchsize = 8
	lr = 0.005
	n_epochs = 30

	# mcmc
	n_step = 10000

	# return
	result_dir_name = '../result/vs_train_data'

	# logger
	now = datetime.datetime.now()
	datename = now.strftime('%Y-%m%d-%H%M-%S')
	logging.basicConfig(level=logging.DEBUG,
	                    format="%(message)s",
	                    datefmt="[%X]",
	                    handlers=[logging.FileHandler(filename="../log/{0}_log.txt".format(datename))])
	logger = logging.getLogger(__name__)

	main()