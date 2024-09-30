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
	# import instance
	fname_in = pathlib.Path(source_dir_name).joinpath('{0}_sites_instance.pickle'.format(n_spin))
	with open(str(fname_in), 'rb') as f:
		instance = pickle.load(f)

	# energy sort
	state = np.array([np.sum(ising.number_to_spin(i, n_spin)) for i in range(2**n_spin)], dtype=np.int8)
	energy = np.array([ising.spin_energy(state[i], instance) for i in range(2**n_spin)], dtype=np.float16)

	energy_sort_idx = np.argsort(energy, kind='quicksort')
 
	# export results
	fname_out_0 = pathlib.Path(result_dir_name).joinpath('{0}_sites_energy_sort_idx.npy'.format(n_spin))
	np.save(str(fname_out_0), energy_sort_idx)
  
if __name__ == '__main__':
	# seed
	seed = 1454
	rng = np.random.default_rng(seed)

	# instance
	source_dir_name = '../data'
	n_spin = 10

	# return
	result_dir_name = '../data'

	main()
  
