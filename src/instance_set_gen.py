import numpy as np
import datetime
import os
import sys
import pathlib
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import src.lib.made as MADE
import src.lib.mcmc_function as mcmc
import src.lib.ising_model as ising
from src.lib.ising_model import Ising_model
import src.lib.QAOA_function as qaoa
from src.lib.QAOA_function import QAOA_ansatz

seed = 1454
rng = np.random.default_rng(seed)

n_spin_min = 3
n_spin_max = 12
n_instance = 100

# instanceフォルダ作成
now = datetime.datetime.now()
datename = now.strftime('%Y-%m%d-%H%M-%S')
path = "../data/instance_set_{0}".format(datename)
folder_name = pathlib.Path(path) #Pathlibオブジェクトの生成
os.makedirs(str(folder_name))

# make instances for each size
for n_spin in np.arange(n_spin_min, n_spin_max+1):
    instance_set = []
    
    for i in range(n_instance):
        instance = Ising_model(n_spin, type='SK')
        instance_set.append(instance)
        
    fname = folder_name.joinpath('{0}_sites_instance_set.pickle'.format(n_spin))
    with open(str(fname), mode='wb') as f:
        pickle.dump(instance_set, f)