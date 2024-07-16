"""
Implements Masked AutoEncoder for Density Estimation, by Germain et al. 2015
Re-implementation by Andrej Karpathy based on https://arxiv.org/abs/1502.03509
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import ising_model as ising

# ------------------------------------------------------------------------------

class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)        
        self.register_buffer('mask', torch.ones(out_features, in_features))
        
    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))
        
    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)

class MADE(nn.Module):
    def __init__(self, nin, hidden_sizes, nout, num_masks=1, natural_ordering=False, seed=0):
        """
        nin: integer; number of inputs
        hidden sizes: a list of integers; number of units in hidden layers
        nout: integer; number of outputs, which usually collectively parameterize some kind of 1D distribution
              note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
              will be all the means and the second nin will be stds. i.e. output dimensions depend on the
              same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
              the output of running the tests for this file makes this a bit more clear with examples.
        num_masks: can be used to train ensemble over orderings/connections
        natural_ordering: force natural ordering of dimensions, don't use random permutations
        """
        
        super().__init__()
        self.nin = nin
        self.nout = nout
        self.hidden_sizes = hidden_sizes
        assert self.nout % self.nin == 0, "nout must be integer multiple of nin"
        
        # define a simple MLP neural net
        self.net = []
        hs = [nin] + hidden_sizes + [nout]
        for h0,h1 in zip(hs, hs[1:]):
            self.net.extend([
                    MaskedLinear(h0, h1),
                    nn.LeakyReLU(), # change ReLU to LeakyReLU
                ])
        self.net.pop() # pop the last ReLU for the output layer
        #self.net.append(nn.Sigmoid()) # add the sigmoid for the output layer
        self.net = nn.Sequential(*self.net)
        
        # seeds for orders/connectivities of the model ensemble
        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = seed # for cycling through num_masks orderings
        
        self.m = {}
        self.update_masks() # builds the initial self.m connectivity
        # note, we could also precompute the masks and cache them, but this
        # could get memory expensive for large number of masks.
        
    def update_masks(self):
        if self.m and self.num_masks == 1: return # only a single seed, skip for efficiency
        L = len(self.hidden_sizes)
        
        # fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks
        
        # sample the order of the inputs and the connectivity of all neurons
        self.m[-1] = np.arange(self.nin) if self.natural_ordering else rng.permutation(self.nin)
        for l in range(L):
            self.m[l] = rng.randint(self.m[l-1].min(), self.nin-1, size=self.hidden_sizes[l])
        
        # construct the mask matrices
        masks = [self.m[l-1][:,None] <= self.m[l][None,:] for l in range(L)]
        masks.append(self.m[L-1][:,None] < self.m[-1][None,:])
        
        # handle the case where nout = nin * k, for integer k > 1
        if self.nout > self.nin:
            k = int(self.nout / self.nin)
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]]*k, axis=1)
        
        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l,m in zip(layers, masks):
            l.set_mask(m)
    
    def forward(self, x):
        return self.net(x)
    
# training function
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_epoch(model, split, dataset, opt, seed=None):
	if seed != None:
		set_seed(seed)
    
	torch.set_grad_enabled(split=='train') # enable/disable grad for efficiency of forwarding test batches

	model.train() if split == 'train' else model.eval()
	x = dataset
	for i, xb in enumerate(x):
		# get the logits, potentially run the same batch a number of times, resampling each time
		model.update_masks()

		# forward the model
		logits = model(xb)

		# evaluate the binary cross entropy loss
		criterion = nn.BCEWithLogitsLoss()
		loss = criterion(logits, xb)

		# backward/update
		if split == 'train':
			opt.zero_grad()
			loss.backward()
			opt.step()

	return loss

def run_train(model, train_data, test_data, n_epochs, opt, scheduler=None, seed=None):
	if seed != None:
		set_seed(seed)

	train_loss = []
	test_loss = []

	for epoch in range(n_epochs):
		test_loss.append(run_epoch(model, 'test', test_data, opt))
		train_loss.append(run_epoch(model, 'train', train_data, opt))
		if scheduler:
			scheduler.step()
	
	test_loss.append(run_epoch(model, 'test', test_data, opt))

	return train_loss, test_loss

# sampling
def predict(model, inputs: np.ndarray):
    n = model.nin
    
    # convert ndarray to torch.tensor
    inputs_th = torch.from_numpy(inputs.copy()).to(dtype=torch.float32)
    
    # apply model and sampling
    logits = model(inputs_th)
    outputs_th = torch.bernoulli(torch.sigmoid(logits))
    outputs = outputs_th.detach().numpy()
    
    return outputs

"""
def sampling_MADE(model):
    # natural_orderung = Trueとしているため、条件付き確率積の順番は昇順になっている前提！
    # ex.) p(x1,x2,x3) = p(x3|x1,x2)p(x2|x1)p(x1)
    n = model.nin
    pred_dist = np.zeros(2**n)

    for i in range(2**n):
        bina = number_to_binary(i, n)
        bina_th = torch.from_numpy(bina.copy())
        bina_th = bina_th.float()
        pred_th = torch.sigmoid(model(bina_th))

        pred = np.zeros(n)
        for j in range(n):
            if bina[j] == 1:
                pred[j] = pred_th.detach().numpy().copy()[j]
            else:
                pred[j] = 1 - pred_th.detach().numpy().copy()[j]

        pred_dist[i] = np.prod(pred)
        
    return pred_dist
"""

def compute_log_prob(model, inputs: np.ndarray):
	n = model.nin

	# convert ndarray to torch.tensor
	inputs_th = torch.from_numpy(inputs.copy()).to(dtype=torch.float32)
    
    # compute the (log) probability of outputs
	logits = model(inputs_th)
	criterion = nn.BCEWithLogitsLoss(reduction='none')
	log_prob = -1.0 * criterion(logits, inputs_th)
	prob = log_prob.sum(dim=-1)

	return prob.detach().numpy()

def number_to_binary(number, n_dim):
    bin = format(number,"b").zfill(n_dim)
    bin_list = np.array([0 if bin[i]=='0' else 1 for i in range(n_dim)])
    bin_list = np.flipud(bin_list)
    
    return bin_list

def binary_to_number(binary_list):
	n_dim = binary_list.shape[0]

	return int(sum([(2**i)*binary_list[i] for i in range(n_dim)]))

def spin_to_binary(spin_list):
	idx = ising.spin_to_number(spin_list)

	return number_to_binary(idx, spin_list.shape[0])

def binary_to_spin(binary_list):
	idx = binary_to_number(binary_list)

	return ising.number_to_spin(idx, binary_list.shape[0])