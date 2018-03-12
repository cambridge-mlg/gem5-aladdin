import numpy as np
from numpy import random
import sys
import argparse
import itertools
sys.path.append("./")

from interface import evaluate
from optimizer.optimizer import optimize
from models.model_param import ModelParam
from optimizer.aquisition.aquisition_param import AquisitionParam

parser = argparse.ArgumentParser(description='Run Bayesian Optimization')
parser.add_argument('name', help='name of experiment')
parser.add_argument('n_eval', type=int, help='maximum number of evaluations', default=10)
parser.add_argument('parameters', nargs='*', help='tunable parameters')

args = parser.parse_args()
if args.parameters == []:
	args.parameters = ['tlb_hit_latency', 'tlb_miss_latency', 'tlb_page_size', 'tlb_entries', 'tlb_bandwidth', 'tlb_max_outstanding_walks']

param_sweeps={
	'cycle_time': range(1, 6),
	'pipelining': [0, 1],
	'enable_l2': [0, 1],
	'pipelined_dma': [0, 1],
	'tlb_entries': range(17),
	'tlb_hit_latency': range(1, 5),
	'tlb_miss_latency': range(10, 21),
	'tlb_page_size': [4096, 8192],
	'tlb_assoc': [4, 8, 16],
	'tlb_bandwidth': [1, 2],
	'tlb_max_outstanding_walks': [4, 8]
	# TODO add more
}

grid = np.array(list(itertools.product(*[param_sweeps[p] for p in args.parameters])))
print(grid)

eval_counter = 0

def f(x):
	global eval_counter
	params = {}
	for p, v in zip(args.parameters, x):
		params[p] = v
	cycle, power, area = evaluate(params, args.name, eval_counter)
	eval_counter += 1
	return np.array([cycle, power])    
	

model_params = {}
aquisition_params = {}

frontier, curve = optimize(f,
                           grid,
                           ModelParam('gp', model_params),
                           AquisitionParam('smsego', aquisition_params),
                           np.min((10, int(args.n_eval*0.1) + 1)),
                           args.n_eval,
                           np.array([2.0, 2.0]))

print('Final curve: {0}'.format(curve))
