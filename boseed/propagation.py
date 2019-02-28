import sys
sys.path.append('./')
from bayes_opt import BayesianOptimization, helpers
from bayes_opt.target_space import TargetSpace
import numpy as np
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import pandas as pd
from scipy.optimize import NonlinearConstraint
from scipy.optimize import fmin_slsqp, minimize
# import matplotlib.pyplot as plt
import time
import dill as pickle
from collections import namedtuple
import datetime
import os
import warnings

def scale_parameters(x, bnds1, bnds2):
	return (x - bnds1[:, 0])/(bnds1[:, 1] - bnds1[:, 0])*(bnds2[:, 1] - bnds2[:, 0]) + bnds2[:, 0]

def rescale_parameters(x, bnds1, bnds2):
	return (x - bnds2[:, 0])*(bnds1[:, 1] - bnds1[:, 0]) / (bnds2[:, 1] - bnds2[:, 0]) + bnds1[:, 0]

def args_dict_to_array(bo, **kwargs):
	return np.array(list(map(lambda key: kwargs[key], bo.space.keys)))

def predict_gp(bo, **kwargs):
	x = args_dict_to_array(bo=bo, **kwargs)
	return bo.gp.predict(np.array(x).reshape((1,-1)), return_std=True)

def prepare_bo(data_init, parameters, scaled_bounds, kernel_params, random_state=1337, verbose=True, f_predict=None, normalize_y=False): 
	bounds = {param : scaled_bound for param, scaled_bound in zip(parameters, scaled_bounds)}
	bo = BayesianOptimization(None, pbounds=bounds, random_state=random_state, verbose=verbose)

	bo.gp.kernel = np.sum(list(map(lambda add: np.prod(list(map(lambda mul: mul['kernel'](**mul['params']), add))), kernel_params)))
	bo.gp.n_restarts_optimizer=30
	bo.gp.alpha = 0.0
	bo.gp.normalize_y = normalize_y

	f_predict = (lambda bo=bo, f_predict=f_predict, **kwargs: f_predict(args_dict_to_array(bo=bo, **kwargs))) if f_predict is not None else (lambda bo=bo, **kwargs: predict_gp(bo, **kwargs)[0][0])
	bo.space = TargetSpace(f_predict, pbounds=bo.pbounds, random_state=bo.random_state)

	bo.initialize_df(data_init)
	
	bo.maximize(init_points=0, n_iter=0, kappa=0, fit=True)
	fit_predict = np.array(list(map(lambda x: predict_gp(bo, **data_init.iloc[x,:][parameters].to_dict()), range(len(data_init)))))
	return bo, fit_predict.squeeze()
	
def check_kappas(kappas, bo_kwargs):
	X = bo_kwargs['data_init']
	bo, fit_predict = prepare_bo(**bo_kwargs, verbose=False)
	xs = []
	acqs = []
	for i in tqdm(range(len(kappas))): 
		kappa = kappas[i]
		x_new = bo.maximize(init_points=0, n_iter=1, kappa=kappa, fit=False, update=False)
		xs.append(x_new.item())
		acqs.append(bo.util.utility(xs[-1].reshape((1,-1)), bo.gp, bo.space.Y.max()))
	xs = np.array(xs)
	acqs = np.array(acqs)

	return kappas, xs, acqs, X, fit_predict

def get_exploit_ineq_constraints(bo, Eps, Exploit, include_bounds=False):
	constraints = [
		lambda kappa, bo=bo, Eps=Eps: bo.maximize(kappa)['k_inf'] - Eps,
		lambda kappa, bo=bo, Exploit=Exploit: Exploit - bo.maximize(kappa)['k_1']
	]
	if include_bounds: constraints.append(lambda kappa: kappa)
	return constraints

def get_explore_ineq_constraints(bo, Exploit, Explore, include_bounds=False):
	constraints = [
		lambda kappa, bo=bo, Explore=Explore: bo.maximize(kappa)['k_1'] - Explore, 
		lambda kappa, bo=bo, Exploit=Exploit: bo.maximize(kappa)['k_1'] - Exploit,
	]
	if include_bounds: constraints.append(lambda kappa: kappa)
	return constraints

def check_exploit_ineq_constraints(kappa, bo, Eps, Exploit, include_bounds=False, rtol=1e-5, atol=1e-8):
	assert Eps <= Exploit
	constraints = get_exploit_ineq_constraints(bo, Eps, Exploit, include_bounds)
	return all(map(lambda c: c(kappa) >= 0 or np.isclose(c(kappa), 0, rtol=rtol, atol=atol), constraints))

def check_some_exploit_ineq_constraints(kappa, bo, Eps, Exploit, include_bounds=False, rtol=1e-5, atol=1e-8, nums=None):
	assert Eps <= Exploit
	constraints = get_exploit_ineq_constraints(bo, Eps, Exploit, include_bounds)
	if nums is None: nums = range(len(constraints))
	return all(map(lambda c: c(kappa) >= 0 or np.isclose(c(kappa), 0, rtol=rtol, atol=atol), [cons for i, cons in constraints if i in nums]))
	

def check_explore_ineq_constraints(kappa, bo, Exploit, Explore, include_bounds=False, rtol=1e-5, atol=1e-8):
	assert Exploit <= Explore
	constraints = get_explore_ineq_constraints(bo, Exploit, Explore, include_bounds)
	return all(map(lambda c: c(kappa) >= 0 or np.isclose(c(kappa), 0, rtol=rtol, atol=atol), constraints))

def optimize_kappa(bo, Eps=None, Exploit=None, Explore=None, max_eval=100, kappa0=None,
		verbose=True, step=1e-4, alg='SLSQP', minimize_kappa=False):
	assert (Eps is not None and Exploit is not None) or (Exploit is not None and Explore is not None)
	assert alg in ['SLSQP', 'COBYLA']
	assert kappa0 is not None
	exploit = Eps is not None and Exploit is not None 
	if exploit: assert Eps < Exploit
	
	assert isinstance(bo, BoWrapper)
#     if not isinstance(bo, BoWrapper): bo = BoWrapper(bo)
	max_iter = int((max_eval-1)/3)
	
	if minimize_kappa is not None:
		func = lambda kappa, minimize_kappa=minimize_kappa: kappa if minimize_kappa else -kappa
	else:
		func = lambda kappa: 0
	
	x0 = np.array([kappa0])
	if exploit: f_ieqcons = get_exploit_ineq_constraints(bo, Eps, Exploit)
	else: f_ieqcons = get_explore_ineq_constraints(bo, Exploit, Explore)
	constraints = tuple(map(lambda x: {
			'type' : 'ineq',
			'fun' : x
		}, f_ieqcons))
	
	kappas = [x0[0]]

	def save_step(x, kappas=kappas):
		kappas.append(x[0])
	def func(kappa, func=func):
		if np.isnan(kappa):
			raise ValueError('kappa is nan')
		save_step(kappa)
		return func(kappa)

	if alg == 'SLSQP':
		bounds = [(0,np.inf)]
		def callback(x, verbose=verbose):
			if verbose:
				print(f'Found new kappa: {x[0]}')
	elif alg == 'COBYLA':
		bounds = None
		constraints += ({
			'type' : 'ineq',
			'fun' : lambda kappa: kappa
		},)
		callback = None


	time_start = time.time()
	options = {'maxiter' : max_iter, 'disp' : 1}
	if alg == 'SLSQP': options['eps'] = step
	try:
		opt_kappa = minimize(fun=func, x0=x0, method=alg, bounds=bounds, constraints=constraints, options=options, callback=callback).x[0]
	except ValueError as e:
		if e.args[0] == 'kappa is nan':
			warnings.warn('Found nan kappa while optimizing: skipping.')
			opt_kappa = kappas[-1]
		else:
			raise e
#     fmin_slsqp(func=func, x0=x0, ieqcons=f_ieqcons, bounds=[(0,np.inf)], 
#                         iter=max_iter, callback=callback, epsilon=step, iprint = 2 if verbose else 1)
	time_end = time.time()
	
	try: kappas.remove(opt_kappa)
	except ValueError as e:
		if any(np.isclose(kappas, opt_kappa)):
			print(f'opt_kappa = {opt_kappa} is not in kappas = {kappas}, but very close ...')
			bo.maximize(opt_kappa)
			assert opt_kappa in bo.results
		else:
			print('kappas', kappas)
			print('opt_kappa', opt_kappa)
			raise e
	kappas.append(opt_kappa)
	
	if exploit:
		ks_1 = list(map(lambda kappa: bo.results[kappa]['k_1'], kappas))
		ks_inf = list(map(lambda kappa: bo.results[kappa]['k_inf'], kappas))
		ks = np.array((ks_1, ks_inf)).T
	else:
		ks = np.array(list(map(lambda kappa: bo.results[kappa]['k_1'], kappas))).reshape((-1,1))

	return np.array(kappas), np.array(ks), time_end - time_start, bo, 

class BoWrapper():
	def __init__(self, bo, f_scale, f_rescale):
		self.bo = bo
		self.results = {}
		self.reset_counters()
		self.f_scale = f_scale
		self.f_rescale = f_rescale
		self.fixed_params = None
		
	def reset_counters(self):
		self.cache_misses = 0
		self.cache_hits = 0
	
	def maximize_and_update(self, kappa):
		self.bo.maximize(init_points=0, n_iter=1, kappa=kappa, fit=False, update=True, fixed_params=self.fixed_params)
		fit_predict = np.array(self.bo.gp.predict(self.bo.space.X, return_std=True))
		self.results = {}
		self.reset_counters()
		return fit_predict.T
	
	def maximize(self, kappa):
		try: len(kappa)
		except: kappa = np.array([kappa])
		bo = self.bo
		key = kappa[0]
		if key in self.results:
			self.cache_hits += 1
			return self.results[key]
		else:
			self.cache_misses += 1
			x = np.array(bo.maximize(init_points=0, n_iter=1, kappa=kappa, fit=False, update=False, fixed_params=self.fixed_params))
			X = bo.space.X
			acq = bo.util.utility(x, bo.gp, bo.space.Y.max())

			f_constraint_1_min = lambda D: np.linalg.norm(D, ord=1, axis=1).min(axis=0)
			f_constraint_1_mean_min = lambda D: np.linalg.norm(D, ord=1, axis=1).min(axis=0) / D.shape[1]
			f_constraint_inf_min = lambda D: np.linalg.norm(D, ord=np.inf, axis=1).min(axis=0)
			X1 = np.einsum('j,i->ij', x.flatten(), np.ones(X.shape[0]))
			X2 = X
			D = X1 - X2
			k_1 = f_constraint_1_min(D)
			k_1_mean = f_constraint_1_mean_min(D)
			k_inf = f_constraint_inf_min(D)

			self.results[key] = {'x' : x, 'acq' : acq, 'k_1' : k_1, 'k_1_mean' : k_1_mean, 'k_inf' : k_inf}
			return self.results[key]
		
def plot_kappa_search(bo, kappas_exploit=None, kappas_explore=None, Eps=None, Exploit=None, Explore=None, 
					  t_exploit=None, t_explore=None):
	plt.figure(figsize=(6,5))
	exploit = kappas_exploit is not None
	explore = kappas_explore is not None
	
	kappas = np.array(list(bo.results.keys()))
	ks = np.array(list(map(lambda x: bo.results[x]['k_1'], kappas)))
	inds = np.argsort(kappas)
	x = kappas[inds]
	y = ks[inds]

	if exploit:
		inds = np.argsort(kappas_exploit)
		x1 = kappas_exploit[inds]
		y1_1 = np.array(list(map(lambda x: bo.results[x]['k_1'], x1)))
		y1_inf = np.array(list(map(lambda x: bo.results[x]['k_inf'], x1)))
	
	if explore:
		inds = np.argsort(kappas_explore)
		x2 = kappas_explore[inds]
		y2_1 = np.array(list(map(lambda x: bo.results[x]['k_1'], x2)))
		
	
	plt.rc('text', usetex=True)
	plt.plot(x, y, '-b')
	if exploit: 
		plt.plot(x1, y1_1, 'x', color='blue', label='Exploit $|\cdot|_1$', markersize=10)
		plt.plot(x1, y1_inf, 'x', color='orange', label='Exploit $|\cdot|_\infty$', markersize=10)
	if explore: 
		plt.plot(x2, y2_1, '+r', label='Explore $|\cdot|_1$', markersize=14)
	# plt.plot(kappas, ks, '-o', label='constraint')
	if exploit:
		plt.plot([kappas.min(), kappas.max()], [Eps]*2, ':', color='black', label=r'$\epsilon$')
		plt.plot([kappas.min(), kappas.max()], [Exploit]*2, '-.', color='black', label=r'$\epsilon_{xploit}$')
	if explore:
		plt.plot([kappas.min(), kappas.max()], [Explore]*2, '--', color='black', label=r'$\epsilon_{xplore}$')
	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)
	plt.xlabel('$\kappa$', fontsize=24)
	plt.ylabel('Constraint', fontsize=20)
	# plt.semilogx()
	plt.semilogy()
	# plt.xlim([kappas[0],10])
	plt.legend(fontsize=18, bbox_to_anchor=(1.6,0.85))
#     plt.tight_layout()
	plt.show()        

def search_kappa_exploit(bo_wrapped, Eps, Exploit, kappa0, alg='SLSQP', max_eval=19, step=1e-4, verbose=True, atol=1e-8, rtol=1e-5):

	if check_exploit_ineq_constraints(kappa0, bo_wrapped, Eps, Exploit, include_bounds=True, atol=atol, rtol=rtol):
		return kappa0, kappa0, \
			bo_wrapped.f_rescale(bo_wrapped.results[kappa0]['x']), \
			bo_wrapped.f_rescale(bo_wrapped.results[kappa0]['x']), \
			(None, None, None, None, None, None, None)

	kappas0, _, t0, _ = optimize_kappa(
		bo_wrapped, Eps=Eps, Exploit=Exploit, 
		max_eval=max_eval, kappa0=kappa0, verbose=verbose, 
		step=step, alg=alg, minimize_kappa=None
	)
	kappas_min, ks_min, t_min, _ = optimize_kappa(
		bo_wrapped, Eps=Eps, Exploit=Exploit, 
		max_eval=max_eval, kappa0=kappas0[-1], verbose=verbose, 
		step=step, alg=alg, minimize_kappa=True
	)
	# kappas_max, ks_max, t_max, _ = optimize_kappa(
	# 	bo_wrapped, Eps=Eps, Exploit=Exploit, 
	# 	max_eval=max_eval, kappa0=kappas0[-1], verbose=verbose, 
	# 	step=step, alg=alg, minimize_kappa=False
	# )
	kappas_max, ks_max, t_max = kappas_min, ks_min, t_min

	kappas = bo_wrapped.results.keys()
	kappas_satisfy = list(filter(lambda kappa: check_exploit_ineq_constraints(kappa, bo_wrapped, Eps, Exploit, include_bounds=True, atol=atol, rtol=rtol), kappas))

	if len(kappas_satisfy) == 0:
		warnings.warn(f"Exploit: no feasible solution: setting k1 = k2 = kappa0 = {kappa0}")
		kappa_min = kappa_max = kappa0
	else:
		kappa_min = min(kappas_satisfy)
		# kappa_max = max(kappas_satisfy)
		kappa_max = kappa_min

	if np.isclose(kappa_min, 0, atol=atol, rtol=rtol): kappa_min = 0
	if np.isclose(kappa_max, 0, atol=atol, rtol=rtol): kappa_max = 0
	bo_wrapped.maximize(kappa_min)
	bo_wrapped.maximize(kappa_max)

	print(f'Exploit: finding kappa0 = {round(kappas0[-1],3)} took {round(t0/60, 2)} minutes')
	print(f'Exploit: finding kappa = {round(kappa_min,3)} took {round(t_min/60, 2)} minutes')
	print(f'Exploit: finding kappa = {round(kappa_max,3)} took {round(t_max/60, 2)} minutes')
	
	# kappas = np.hstack((kappas0.flatten(), kappas_min.flatten()))
	# ks = ks_min
	kappas = np.hstack((kappas0.flatten(), kappas_max.flatten(), kappas_min.flatten()))
	ks = np.vstack((ks_max, ks_min))
	t = t0 + t_min + t_max
	
	# plot_kappa_search(bo_wrapped, kappas_exploit=kappas, Eps=Eps, Exploit=Exploit, t_exploit=t)

	# return kappa_min, None, \
	# 	bo_wrapped.f_rescale(bo_wrapped.results[kappa_min]['x']), \
	# 	None, \
	# 	(kappas0, kappas_min, None, ks_min, None, t_min, None)
	
	return kappa_min, kappa_max, \
		bo_wrapped.f_rescale(bo_wrapped.results[kappa_min]['x']), \
		bo_wrapped.f_rescale(bo_wrapped.results[kappa_max]['x']), \
		(kappas0, kappas_min, kappas_max, ks_min, ks_max, t_min, t_max)

def search_kappa_explore(bo_wrapped, Exploit, Explore, kappa0, alg='SLSQP', max_eval=19, step=1e-4, verbose=True, atol=1e-8, rtol=1e-5):

	# if check_explore_ineq_constraints(kappa0, bo_wrapped, Exploit, Explore, include_bounds=True, rtol=rtol, atol=atol):
	# 	return kappa0, None, \
	# 		bo_wrapped.f_rescale(bo_wrapped.results[kappa0]['x']), \
	# 		None, \
	# 		(None, None, None, None, None, None, None)


	kappas0, _, t0, _ = optimize_kappa(
		bo_wrapped, Exploit=Exploit, Explore=Explore, 
		max_eval=max_eval, kappa0=kappa0, verbose=verbose, step=step, alg=alg,
		minimize_kappa=None
	)
	kappas_min, ks, t, _ = optimize_kappa(
		bo_wrapped, Exploit=Exploit, Explore=Explore, 
		max_eval=max_eval, kappa0=kappas0[-1], verbose=verbose, step=step, alg=alg,
		minimize_kappa=True
	)
	kappas = bo_wrapped.results.keys()
	kappas_satisfy = list(filter(lambda kappa: check_explore_ineq_constraints(kappa, bo_wrapped, Exploit, Explore, include_bounds=True, atol=atol, rtol=rtol), kappas))

	if len(kappas_satisfy) == 0:
		warnings.warn(f"Explore: no feasible solution: setting k3 = kappa0 = {kappa0}")
		kappa_min = kappa0
	else:
		kappa_min = min(kappas_satisfy)

	if np.isclose(kappa_min, 0, atol=atol, rtol=rtol): kappa_min = 0
	bo_wrapped.maximize(kappa_min)


	kappas = np.hstack((kappas0.flatten(), kappas_min.flatten()))

	print(f'Explore: finding kappa0 = {round(kappas0[-1],3)} took {round(t0/60, 2)} minutes')
	print(f'Explore: finding kappa = {round(kappa_min,3)} took {round(t/60, 2)} minutes')
	# plot_kappa_search(bo_wrapped, kappas_explore=kappas, Explore=Explore, t_explore=t)
	
	return kappa_min, None, bo_wrapped.f_rescale(bo_wrapped.results[kappa_min]['x']), None, \
		(kappas0, kappas_min, None, ks, None, t, None)
	
def sample_kappa_exploit(k1, k2, k3):
	# return np.random.uniform(k1, k2, 1)[0]
	return k1

def sample_kappa_explore(k1, k2, k3):
	# return k2 + np.random.exponential(k3-k2, 1)[0]
	return k3
	# return k3

def bo_exploit(bo_wrapped, k1, k2, k3, Eps, Exploit, Explore):
	kappa_exploit = sample_kappa_exploit(k1, k2, k3)
	res_exploit = bo_wrapped.maximize(kappa_exploit)

	conditions = [
		Eps < res_exploit['k_inf'] or np.isclose(Eps,res_exploit['k_inf']),
		Exploit > res_exploit['k_1'] or np.isclose(Exploit, res_exploit['k_1'])
	]

	print(f"Exploit: kappa={kappa_exploit}" )
	print(f"Eps < ||_inf: {round(Eps, 5)} < {round(res_exploit['k_inf'], 5)}: {conditions[0]}")
	print(f"Exploit > ||_1: {round(Exploit, 5)} > {round(res_exploit['k_1'], 5)}: {conditions[1]}")
	print(f'Exploit: conditions OK: {all(conditions)}')
	
	print('Exploit: average change of every T and W: {:.2f} oC, {:.2f} ml'.format(
		*bo_wrapped.f_rescale(np.ones(bo_wrapped.bo.space.X.shape[1])*res_exploit['k_1_mean'])[[0,-1]]
	))

	return kappa_exploit, res_exploit['x']
	
def bo_explore(bo_wrapped, k1, k2, k3, Eps, Exploit, Explore):
	kappa_explore = sample_kappa_explore(k1, k2, k3)
	res_explore = bo_wrapped.maximize(kappa_explore)
	
	conditions = [
		Exploit < res_explore['k_1'] or np.isclose(Exploit, res_explore['k_1'])
	]
	
	print(f"Explore: kappa={kappa_explore}" )
	print(f"Explore < ||_1: {round(Exploit, 5)} < {round(res_explore['k_1'], 5)}: {conditions[0]}")
	print(f'Explore: conditions OK: {all(conditions)}')

	print('Explore: average change of every T and W: {:.2f} oC, {:.2f} ml'.format(
		*bo_wrapped.f_rescale(np.ones(bo_wrapped.bo.space.X.shape[1])*res_explore['k_1_mean'])[[0,-1]]
	))
	
	return kappa_explore, res_explore['x']

