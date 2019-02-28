from boseed.propagation import *
import warnings
import numpy as np
import pandas as pd
import json
import sklearn.gaussian_process.kernels


def load_data(path_data, path_bounds):
	data_init = pd.read_csv(path_data).astype(float).dropna().reset_index(drop=True)
	columns = data_init.columns.tolist()
	if 'target' not in columns:
		raise ValueError('target must be a clumn in data csv')
	parameters = list(columns)
	parameters.remove('target')

	bounds = pd.read_csv(path_bounds).astype(float).dropna().reset_index(drop=True)
	orig_bounds = np.array(bounds.iloc[:2, :]).reshape((2, -1)).T
	scaled_bounds = np.array(bounds.iloc[2:, :]).reshape((2, -1)).T

	data_init_orig = data_init.copy()
	print('Original bounds\n', bounds.iloc[:2, :].reset_index(drop=True))
	print('\nOriginal initial data\n')
	print(data_init_orig)
	data_init[parameters] = data_init[parameters].apply(lambda x: scale_parameters(x, orig_bounds, scaled_bounds), axis=1)
	print('\nScaled bounds\n', bounds.iloc[2:, :].reset_index(drop=True))
	print('\nScaled initial data')
	print(data_init.rename(columns={'target' : 'target-scaled'}))

	while True:
		inds = data_init.duplicated(subset=parameters)
		if sum(inds) == 0: break
		data_init.loc[inds, parameters] += 1e-12

	return parameters, data_init_orig, data_init, orig_bounds, scaled_bounds

def load_config(config_path):
	config = json.load(open('config.json'))
	for l in config['kernel_params']:
		for d in l:
			d['kernel'] = getattr(sklearn.gaussian_process.kernels, d['kernel'])
	return config


Parameters = namedtuple('Parameters', [
	'data_init', 'parameters',
	'Eps', 'Exploit', 'Explore',
	'orig_bounds', 'scaled_bounds',
	'kernel_params',
	'alg', 'max_eval', 'step',
	'random_state', 'n_chambers',
	'kappa0_exploit', 'kappa0_explore',
	'atol', 'rtol',
	'f_predict', 'optimize_kappa',
	'normalize_y',
	'fixed_params',
	'fixed_params_after_first_step'
])

def check_yes(inp):
	if inp in ['y', 'Y', 'yes', 'Yes']: return True
	else: return False
def check_no(inp):
	if inp in ['n', 'N', 'no', 'No']: return True
	else: return False

class ExperimentAction:
	Exploit = 'Exploit'
	Explore = 'Explore'
	Update = 'Update'
	
ExperimentEvent = namedtuple('ExperimentEvent', ['action', 'kappa', 'x'])
ExperimentState = namedtuple('ExperimentState', ['params', 'bo', 'history', 
												 'kappa_exploit_min', 'kappa_exploit_max', 'kappa_explore_min'])

class Experiment:
	def __init__(
		self, params, states_dir='./states', verbose=False, append_state=True
	):
		self.params = params
		self.states_dir = states_dir
		os.makedirs(states_dir, exist_ok=True)
		self.verbose=verbose
		self.history = []
		self.append_state = append_state
	
	def save_state(self):
		dr = self.states_dir
		state = ExperimentState(
			params=self.params,
			bo=self.bo,
			history=self.history,
			kappa_exploit_min=self.kappa_exploit_min,
			kappa_exploit_max=self.kappa_exploit_max,
			kappa_explore_min=self.kappa_explore_min
		)
		if self.append_state:
			fl = 'state.pickle'
			path = os.path.join(dr, fl)
			states = pickle.load(file=open(path, 'rb')) if os.path.exists(path) else []
			pickle.dump(file=open(path, 'wb'), obj=states + [state])
			states = pickle.load(file=open(path, 'rb'))
			return states
		else:
			fl = 'state_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.pickle'
			path = os.path.join(dr, fl)
			pickle.dump(file=open(path, 'wb'), obj=state)
			state = pickle.load(file=open(path, 'rb'))
			return state
	
	@staticmethod
	def load_state(path):
		raise NotImplementedError
		
	def prepare(self):
		params = self.params
		
		bo, fit_predict = prepare_bo(
			data_init=params.data_init,
			parameters=params.parameters,
			scaled_bounds=params.scaled_bounds,
			kernel_params=params.kernel_params,
			random_state=params.random_state,
			f_predict=params.f_predict,
			normalize_y=params.normalize_y,
			verbose=False
		)
		
		f_scale = lambda x, bnds1=params.orig_bounds, bnds2=params.scaled_bounds: scale_parameters(x, bnds1, bnds2)
		f_rescale = lambda x, bnds1=params.orig_bounds, bnds2=params.scaled_bounds: rescale_parameters(x, bnds1, bnds2)
		
		self.bo = BoWrapper(bo, f_scale, f_rescale)
		self.fit_predict = fit_predict
	
	def run(self, exploits=None, interactive=False, prepare=True, callback=None):
		assert exploits is not None or interactive
		interactive_exploit_explore = exploits is None 

		params = self.params
		if prepare: self.prepare()
		
		if interactive and exploits is None:
			def exploits(n=params.n_chambers):
				for i in range(n):
					while True:
						print(f'Chamber {i+1}: exploit? (y/n)')
						inp = input()
						if check_yes(inp): exploit = True
						elif check_no(inp): exploit = False
						else: continue
						yield exploit
						break
			exploits = exploits()
			
		states = []


		fixed_params = np.array([
			params.fixed_params[k] if k in params.fixed_params else np.nan
			for k in self.bo.bo.keys
		])
		self.bo.fixed_params = fixed_params

					
		for iter, exploit in enumerate(exploits):
			if iter == 1:
				fixed_params = np.array([
					x if k in params.fixed_params_after_first_step else np.nan
					for x,k in zip(self.bo.bo.space.X[-1], self.bo.bo.keys)
				])
				self.bo.fixed_params = fixed_params

			print("Exploiting..." if exploit else "Exploring...")
			
			print(
				pd.DataFrame(data=np.hstack((self.bo.f_rescale(self.bo.bo.space.X), self.bo.bo.space.Y.reshape((-1,1)), self.fit_predict)), 
							  columns=params.parameters + ['target', 'target predict mean', 'target predict std']).round(3)
			)
			if params.optimize_kappa:
				kappa_exploit_min, kappa_exploit_max, x_exploit_min, x_exploit_max, _ = \
					search_kappa_exploit(self.bo, params.Eps, params.Exploit, alg=params.alg, max_eval=params.max_eval, 
										kappa0=params.kappa0_exploit, step=params.step, verbose=self.verbose, atol=params.atol, rtol=params.rtol)

				assert kappa_exploit_min >=0 and kappa_exploit_max >= 0

				if self.bo.results[kappa_exploit_max]['k_1'] > params.Exploit and not np.isclose(self.bo.results[kappa_exploit_max]['k_1'], params.Exploit, atol=params.atol, rtol=params.rtol):
					warnings.warn(f"Exploit condition not met: setting to default {params.kappa0_exploit}")
					kappa_exploit_max = params.kappa0_exploit
				if self.bo.results[kappa_exploit_min]['k_inf'] < params.Eps and not np.isclose(self.bo.results[kappa_exploit_min]['k_inf'], params.Eps, atol=params.atol, rtol=params.rtol):
					warnings.warn(f"Eps condition not met: setting to default {params.kappa0_exploit}")
					kappa_exploit_min = params.kappa0_exploit
				# if kappa_exploit_min > kappa_exploit_max and not np.isclose(kappa_exploit_min, kappa_exploit_max, atol=params.atol, rtol=params.rtol): 
				# 	warnings.warn(f"k1 > k2: {round(kappa_exploit_min,3)} > {round(kappa_exploit_max,3)}: swapping")
				# 	kappa_exploit_min, kappa_exploit_max = kappa_exploit_max, kappa_exploit_min

				if not exploit:
					kappa_explore_min, _, x_explore_min, _, _ = search_kappa_explore(
						self.bo, Exploit=params.Exploit, Explore=params.Explore, alg=params.alg, max_eval=params.max_eval,
						kappa0=params.kappa0_explore, step=params.step, verbose=self.verbose, atol=params.atol, rtol=params.rtol)

					if self.bo.results[kappa_explore_min]['k_1'] < params.Explore:
						warnings.warn(f"Explore condition not met: setting to default {params.kappa0_explore}")
						kappa_explore_min = params.kappa0_explore

					# if self.bo.results[kappa_explore_min]['k_1'] < params.Explore and kappa_explore_min < kappa_exploit_max:
					# 	warnings.warn(f"k2 > k3: {round(kappa_exploit_max,3)} > {round(kappa_explore_min,3)}: swapping.")
					# 	kappa_exploit_max, kappa_explore_min = kappa_explore_min, kappa_exploit_max
					assert kappa_explore_min >= 0

					# if kappa_exploit_max > kappa_explore_min:
					# 	if kappa_exploit_max == params.kappa0_exploit:
					# 		warnings.warn(f"not k1 <= k2 <= k3: not {round(kappa_exploit_min,3)} {round(kappa_exploit_max,3)} <= {round(kappa_explore_min,3)}: changing")
					# 		kappa_explore_min = params.kappa0_explore
					# 	else:
					# 		warnings.warn(f"not k1 <= k2 <= k3: not {round(kappa_exploit_min,3)} {round(kappa_exploit_max,3)} <= {round(kappa_explore_min,3)}: reordering")
					# 		kappa_exploit_min, kappa_exploit_max, kappa_explore_min = np.sort([kappa_exploit_min, kappa_explore_min, kappa_exploit_max])

					# if np.isclose(kappa_exploit_max, kappa_explore_min, atol=params.atol, rtol=params.rtol):
					# 	warnings.warn(f"k2 == k3: {round(kappa_exploit_max,3)} == {round(kappa_explore_min,3)}: setting k3 = k2 + (k2-k1)*(k_explore_0 - k_exploit_0) / k_exploit_0")
					# 	kappa_explore_min = kappa_exploit_max + (kappa_exploit_max - kappa_exploit_min)*( (params.kappa0_explore -  params.kappa0_exploit)/ params.kappa0_exploit)
				else:
					kappa_explore_min, x_explore_min = None, None
			else:
				kappa_exploit_min = params.kappa0_exploit
				kappa_exploit_max = params.kappa0_exploit
				kappa_explore_min = params.kappa0_explore
			
			self.kappa_exploit_min = kappa_exploit_min
			self.kappa_exploit_max = kappa_exploit_max
			self.kappa_explore_min = kappa_explore_min

			print(f'k1, k2, k3: {kappa_exploit_min:.4f}, {kappa_exploit_max:.4f}, {kappa_explore_min if kappa_explore_min is not None else np.nan:.4f}')
					
			while True:
				if exploit:
					kappa, x = bo_exploit(self.bo, kappa_exploit_min, kappa_exploit_max, kappa_explore_min, 
										  params.Eps, params.Exploit, params.Explore
										 )

				else:
					kappa, x = bo_explore(self.bo, kappa_exploit_min, kappa_exploit_max, kappa_explore_min,
										  params.Eps, params.Exploit, params.Explore
										 )
				
				self.history.append(ExperimentEvent(
					action=ExperimentAction.Exploit if exploit else ExperimentAction.Explore,
					kappa=kappa,
					x=self.bo.f_rescale(x).squeeze()
				))

				if callback is not None: callback(self)
				print(f'Sampled kappa: {kappa}')
				print(f'Sampled x: {np.round(self.history[-1].x, decimals=2)}')
				if not interactive: break
				print('OK? y/n')
				if check_yes(input()): break
				else: continue
				
			if self.append_state:	
				states = self.save_state()
			else:
				states.append(self.save_state())
			self.fit_predict = self.bo.maximize_and_update(kappa)
			if callback is not None: callback(self)

		if callback is not None: callback(self)

		x_predict = np.array(list(map(lambda s: s.history[-1].x, states))).reshape((-1, len(params.parameters)))
		target_predict = self.bo.bo.gp.predict(self.bo.f_scale(x_predict))
		df = pd.DataFrame(np.hstack((x_predict, target_predict.reshape((-1,1)))), columns=params.parameters + ['target'])
		print(df.round(2))
		return states, df
	
