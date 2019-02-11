from argparse import ArgumentParser

from boseed.experiment import load_data, load_config, Parameters, Experiment
from boseed.propagation import scale_parameters
import numpy as np

parser = ArgumentParser()

parser.add_argument('--path-data', required=True)
parser.add_argument('--path-bounds', required=True)
parser.add_argument('--path-config', required=True)


def main():
	args = parser.parse_args()

	parameters, data_init_orig, data_init, orig_bounds, scaled_bounds = load_data(
		args.path_data, args.path_bounds
	)

	config = load_config(args.path_config)

	Eps = np.mean(scale_parameters(
		np.array(config.pop('Exploit')).reshape(1, -1),
		orig_bounds, scaled_bounds
	))
	Exploit = Explore = np.sum(scale_parameters(
		np.array(config.pop('Explore')).reshape(1, -1),
		orig_bounds, scaled_bounds
	))

	params = Parameters(
		**config,
		data_init=data_init, parameters=parameters,
		orig_bounds=orig_bounds, scaled_bounds=scaled_bounds,
		Eps=Eps, Exploit=Exploit, Explore=Explore,
		f_predict=None,
		fixed_params={},
		fixed_params_after_first_step=set([])
	)

	experiment = Experiment(params)
	experiment.prepare()
	print('\nOptimized kernel parameters\n')
	print(experiment.bo.bo.gp.kernel_)

	experiment.run(exploits=None, interactive=True)

if __name__ == "__main__":
	main()