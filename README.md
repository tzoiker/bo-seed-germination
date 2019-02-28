# bo-seed-germination
Source code for BO seed germination experiments

# Installation

* Up to date version is `v0.1`.
* If you do not have python on your system, install Python 3.X [Miniconda](https://docs.conda.io/en/latest/miniconda.html#miniconda).
* Setup Python 3.6 environment: with miniconda installed it can be done as follows
    * `conda create -n <env-name> python=3.6`
    * `conda activate <env-name>`
* Install dependencies: `pip install -r requirements.txt`
* Install the package: `pip install .`

# Running

* Create `.csv` file with initial data (or from all previous iterations).
It must contain names of parameters in the header with the column `target`
containing values of the target function (see example).
* Create `.csv` file with the values of bounds for the parameters (see example).
First two rows correspond to lower/upper bounds of original bounds.
Third and fourth rows correspond to lower/upper bounds of the scaled bounds (
if all parameters are considered "equivalent" in terms of scale, lower values
should be 0 and upper 1; if not, lower the upper bound is, lower is the influence
of the change in the value of this parameter).
* Create `.json` file with configuration parameters (see example).
Parameters that must be changed are `n_chambers` - number of sequentual predictions,
'Exploit' - exploitation parameter (see the paper), 0.1 degrees in the example.
`Explore` - exploration parameter (see the paper), 4 degrees in the example.
* Run `boseed --path-data=<path> --path-bounds=<path> --path-config=<path>`
* Some auxiliary information will be printed and user asked to do exploitation
or exploration for each sequential prediction.
If the proposed values of parameters are not considered good by the user,
they can be tried to be changed by entering `n` when prompted.
* More internal information is stored in `states/state.pickle` file, which
can be later opened using `dill` python package.
It is good to backup this file after each experiment.
