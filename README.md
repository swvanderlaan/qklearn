# qklearn (CURRENTLY IN BETA!)
Tools for Parallelized Machine Learning using `sklearn` on a [qsub](http://pubs.opengroup.org/onlinepubs/009696799/utilities/qsub.html)-based High-Performance Cluster architecture

## Features
`qklearn` currently features:
- K-Fold cross-validation for classification (untested) and regression (tested) problems.
- Feature importances, and feature importance plots when estimators support this.

## Installation
Use pip to install `qklearn` from this repository:
```bash
pip install git+git://github.com/tbezemer/qklearn
```

## Usage
`qklearn` uses a configuration file to perform experiments. The file contains some basic parameters needed for the experiment.
It is essentially a tab/space separated file with a `parameter` name on the left (case insensitive) and a `value` on the right.
Spaces or tabs after the first space (separating the two main fields) will not be considered field separators (e.g. spaces in `value` are allowed)

```
#Comment lines may start with '#' or '//'. Comments cannot occur 'in-line'
//The project path is a required parameter. Always use ABSOLUTE paths to prevent errors.
project_path	/home/myproject
//The data_file parameter is also an ABSOLUTE path. The datafile should be a pickled Pandas DataFrame.
data_file	/home/some_data_file.pkl
//The experiment name will be converted to upper-case and non-filesystem safe characters are converted to underscores (below will become MY_COOL_ML_EXPERIMENT)
experiment_name	My cool ML experiment
//Specify the number of cross-validation folds using KCV:
KCV	30
//Specify the numer of parallel jobs to execute using n_jobs. Ideally, this is the number of cores on the compute node
n_jobs	12
//Specify the target variable in the data file. This is used
target_variable	my_output
```

You can also use the `MLConfig` object in the toolkit to programmatically define each of the properties, or to lead a config file and edit specific properties:

```python
from qklearn import MLConfig

C = MLConfig("path/to/my/config.txt")
C.experiment_name = "I changed my mind"

p = Pipeline([("Standard Scale", StandardScaler()), ("RF", RandomForestRegressor(n_estimators=30))])

execute_experiment_kfold(C, p)

```

You can run an experiment using a predefined config file (e.g. `config.txt`, following the rules described above) as follows:

```python
from qklearn import execute_experiment_kfold

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

p = Pipeline([("Standard Scale", StandardScaler()), ("RF", RandomForestRegressor(n_estimators=30))])

execute_experiment_kfold("path/to/my/config.txt", p)
```

If you want to further automate for instance a process of parameter optimization, you can load a config file, and use a loop, editing the config file to set up a new experiment, while keeping the other properties of the configuration:

```python
from qklearn import execute_experiment_kfold

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

C = MLConfig("path/to/my/config.txt")

for n_estimators in [10, 25, 50, 100, 250, 500]:

	p = Pipeline([("Standard Scale", StandardScaler()), ("RF", RandomForestRegressor(n_estimators=n_estimators))])
	C.experiment_name = "optimization_n={n}".format(ns=n_estimators)
	execute_experiment_kfold(C, p)


```

Training and Validation results are reported per fold, in each fold's folder, named `ML_RESULTS_[EXPERIMENT_NAME].csv`, so as to be collected and combined later on.
An example CSV result file:
```csv
fold,train_error,validation_error
fold0,0.035114868377720494,0.0351148683777205
```

Where possible, `qklearn` will extract and plot feature importances from the estimator. It will also attempt to automatically extract these from a `Pipeline` object.