# qklearn
Tools for Parallelized Machine Learning using `sklearn` on a [qsub](http://pubs.opengroup.org/onlinepubs/009696799/utilities/qsub.html)-based High-Performance Cluster architecture

## Installation
Use pip to install `qklearn` from this repository:
```pip install git+git://github.com/tbezemer/qklearn```

## Usage
`qklearn` uses a configuration file to perform experiments. The file contains some basic parameters needed for the experiment.
It is essentially a tab/space separated file with a `parameter` name on the left (case insensitive) and a `value` on the right.
Spaces or tabs after the first space (separating the two main fields) will not be considered field separators (e.g. spaces in `value` are allowed)

```
#Comment lines may start with '#' or '//'. Comments cannot occur 'in-line'
//The project path is a required parameter. Always use ABSOLUTE paths to prevent errors.
project_path	/home/myproject
//The data_file parameter is also an ABSOLUTE path
//The experiment name will be converted to upper-case and non-filesystem safe characters are converted to underscores (below will become MY_COOL_ML_EXPERIMENT)
experiment_name	My cool ML experiment
//Specify the number of cross-validation folds using KCV:
KCV	30
//Specify the numer of parallel jobs to execute using n_jobs. Ideally, this is the number of cores on the compute node
n_jobs	12
