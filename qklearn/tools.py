import matplotlib
matplotlib.use('Agg')

from qklearn.funcs import _initialize_experiment, _do_fold, _distribute_estimator, _distribute_metric, _extract_feature_importances, _collect_results, _collect_importances


class MLConfig:

	_config_dict = {}

	@property
	def KCV(self):
		return int(self._config_dict['kcv']) if "kcv" in self._config_dict else None

	@KCV.setter
	def KCV(self, kcv):
		self._config_dict['kcv'] = kcv

	@property
	def project_path(self):
		return self._config_dict['project_path']

	@project_path.setter
	def project_path(self, project_path):
		self._config_dict['project_path'] = project_path

	@property
	def qsub_mail(self):
		return self._config_dict['qsub_mail'] if 'qsub_mail' in self._config_dict else False

	@qsub_mail.setter
	def qsub_mail(self, qsub_mail):
		self._config_dict['qsub_mail'] = qsub_mail

	@property
	def qsub_mem(self):
		return self._config_dict['qsub_mem'] if 'qsub_mem' in self._config_dict else "20G"

	@qsub_mem.setter
	def qsub_mem(self, qsub_mem):
		self._config_dict['qsub_mem'] = qsub_mem

	@property
	def data_file(self):
		return self._config_dict['data_file']

	@data_file.setter
	def data_file(self, data_file):
		self._config_dict['data_file'] = data_file

	@property
	def config_path(self):
		return self._config_dict['config_path']

	@config_path.setter
	def config_path(self, config_path):
		self._config_dict['config_path'] = config_path

	@property
	def target_variable(self):
		return self._config_dict['target_variable'] if 'target_variable' in self._config_dict else False

	@target_variable.setter
	def target_variable(self, target_variable):
		self._config_dict['target_variable'] = target_variable

	@property
	def experiment_name(self):
		remove_punctuation_map = dict((ord(char), "_") for char in ' \/*?:"<>|')

		return self._config_dict['experiment_name'].upper().translate(remove_punctuation_map)

	@experiment_name.setter
	def experiment_name(self, experiment_name):
		self._config_dict['experiment_name'] = experiment_name

	@property
	def n_jobs(self):
		return self._config_dict['n_jobs'] if 'n_jobs' in self._config_dict else 1

	@n_jobs.setter
	def n_jobs(self, n_jobs):
		self._config_dict['n_jobs'] = n_jobs

	def __str__(self):

		return """PROJECT_PATH={0}
DATA_FILE={1}
KCV={2}""".format(self.project_path, self.data_file, self.KCV)


	def __init__(self, *args, **kwargs):

		from os import path, sep

		def _newline_cleanup(s):
			if "\n" in s: s = s.replace("\r", "");
			else: s = s.replace("\r", "\n");
			return s
		def _whitespace_cleanup(s):
			while "  " in s: s = s.replace("  ", " ");
			s = s.replace(' ', '\t')
			while "\t\t" in s: s = s.replace("\t\t", "\t");
			return s

		if len(args)== 1:
			with open(args[0], "r") as config_file:

				config_file_contents = _newline_cleanup(config_file.read())

				for line in config_file_contents.split('\n'):
					line = line.strip()
					if line.startswith("#") or line.startswith("//"): continue;
					line = _whitespace_cleanup(line)
					fields = line.split('\t')
					if len(fields) != 2:

						fields = [fields[0], " ".join(fields[1:])]
						

					fields[0] = fields[0].lower()
					
					self._config_dict[fields[0]] = fields[1]


			if self.data_file == None:
				raise ValueError("Incorrect configuration! data_file must be set!")
			elif self.project_path == None:
				raise ValueError("Incorrect configuration! project_path must be set!")
			elif self.experiment_name == None:
				raise ValueError("Incorrect configuration! experiment_name must be set!")

			self._config_dict['config_path'] = args[0]
			
		elif "data_file" in kwargs and "project_path" in kwargs and "experiment_name" in kwargs:

			for param, value in kwargs.items():

				self._config_dict[param] = value

			self._config_dict['project_path'] = path.join(self.project_path, self.experiment_name).replace('\\', sep).replace('/', sep)
			self._config_dict['config_path'] = False
		else:

			raise ValueError("Incorrect Config initialization. Please refer to the documentation.")

def create_kfold_cv(CONFIG):
	
	import pandas as pd
	from joblib import Parallel, delayed

	if CONFIG.KCV == None:
		raise ValueError("KCV parameter of the MLConfig object cannot be left unspecified for this function!")
	if not CONFIG.target_variable:
		raise ValueError("Target variable parameter of the MLConfig object cannot be left unspecified for this function!")

	from sklearn.model_selection import KFold
	
	print("- Preparing {0}-fold cross-validation for parallelization".format(CONFIG.KCV))
	print("\t* Reading data file")
	df = pd.read_pickle(CONFIG.data_file).dropna()
	
	INPUT = df[[col for col in df.columns.values if col != CONFIG.target_variable]]
	OUTPUT = df[CONFIG.target_variable]
	print("\t* Creating folds")

	_ = Parallel(n_jobs=-1,max_nbytes=None)( delayed(_do_fold)(train, test, i, CONFIG.KCV, INPUT, OUTPUT, CONFIG.project_path) for (train, test), i in zip(KFold(CONFIG.KCV).split(INPUT), range(0, CONFIG.KCV)) )

def execute_experiment_kfold(CONFIG, estimator, metric=False):

	if isinstance(CONFIG, str): CONFIG = MLConfig(CONFIG);
	_initialize_experiment(CONFIG)

	from sklearn.pipeline import Pipeline
	from sklearn.metrics import mean_squared_error, accuracy_score
	from glob import glob
	import pandas as pd
	from joblib import Parallel, delayed
	from os import path, system
	from sys import executable

	folds = [fold for fold in glob(path.join(CONFIG.project_path, "fold*/")) if path.isdir(path.join(CONFIG.project_path, fold))]

	if not folds or len(folds) != CONFIG.KCV:
		
		print("- {0}-fold cross-validation has not yet been prepared. Doing it now!".format(CONFIG.KCV))
		create_kfold_cv(CONFIG)
		print("- {0}-fold cross-validation has been prepared. Continuing".format(CONFIG.KCV))
	else:
		print("- {0}-fold Cross-validation scheme was previously prepared. Continuing.".format(CONFIG.KCV))

	folds = [fold for fold in glob(path.join(CONFIG.project_path, "fold*/")) if path.isdir(path.join(CONFIG.project_path, fold))]

	print("- Distributing classifier object to each fold")

	_ = Parallel(n_jobs=-1,max_nbytes=None)(delayed(_distribute_estimator)(estimator, CONFIG.experiment_name, CONFIG.project_path, fold) for fold in folds)

	if metric != False:

		print("- Distributing custom metric object to each fold")
		_ = Parallel(n_jobs=-1,max_nbytes=None)(delayed(_distribute_metric)(metric, CONFIG.experiment_name, CONFIG.project_path, fold) for fold in folds)

	print("- Executing experiment")
	i=0
	print("\t* Setting up and Submitting jobs:")

	all_jobnames = []

	for fold in ["fold{0}".format(f) for f in range(0, CONFIG.KCV)]:

		print("\t\t* {fold}".format(fold=fold))

		JOB_TEMPLATE = """#!{shebang}
from qklearn import apply_estimator_to_fold
apply_estimator_to_fold("{config_path}", "{fold}")
		""".format(shebang=executable, fold=fold, config_path=path.join(CONFIG.project_path, "CONFIG_{experiment_name}".format(experiment_name=CONFIG.experiment_name)))

		with open(path.join(CONFIG.project_path, fold, "JOB_SCRIPT_{experiment_name}.py".format(experiment_name=CONFIG.experiment_name)), "w") as js:
			js.write(JOB_TEMPLATE)
		
		job_name=CONFIG.experiment_name + "_" + fold

		system("echo \"python {job_script_path}\" | qsub {qsub_mail} -cwd -N {job_name} -o {log_file} -e {error_file} -l h_vmem={qsub_mem} -l h_rt=01:00:00 -pe threaded {num_cores}".format(
			job_script_path=path.join(CONFIG.project_path, fold, "JOB_SCRIPT_{experiment_name}.py".format(experiment_name=CONFIG.experiment_name)), 
			job_name=job_name, 
			project_dir=path.join(CONFIG.project_path, fold),
			log_file=path.join(CONFIG.project_path, fold, job_name + ".log"),
			error_file=path.join(CONFIG.project_path, fold, job_name + ".errors"),
			num_cores=CONFIG.n_jobs if CONFIG.n_jobs != -1 else 1, 
			qsub_mail="" if not CONFIG.qsub_mail else "-m a -M " + CONFIG.qsub_mail,
			qsub_mem=CONFIG.qsub_mem
			)
		)
		#system("python {job_script_path}".format(job_script_path=path.join(CONFIG.project_path, fold, "JOB_SCRIPT_{experiment_name}.py")))
		all_jobnames.append(job_name)
		i+=1

	hold_jid = ",".join(all_jobnames)

	COLLECT_TEMPLATE = """#!{shebang}
from qklearn import collect_results
collect_results("{config_path}")
""".format(shebang=executable, 
	config_path=path.join(CONFIG.project_path, "CONFIG_{experiment_name}".format(experiment_name=CONFIG.experiment_name)))

	with open(path.join(CONFIG.project_path, "COLLECT_SCRIPT_{experiment_name}.py".format(experiment_name=CONFIG.experiment_name)), "w") as js:
			js.write(COLLECT_TEMPLATE)

	system("echo \"python {collect_script_path}\" | qsub {qsub_mail} -cwd -N {job_name} -o {project_dir} -e {project_dir} -hold_jid {hold_jid} -l h_vmem=1G -l h_rt=00:15:00".format(
		hold_jid=hold_jid, 
		collect_script_path=path.join(CONFIG.project_path, "COLLECT_SCRIPT_{experiment_name}.py".format(experiment_name=CONFIG.experiment_name)), 
		job_name=CONFIG.experiment_name + "_COLLECTOR", 
		project_dir=CONFIG.project_path),
		qsub_mail= "" if not CONFIG.qsub_mail else "-m a -M " + CONFIG.qsub_mail
	)

def collect_results(CONFIG):

	if not isinstance(CONFIG, MLConfig): CONFIG = MLConfig(CONFIG);

	_collect_importances(CONFIG)
	_collect_results(CONFIG)

def apply_estimator_to_fold(CONFIG, fold):

	if not isinstance(CONFIG, MLConfig): CONFIG = MLConfig(CONFIG);

	import pandas as pd
	from os import path, sep
	from sklearn.pipeline import Pipeline
	from sklearn.metrics import mean_squared_error, accuracy_score
	from glob import glob
	from joblib import load

	TRAIN_INPUT = pd.read_pickle(path.join(CONFIG.project_path, fold, "TRAIN_INPUT.pkl"))
	TRAIN_OUTPUT = pd.read_pickle(path.join(CONFIG.project_path, fold, "TRAIN_OUTPUT.pkl"))

	ESTIMATOR = load(path.join(CONFIG.project_path, fold, "ESTIMATOR_{experiment_name}.pkl".format(experiment_name=CONFIG.experiment_name)))

	if not path.isfile(path.join(CONFIG.project_path, fold, "METRIC_{experiment_name}.pkl".format(experiment_name=CONFIG.experiment_name))):

		if TRAIN_OUTPUT.dtype.name.startswith("float") or TRAIN_OUTPUT.dtype.name.startswith("int"):
			metric = mean_squared_error
		elif TRAIN_OUTPUT.dtype.name == "category":
			metric = accuracy_score
		else:
			raise ValueError("Unsupported dtype for output variable")
	else:

		metric = load(path.join(CONFIG.project_path, fold, "METRIC_{0}.pkl".format(CONFIG.experiment_name)))

	# Configure the estimator, or each of the steps in the Pipeline to utilize all cores, when the algorithm allows for it:
	if isinstance(ESTIMATOR, Pipeline) and hasattr(ESTIMATOR, "steps"):

		for i, (name, step) in enumerate(ESTIMATOR.steps):

		   if hasattr(step, "n_jobs"):

			   ESTIMATOR.steps[i][1].n_jobs = -1

	else:

		if hasattr(ESTIMATOR, "n_jobs"):

		   ESTIMATOR.n_jobs = -1

	#Fit the estimator/pipeline to the dats
	ESTIMATOR.fit(TRAIN_INPUT, TRAIN_OUTPUT)

	#Generate the training predictions
	train_error = metric(TRAIN_OUTPUT, ESTIMATOR.predict(TRAIN_INPUT))

	#Load the test set data
	VALIDATION_INPUT = pd.read_pickle(path.join(CONFIG.project_path, fold, "VALIDATION_INPUT.pkl"))
	VALIDATION_OUTPUT = pd.read_pickle(path.join(CONFIG.project_path, fold, "VALIDATION_OUTPUT.pkl"))

	#And make the predictions:
	validation_error = metric(VALIDATION_OUTPUT, ESTIMATOR.predict(VALIDATION_INPUT))

	oob_score = False
	#Extract feature importances if they are present, and account for the possibility of a Pipeline object (which will have a "steps" attribute)
	if isinstance(ESTIMATOR, Pipeline) and hasattr(ESTIMATOR, "steps"):

		for name, step in ESTIMATOR.steps:

			if hasattr(step, 'feature_importances_'):

				_extract_feature_importances(CONFIG,fold,step, VALIDATION_INPUT.columns.values)
			
			if hasattr(step, 'oob_score_'):
				oob_score = step.oob_score_
	else:

		if hasattr(step, 'feature_importances_'):

			_extract_feature_importances(CONFIG,fold,ESTIMATOR, VALIDATION_INPUT.columns.values)

		if hasattr(step, 'oob_score_'):

				oob_score = step.oob_score_

	#Save the results for this experiment in table format for easy processing later
	d = {"experiment_name" : [CONFIG.experiment_name],
		"fold" : [fold],
		"train_error" : [train_error],
		"validation_error" : [validation_error]
	}

	if oob_score != False: d['oob_error'] = [1.0-oob_score];

	pd.DataFrame.from_dict(d).to_csv(path.join(CONFIG.project_path, fold, "ML_RESULT_{experiment_name}_{fold}.csv".format(experiment_name=CONFIG.experiment_name,fold=fold.replace(sep, ''))), index=False)

