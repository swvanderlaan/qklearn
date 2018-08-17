def _initialize_experiment(CONFIG):
    from os import path, system
    from shutil import copyfile

    if not path.isdir(CONFIG.experiment_path): system("mkdir {experiment_path}".format(experiment_path=CONFIG.experiment_path));
    copyfile(CONFIG.config_path, path.join(CONFIG.experiment_path, "CONFIG"))

def _do_fold(train, test, i, K, X, Y, experiment_path):
    
    import pandas as pd
    from os import system, path

    system("mkdir {fold_path}".format(fold_path=path.join(experiment_path, "fold{0}".format(i))))

    print("\t\t* fold {0}".format(i, K))

    train_input, train_output = X.iloc[train], Y.iloc[train]
    test_input, test_output = X.iloc[test], Y.iloc[test]

    train_input.to_pickle(path.join(experiment_path, "fold{0}".format(i), "TRAIN_INPUT.pkl"))
    train_output.to_pickle(path.join(experiment_path, "fold{0}".format(i), "TRAIN_OUTPUT.pkl"))

    train_input.to_pickle(path.join(experiment_path, "fold{0}".format(i), "VALIDATION_INPUT.pkl"))
    train_output.to_pickle(path.join(experiment_path, "fold{0}".format(i), "VALIDATION_OUTPUT.pkl"))

def _distribute_estimator(estimator, experiment_name, project_path, fold):
    from joblib import dump
    from os.path import join
    dump(estimator, join(project_path, fold, "ESTIMATOR_{0}.pkl".format(experiment_name)))

def _distribute_metric(metric, experiment_name, project_path, fold):
    from joblib import dump
    from os.path import join
    dump(metric, join(project_path, fold, "METRIC_{0}.pkl".format(experiment_name)))

def _extract_feature_importances(CONFIG, fold, e, colnames):
    import pandas as pd
    import numpy as np
    from os import path
    import matplotlib.pyplot as plt

    importances = e.feature_importances_
    std = np.std([tree.feature_importances_ for tree in e.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking

    feature_names = [colnames[i] for i in indices]
    
    plt.figure(figsize=(20, 10))
    plt.title("Feature importances")
    plt.bar(range(len(colnames)), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(len(colnames)), feature_names, rotation=45)
    plt.xlim([-1, len(colnames)])
    plt.savefig(path.join(CONFIG.experiment_path, fold, "feature_importance_plot_{0}.png".format(CONFIG.experiment_name)))

    pd.DataFrame.from_dict({"feature" : feature_names, "importance" : importances[indices]}).to_csv(path.join(CONFIG.experiment_path, fold, "feature_importances_{0}.csv".format(CONFIG.experiment_name)), index=False)