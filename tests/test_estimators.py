from leiah.estimators import Estimator


def test_estimator_get_training_job_name():
    estimator = Estimator(model="hello", experiment="world", hyperparameters=dict())
    assert estimator.get_training_job_name() == "training-hello-world"


def test_estimator_get_tuning_job_name():
    estimator = Estimator(model="hello", experiment="world", hyperparameters=dict())
    assert estimator.get_tuning_job_name() == "tuning-hello-world"


def test__get_hyperparameter_ranges():
    estimator = Estimator(model="hello", experiment="world", ranges=dict())