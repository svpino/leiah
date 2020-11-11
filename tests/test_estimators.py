from sagemaker.parameter import ContinuousParameter
from tests.resources.estimators import DummyEstimator
from leiah.estimators import Estimator


def test_estimator_get_training_job_name():
    estimator = Estimator(model="hello", process="world", hyperparameters=dict())
    assert estimator.get_training_job_name() == "training-hello-world"


def test_estimator_get_tuning_job_name():
    estimator = Estimator(model="hello", process="world", hyperparameters=dict())
    assert estimator.get_tuning_job_name() == "experiment-hello-world"


def test_estimator_get_sagemaker_tuner_default_values():
    estimator = DummyEstimator(model="hello", process="world", hyperparameters=dict())

    hyperparameter_ranges = {"sample": ContinuousParameter(1.0, 2.0)}
    tuner = estimator.get_sagemaker_tuner(hyperparameter_ranges=hyperparameter_ranges)

    assert tuner.objective_type == "Minimize"
    assert tuner.max_jobs == 1
    assert tuner.max_parallel_jobs == 1


def test_estimator_get_sagemaker_tuner_supplied_values():
    estimator = DummyEstimator(model="hello", process="world", hyperparameters=dict())

    hyperparameter_ranges = {"sample": ContinuousParameter(1.0, 2.0)}
    tuner = estimator.get_sagemaker_tuner(
        hyperparameter_ranges=hyperparameter_ranges,
        objective_type="Maximize",
        max_jobs=3,
        max_parallel_jobs=5,
    )

    assert tuner.objective_type == "Maximize"
    assert tuner.max_jobs == 3
    assert tuner.max_parallel_jobs == 5


def test_estimator_objective_metric_name():
    estimator = DummyEstimator(model="hello", process="world", hyperparameters=dict())

    hyperparameter_ranges = {"sample": ContinuousParameter(1.0, 2.0)}
    tuner = estimator.get_sagemaker_tuner(hyperparameter_ranges=hyperparameter_ranges)

    assert tuner.objective_metric_name == estimator.get_tuner_objective_metric_name()


def test_estimator_metric_definitions():
    estimator = DummyEstimator(model="hello", process="world", hyperparameters=dict())

    hyperparameter_ranges = {"sample": ContinuousParameter(1.0, 2.0)}
    tuner = estimator.get_sagemaker_tuner(hyperparameter_ranges=hyperparameter_ranges)

    assert tuner.metric_definitions == estimator.get_tuner_metric_definitions()
