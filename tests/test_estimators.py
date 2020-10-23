from sagemaker.parameter import CategoricalParameter
from leiah.estimators import Estimator


def test_estimator_get_training_job_name():
    estimator = Estimator(model="hello", experiment="world", hyperparameters=dict())
    assert estimator.get_training_job_name() == "training-hello-world"


def test_estimator_get_tuning_job_name():
    estimator = Estimator(model="hello", experiment="world", hyperparameters=dict())
    assert estimator.get_tuning_job_name() == "tuning-hello-world"


def test__get_categorical_parameter():
    estimator = Estimator(model="1", experiment="1")
    parameter = estimator._get_categorical_parameter(
        data={"type": "categorical", "values": [1.0, 2.0]}
    )

    assert isinstance(parameter, CategoricalParameter)
    assert parameter.values == ["1.0", "2.0"]


def test__get_hyperparameter_ranges():
    Estimator(
        model="1",
        experiment="1",
        hyperparameters=None,
        ranges={"learning_rate": {"type": "categorical", "values": [1.0, 2.0]}},
    )