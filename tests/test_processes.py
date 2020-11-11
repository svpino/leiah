from leiah.estimators import Estimator
import pytest

from sagemaker.parameter import (
    CategoricalParameter,
    ContinuousParameter,
    IntegerParameter,
)
from leiah.descriptor import Model
from leiah.processes import Experiment, Process
from leiah.exceptions import DescriptorError


@pytest.fixture
def model():
    return Model("model1", data={})


@pytest.fixture
def experiment_process(model):
    return Experiment(
        model=model,
        identifier="experiment1",
        data={
            "estimator": "tests.resources.estimators.DummyEstimator",
            "hyperparameter_ranges": {
                "property1": {"type": "categorical", "values": [1.0, 2.0]},
                "property2": {"type": "integer", "min_value": 1.0, "max_value": 2.0},
                "property3": {"type": "continuous", "min_value": 3.0, "max_value": 4.0},
            },
        },
    )


def test_experiment_estimator():
    model = Model("model1", data={})
    experiment = Experiment(
        model=model,
        identifier="process1",
        data={
            "estimator": "tests.resources.estimators.DummyEstimator",
            "hyperparameters": {"hp1": 123},
        },
    )

    assert experiment.estimator.model == "model1"
    assert experiment.estimator.process == "process1"
    assert experiment.estimator.hyperparameters["hp1"] == 123


@pytest.mark.parametrize(
    "estimator",
    [("invalid"), ("invalid.module.Estimator"), ("tests.resources.estimators.Invalid")],
)
def test_experiment_invalid_estimator(estimator):
    model = Model("model1", data={})
    with pytest.raises(DescriptorError):
        Experiment(
            model=model,
            identifier="experiment1",
            data={
                "estimator": estimator,
                "hyperparameters": {"hp1": 123},
            },
        )


def test__get_categorical_parameter(experiment_process):
    parameter = experiment_process._get_categorical_parameter(
        data={"type": "categorical", "values": [1.0, 2.0]}
    )

    assert isinstance(parameter, CategoricalParameter)
    assert parameter.values == ["1.0", "2.0"]


def test__get_categorical_parameter_missing_attribute(experiment_process):
    with pytest.raises(DescriptorError):
        experiment_process._get_categorical_parameter(data={"type": "categorical"})


def test__get_integer_parameter(experiment_process):
    parameter = experiment_process._get_integer_parameter(
        data={
            "type": "integer",
            "min_value": 1.0,
            "max_value": 10,
            "scaling_type": "Linear",
        }
    )

    assert isinstance(parameter, IntegerParameter)
    assert parameter.min_value == 1.0
    assert parameter.max_value == 10.0
    assert parameter.scaling_type == "Linear"


def test__get_integer_parameter_missing_attribute(experiment_process):
    with pytest.raises(DescriptorError):
        experiment_process._get_integer_parameter(
            data={
                "type": "integer",
                "max_value": 10,
            }
        )

    with pytest.raises(DescriptorError):
        experiment_process._get_integer_parameter(
            data={
                "type": "integer",
                "min_value": 1,
            }
        )


def test__get_integer_parameter_default_scaling_type(experiment_process):
    parameter = experiment_process._get_integer_parameter(
        data={
            "type": "integer",
            "min_value": 10,
            "max_value": 20,
        }
    )

    assert parameter.scaling_type == "Auto"


def test__get_continuous_parameter(experiment_process):
    parameter = experiment_process._get_continuous_parameter(
        data={
            "type": "continuous",
            "min_value": 1.0,
            "max_value": 10,
            "scaling_type": "Linear",
        }
    )

    assert isinstance(parameter, ContinuousParameter)
    assert parameter.min_value == 1.0
    assert parameter.max_value == 10.0
    assert parameter.scaling_type == "Linear"


def test__get_continuous_parameter_missing_attribute(experiment_process):
    with pytest.raises(DescriptorError):
        experiment_process._get_continuous_parameter(
            data={
                "type": "integer",
                "max_value": 10,
            }
        )

    with pytest.raises(DescriptorError):
        experiment_process._get_continuous_parameter(
            data={
                "type": "integer",
                "min_value": 1,
            }
        )


def test__get_continuous_parameter_default_scaling_type(experiment_process):
    parameter = experiment_process._get_continuous_parameter(
        data={
            "type": "integer",
            "min_value": 10,
            "max_value": 20,
        }
    )

    assert parameter.scaling_type == "Auto"


def test_hyperparameter_ranges(experiment_process):
    assert len(experiment_process.hyperparameter_ranges) == 3


def test_hyperparameter_ranges_missing_parameter_type(model):
    with pytest.raises(DescriptorError):
        Experiment(
            model=model,
            identifier="experiment1",
            data={
                "estimator": "tests.resources.estimators.DummyEstimator",
                "hyperparameter_ranges": {
                    "property1": {"values": [1.0, 2.0]},
                },
            },
        )


def test_hyperparameter_ranges_invalid_parameter_type(model):
    with pytest.raises(DescriptorError):
        Experiment(
            model=model,
            identifier="experiment1",
            data={
                "estimator": "tests.resources.estimators.DummyEstimator",
                "hyperparameter_ranges": {
                    "property1": {"type": "invalid", "values": [1.0, 2.0]},
                },
            },
        )


def test_experiment_process_kwargs(model):
    process = Experiment(
        model=model,
        identifier="process1",
        data={
            "estimator": "tests.resources.estimators.DummyEstimator",
            "max_jobs": 10,
            "max_parallel_jobs": 4,
            "objective_type": "Maximize",
        },
    )

    process.run()
    assert process.estimator.kwargs["max_jobs"] == 10
    assert process.estimator.kwargs["max_parallel_jobs"] == 4
    assert process.estimator.kwargs["objective_type"] == "Maximize"


def test_experiment_process_hyperparameter_ranges(experiment_process):
    experiment_process.run()
    assert len(experiment_process.estimator.kwargs["hyperparameter_ranges"]) == len(
        experiment_process.hyperparameter_ranges
    )
