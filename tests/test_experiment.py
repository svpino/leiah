import pytest

from sagemaker.parameter import (
    CategoricalParameter,
    ContinuousParameter,
    IntegerParameter,
)
from leiah.descriptor import Model
from leiah.experiments import Experiment, TuningExperiment
from leiah.exceptions import DescriptorError


@pytest.fixture
def model():
    return Model("model1", data={})


@pytest.fixture
def tuning_experiment(model):
    return TuningExperiment(
        model=model,
        identifier="experiment1",
        data={
            "estimator": {
                "classname": "tests.resources.estimators.DummyEstimator",
            },
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
        identifier="experiment1",
        data={
            "estimator": {
                "classname": "tests.resources.estimators.DummyEstimator",
            },
            "hyperparameters": {"hp1": 123},
        },
    )

    assert experiment.estimator.model == "model1"
    assert experiment.estimator.experiment == "experiment1"
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
                "estimator": {
                    "classname": estimator,
                },
                "hyperparameters": {"hp1": 123},
            },
        )


def test__get_categorical_parameter(tuning_experiment):
    parameter = tuning_experiment._get_categorical_parameter(
        data={"type": "categorical", "values": [1.0, 2.0]}
    )

    assert isinstance(parameter, CategoricalParameter)
    assert parameter.values == ["1.0", "2.0"]


def test__get_categorical_parameter_missing_attribute(tuning_experiment):
    with pytest.raises(DescriptorError):
        tuning_experiment._get_categorical_parameter(data={"type": "categorical"})


def test__get_integer_parameter(tuning_experiment):
    parameter = tuning_experiment._get_integer_parameter(
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


def test__get_integer_parameter_missing_attribute(tuning_experiment):
    with pytest.raises(DescriptorError):
        tuning_experiment._get_integer_parameter(
            data={
                "type": "integer",
                "max_value": 10,
            }
        )

    with pytest.raises(DescriptorError):
        tuning_experiment._get_integer_parameter(
            data={
                "type": "integer",
                "min_value": 1,
            }
        )


def test__get_integer_parameter_default_scaling_type(tuning_experiment):
    parameter = tuning_experiment._get_integer_parameter(
        data={
            "type": "integer",
            "min_value": 10,
            "max_value": 20,
        }
    )

    assert parameter.scaling_type == "Auto"


def test__get_continuous_parameter(tuning_experiment):
    parameter = tuning_experiment._get_continuous_parameter(
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


def test__get_continuous_parameter_missing_attribute(tuning_experiment):
    with pytest.raises(DescriptorError):
        tuning_experiment._get_continuous_parameter(
            data={
                "type": "integer",
                "max_value": 10,
            }
        )

    with pytest.raises(DescriptorError):
        tuning_experiment._get_continuous_parameter(
            data={
                "type": "integer",
                "min_value": 1,
            }
        )


def test__get_continuous_parameter_default_scaling_type(tuning_experiment):
    parameter = tuning_experiment._get_continuous_parameter(
        data={
            "type": "integer",
            "min_value": 10,
            "max_value": 20,
        }
    )

    assert parameter.scaling_type == "Auto"


def test_hyperparameter_ranges(tuning_experiment):
    assert len(tuning_experiment.hyperparameter_ranges) == 3


def test_hyperparameter_ranges_missing_parameter_type(model):
    with pytest.raises(DescriptorError):
        TuningExperiment(
            model=model,
            identifier="experiment1",
            data={
                "estimator": {
                    "classname": "tests.resources.estimators.DummyEstimator",
                },
                "hyperparameter_ranges": {
                    "property1": {"values": [1.0, 2.0]},
                },
            },
        )


def test_hyperparameter_ranges_invalid_parameter_type(model):
    with pytest.raises(DescriptorError):
        TuningExperiment(
            model=model,
            identifier="experiment1",
            data={
                "estimator": {
                    "classname": "tests.resources.estimators.DummyEstimator",
                },
                "hyperparameter_ranges": {
                    "property1": {"type": "invalid", "values": [1.0, 2.0]},
                },
            },
        )


def test_tuning_experiment_max_jobs(model):
    experiment = TuningExperiment(
        model=model,
        identifier="experiment1",
        data={
            "estimator": {
                "classname": "tests.resources.estimators.DummyEstimator",
            },
            "max_jobs": 2,
        },
    )

    assert experiment.max_jobs == 2
    assert (
        experiment.max_parallel_jobs == 1
    ), "The max number of parallel jobs should be 1 by default"

    experiment.process()
    assert experiment.estimator.max_jobs == 2
    assert (
        experiment.estimator.max_parallel_jobs == 1
    ), "The max number of parallel jobs should be 1 by default"


def test_tuning_experiment_max_parallel_jobs(model):
    experiment = TuningExperiment(
        model=model,
        identifier="experiment1",
        data={
            "estimator": {
                "classname": "tests.resources.estimators.DummyEstimator",
            },
            "max_parallel_jobs": 2,
        },
    )

    assert experiment.max_jobs == 1, "The max number of jobs should be 1 by default"
    assert experiment.max_parallel_jobs == 2

    experiment.process()
    assert (
        experiment.estimator.max_jobs == 1
    ), "The max number of jobs should be 1 by default"
    assert experiment.estimator.max_parallel_jobs == 2


def test_tuning_experiment_hyperparameter_ranges(tuning_experiment):
    tuning_experiment.process()
    assert (
        tuning_experiment.estimator.hyperparameter_ranges
        == tuning_experiment.hyperparameter_ranges
    )
