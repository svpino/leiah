import pytest

from sagemaker.parameter import (
    CategoricalParameter,
    ContinuousParameter,
    IntegerParameter,
)
from leiah.descriptor import Model
from leiah.jobs import HyperparameterTuningJob
from leiah.exceptions import DescriptorError


@pytest.fixture
def model():
    return Model("model1", data={})


@pytest.fixture
def hyperparameter_tuning_job(model):
    return HyperparameterTuningJob(
        model=model,
        identifier="job1",
        data={
            "estimator": "tests.resources.estimators.DummyEstimator",
            "hyperparameter_ranges": {
                "property1": {"type": "categorical", "values": [1.0, 2.0]},
                "property2": {"type": "integer", "min_value": 1.0, "max_value": 2.0},
                "property3": {"type": "continuous", "min_value": 3.0, "max_value": 4.0},
            },
        },
    )


def test_job_estimator():
    model = Model("model1", data={})
    job = HyperparameterTuningJob(
        model=model,
        identifier="job1",
        data={
            "estimator": "tests.resources.estimators.DummyEstimator",
            "hyperparameters": {"hp1": 123},
        },
    )

    assert job.estimator.model == "model1"
    assert job.estimator.job == "job1"
    assert job.estimator.hyperparameters["hp1"] == 123


@pytest.mark.parametrize(
    "estimator",
    [("invalid"), ("invalid.module.Estimator"), ("tests.resources.estimators.Invalid")],
)
def test_job_invalid_estimator(estimator):
    model = Model("model1", data={})
    with pytest.raises(DescriptorError):
        HyperparameterTuningJob(
            model=model,
            identifier="job1",
            data={
                "estimator": estimator,
                "hyperparameters": {"hp1": 123},
            },
        )


def test__get_categorical_parameter(hyperparameter_tuning_job):
    parameter = hyperparameter_tuning_job._get_categorical_parameter(
        data={"type": "categorical", "values": [1.0, 2.0]}
    )

    assert isinstance(parameter, CategoricalParameter)
    assert parameter.values == ["1.0", "2.0"]


def test__get_categorical_parameter_missing_attribute(hyperparameter_tuning_job):
    with pytest.raises(DescriptorError):
        hyperparameter_tuning_job._get_categorical_parameter(
            data={"type": "categorical"}
        )


def test__get_integer_parameter(hyperparameter_tuning_job):
    parameter = hyperparameter_tuning_job._get_integer_parameter(
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


def test__get_integer_parameter_missing_attribute(hyperparameter_tuning_job):
    with pytest.raises(DescriptorError):
        hyperparameter_tuning_job._get_integer_parameter(
            data={
                "type": "integer",
                "max_value": 10,
            }
        )

    with pytest.raises(DescriptorError):
        hyperparameter_tuning_job._get_integer_parameter(
            data={
                "type": "integer",
                "min_value": 1,
            }
        )


def test__get_integer_parameter_default_scaling_type(hyperparameter_tuning_job):
    parameter = hyperparameter_tuning_job._get_integer_parameter(
        data={
            "type": "integer",
            "min_value": 10,
            "max_value": 20,
        }
    )

    assert parameter.scaling_type == "Auto"


def test__get_continuous_parameter(hyperparameter_tuning_job):
    parameter = hyperparameter_tuning_job._get_continuous_parameter(
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


def test__get_continuous_parameter_missing_attribute(hyperparameter_tuning_job):
    with pytest.raises(DescriptorError):
        hyperparameter_tuning_job._get_continuous_parameter(
            data={
                "type": "integer",
                "max_value": 10,
            }
        )

    with pytest.raises(DescriptorError):
        hyperparameter_tuning_job._get_continuous_parameter(
            data={
                "type": "integer",
                "min_value": 1,
            }
        )


def test__get_continuous_parameter_default_scaling_type(hyperparameter_tuning_job):
    parameter = hyperparameter_tuning_job._get_continuous_parameter(
        data={
            "type": "integer",
            "min_value": 10,
            "max_value": 20,
        }
    )

    assert parameter.scaling_type == "Auto"


def test_hyperparameter_ranges(hyperparameter_tuning_job):
    assert len(hyperparameter_tuning_job.hyperparameter_ranges) == 3


def test_hyperparameter_ranges_missing_parameter_type(model):
    with pytest.raises(DescriptorError):
        HyperparameterTuningJob(
            model=model,
            identifier="job1",
            data={
                "estimator": "tests.resources.estimators.DummyEstimator",
                "hyperparameter_ranges": {
                    "property1": {"values": [1.0, 2.0]},
                },
            },
        )


def test_hyperparameter_ranges_invalid_parameter_type(model):
    with pytest.raises(DescriptorError):
        HyperparameterTuningJob(
            model=model,
            identifier="job1",
            data={
                "estimator": "tests.resources.estimators.DummyEstimator",
                "hyperparameter_ranges": {
                    "property1": {"type": "invalid", "values": [1.0, 2.0]},
                },
            },
        )


def test_hyperparameter_tuning_job_kwargs(model):
    job = HyperparameterTuningJob(
        model=model,
        identifier="job1",
        data={
            "estimator": "tests.resources.estimators.DummyEstimator",
            "max_jobs": 10,
            "max_parallel_jobs": 4,
            "objective_type": "Maximize",
        },
    )

    job.run()
    assert job.estimator.kwargs["max_jobs"] == 10
    assert job.estimator.kwargs["max_parallel_jobs"] == 4
    assert job.estimator.kwargs["objective_type"] == "Maximize"


def test_hyperparameter_tuning_job_hyperparameter_ranges(hyperparameter_tuning_job):
    hyperparameter_tuning_job.run()
    assert len(
        hyperparameter_tuning_job.estimator.kwargs["hyperparameter_ranges"]
    ) == len(hyperparameter_tuning_job.hyperparameter_ranges)
