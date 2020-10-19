from _pytest import python
import pytest
from pathlib import Path

from leiah.descriptor import Descriptor, Experiment, Model, _get_estimator
from leiah.exceptions import (
    InvalidDescriptorError,
    InvalidEstimatorError,
    ExperimentNotFoundError,
)

from tests.resources.estimators import ModelEstimator, ExperimentEstimator


@pytest.fixture
def descriptor_base_path():
    return Path().cwd() / "tests" / "resources"


@pytest.fixture
def descriptor(descriptor_base_path):
    descriptor_file_path = descriptor_base_path / "descriptor-01.yaml"
    return Descriptor(descriptor_file_path)


def test_models(descriptor):
    assert len(descriptor.models) == 3

    assert isinstance(descriptor.models["model-01"], Model)
    assert isinstance(descriptor.models["model-02"], Model)
    assert isinstance(descriptor.models["model-03"], Model)


def test_models_from_dict():
    descriptor = Descriptor({"models": {"model-01": {}, "model-02": {}}})
    assert len(descriptor.models) == 2

    assert isinstance(descriptor.models["model-01"], Model)
    assert isinstance(descriptor.models["model-02"], Model)


def test_model_estimator(descriptor):
    estimator = descriptor.models["model-01"].estimator

    assert estimator.role == "role-name"
    assert estimator.version == 3


def test_model_hyperparameters(descriptor):
    model1 = descriptor.models["model-01"]
    assert len(model1.hyperparameters) == 2
    assert "application" in model1.hyperparameters
    assert "epochs" in model1.hyperparameters

    model3 = descriptor.models["model-03"]
    assert len(model3.hyperparameters) == 0


def test_experiments(descriptor):
    assert len(descriptor.models["model-01"].experiments) == 3
    assert len(descriptor.models["model-02"].experiments) == 1

    assert isinstance(descriptor.models["model-01"].experiments["1"], Experiment)
    assert isinstance(descriptor.models["model-01"].experiments["2"], Experiment)
    assert isinstance(descriptor.models["model-01"].experiments["hpt-01"], Experiment)
    assert isinstance(descriptor.models["model-02"].experiments["1.0.1"], Experiment)


def test_experiments_estimator(descriptor):
    model = descriptor.models["model-01"]

    assert isinstance(model.experiments["1"].estimator, ExperimentEstimator)
    assert model.experiments["1"].estimator.sample == 123

    assert isinstance(model.experiments["2"].estimator, ModelEstimator)


def test_experiments_description(descriptor):
    model = descriptor.models["model-01"]
    assert model.experiments["1"].description == "Beautiful is better than ugly."
    assert model.experiments["2"].description is None


def test_experiments_hyperparameters(descriptor):
    experiment1 = descriptor.models["model-01"].experiments["1"]
    assert "learning_rate" in experiment1.hyperparameters
    assert "batch_size" in experiment1.hyperparameters

    experiment2 = descriptor.models["model-01"].experiments["2"]
    assert "epochs" in experiment2.hyperparameters
    assert "batch_size" in experiment2.hyperparameters


def test_hyperparameters_inheritance(descriptor):
    model1 = descriptor.models["model-01"]
    experiment1 = descriptor.models["model-01"].experiments["1"]

    assert all(
        [h in experiment1.hyperparameters for h in model1.hyperparameters.keys()]
    ), "Every model hyperparameter should be present in the experiment"

    assert (
        experiment1.hyperparameters["epochs"] == model1.hyperparameters["epochs"]
    ), "Hyperparameter value should have been inherited"

    assert (
        model1.experiments["2"].hyperparameters["epochs"]
        != model1.hyperparameters["epochs"]
    ), "Hyperparameter should have been overwritten"


def test_get_estimator_invalid_estimator():
    with pytest.raises(InvalidEstimatorError):
        _get_estimator(data={"classname": "invalid.module.Estimator"})

    with pytest.raises(InvalidEstimatorError):
        _get_estimator(data={"classname": "tests.resources.estimators.Invalid"})


def test_no_models(descriptor_base_path):
    descriptor_file_path = descriptor_base_path / "descriptor-02.yaml"
    descriptor = Descriptor(descriptor_file_path)

    assert len(descriptor.models) == 0


def test_descriptor_not_found():
    with pytest.raises(FileNotFoundError):
        Descriptor("unexistent")


def test_invalid_descriptor_non_yaml_file(descriptor_base_path):
    with pytest.raises(InvalidDescriptorError):
        Descriptor(descriptor_base_path / "invalid-descriptor-1.yaml")


def test_invalid_descriptor_missing_root(descriptor_base_path):
    with pytest.raises(InvalidDescriptorError):
        Descriptor(descriptor_base_path / "invalid-descriptor-2.yaml")


def test_invalid_descriptor_invalid_yaml(descriptor_base_path):
    with pytest.raises(InvalidDescriptorError):
        Descriptor(descriptor_base_path / "invalid-descriptor-4.yaml")


def test_invalid_descriptor_source():
    with pytest.raises(InvalidDescriptorError):
        Descriptor(123)


def test_get_experiments_single_experiment(descriptor):
    experiments = descriptor._get_experiments(experiments="model-01.2")

    assert len(experiments) == 1
    assert experiments[0].identifier == "2"


def test_get_experiments_single_experiment_multiple_separators(descriptor):
    experiments = descriptor._get_experiments(experiments="model-02.1.0.1")

    assert len(experiments) == 1
    assert experiments[0].identifier == "1.0.1"


def test_get_experiments_invalid_experiment_name(descriptor):
    with pytest.raises(ExperimentNotFoundError):
        descriptor._get_experiments(experiments="unexistent.1")


def test_get_experiments_invalid_experiment_identifier(descriptor):
    with pytest.raises(ExperimentNotFoundError):
        descriptor._get_experiments(experiments="model-01.unexistent")


def test_get_experiments_multiple_experiments(descriptor):
    experiments = descriptor._get_experiments(
        experiments=["model-01.1", "model-02.1.0.1"]
    )

    assert len(experiments) == 2
    assert experiments[0].identifier == "1"
    assert experiments[1].identifier == "1.0.1"


def test_get_experiments_from_model(descriptor):
    experiments = descriptor._get_experiments(
        experiments=["model-01", "model-02.1.0.1"]
    )

    assert len(experiments) == 4
    assert experiments[0].identifier == "1"
    assert experiments[1].identifier == "2"
    assert experiments[2].identifier == "hpt-01"
    assert experiments[3].identifier == "1.0.1"


def test_get_all_experiments(descriptor):
    experiments = descriptor._get_experiments()
    assert len(experiments) == 4
