import pytest
from pathlib import Path

from leiah.descriptor import Descriptor, _get_estimator
from leiah.exceptions import InvalidDescriptorError, InvalidEstimatorError

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

    assert descriptor.models[0].name == "model-01"
    assert descriptor.models[1].name == "model-02"
    assert descriptor.models[2].name == "model-03"


def test_models_from_dict():
    descriptor = Descriptor({"models": {"model1": {}, "model2": {}}})
    assert len(descriptor.models) == 2

    assert descriptor.models[0].name == "model1"
    assert descriptor.models[1].name == "model2"


def test_model_estimator(descriptor):
    estimator = descriptor.models[0].estimator

    assert estimator.role == "role-name"
    assert estimator.version == 3


def test_model_hyperparameters(descriptor):
    model1 = descriptor.models[0]
    assert len(model1.hyperparameters) == 2
    assert "application" in model1.hyperparameters
    assert "epochs" in model1.hyperparameters

    model3 = descriptor.models[2]
    assert len(model3.hyperparameters) == 0


def test_experiments(descriptor):
    assert len(descriptor.models[0].experiments) == 3
    assert len(descriptor.models[1].experiments) == 1

    descriptor.models[0].experiments[0].identifier == "1"
    descriptor.models[0].experiments[1].identifier == "2"
    descriptor.models[0].experiments[2].identifier == "hpt-01"
    descriptor.models[1].experiments[0].identifier == "1.0.1"


def test_experiments_estimator(descriptor):
    model = descriptor.models[0]

    assert isinstance(model.experiments[0].estimator, ExperimentEstimator)
    assert model.experiments[0].estimator.sample == 123

    assert isinstance(model.experiments[1].estimator, ModelEstimator)


def test_experiments_description(descriptor):
    model = descriptor.models[0]
    assert model.experiments[0].description == "Beautiful is better than ugly."
    assert model.experiments[1].description is None


def test_experiments_hyperparameters(descriptor):
    experiment1 = descriptor.models[0].experiments[0]
    assert "learning_rate" in experiment1.hyperparameters
    assert "batch_size" in experiment1.hyperparameters

    experiment2 = descriptor.models[0].experiments[1]
    assert "epochs" in experiment2.hyperparameters
    assert "batch_size" in experiment2.hyperparameters


def test_hyperparameters_inheritance(descriptor):
    model1 = descriptor.models[0]
    experiment1 = descriptor.models[0].experiments[0]

    assert all(
        [h in experiment1.hyperparameters for h in model1.hyperparameters.keys()]
    ), "Every model hyperparameter should be present in the experiment"

    assert (
        experiment1.hyperparameters["epochs"] == model1.hyperparameters["epochs"]
    ), "Hyperparameter value should have been inherited"

    assert (
        model1.experiments[1].hyperparameters["epochs"]
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


def test_process(descriptor):
    pass
