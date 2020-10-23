import pytest
from pathlib import Path

from leiah.descriptor import Descriptor, Experiment, Model, _get_estimator
from leiah.exceptions import (
    EstimatorMissingPropertyError,
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


def test_models_numeric_name():
    descriptor = Descriptor({"models": {101: {}, 2: {}}})

    assert isinstance(descriptor.models["101"], Model)
    assert isinstance(descriptor.models["2"], Model)


def test_models_from_dict():
    descriptor = Descriptor({"models": {"model-01": {}, "model-02": {}}})
    assert len(descriptor.models) == 2

    assert isinstance(descriptor.models["model-01"], Model)
    assert isinstance(descriptor.models["model-02"], Model)


def test_experiments(descriptor):
    assert len(descriptor.models["model-01"].experiments) == 3
    assert len(descriptor.models["model-02"].experiments) == 1

    assert isinstance(descriptor.models["model-01"].experiments["1"], Experiment)
    assert isinstance(descriptor.models["model-01"].experiments["2"], Experiment)
    assert isinstance(descriptor.models["model-01"].experiments["hpt-01"], Experiment)
    assert isinstance(descriptor.models["model-02"].experiments["1.0.1"], Experiment)

    descriptor.models["model-01"].experiments[
        "hpt-01"
    ].estimator._get_hyperparameter_ranges()


def test_experiments_estimator(descriptor):
    model = descriptor.models["model-01"]

    assert isinstance(model.experiments["1"].estimator, ExperimentEstimator)
    assert model.experiments["1"].estimator.sample == 123

    assert isinstance(model.experiments["2"].estimator, ModelEstimator)


def test_experiments_description(descriptor):
    model = descriptor.models["model-01"]
    assert model.experiments["1"].description == "Beautiful is better than ugly."
    assert model.experiments["2"].description is None


def test_experiments_estimator_hyperparameters(descriptor):
    estimator = descriptor.models["model-01"].experiments["1"].estimator
    assert len(estimator.hyperparameters) == 4
    assert "learning_rate" in estimator.hyperparameters
    assert "application" in estimator.hyperparameters
    assert "epochs" in estimator.hyperparameters
    assert "batch_size" in estimator.hyperparameters

    estimator = descriptor.models["model-01"].experiments["2"].estimator

    assert len(estimator.hyperparameters) == 3
    assert "application" in estimator.hyperparameters
    assert "epochs" in estimator.hyperparameters
    assert "batch_size" in estimator.hyperparameters


def test_experiments_estimator_hyperparameters_inheritance(descriptor):
    model = descriptor.models["model-01"]
    estimator = model.experiments["1"].estimator

    assert all(
        [h in estimator.hyperparameters for h in model.data["hyperparameters"].keys()]
    ), "Every model hyperparameter should be present in the experiment"

    assert (
        estimator.hyperparameters["epochs"] == model.data["hyperparameters"]["epochs"]
    ), "Hyperparameter value should have been inherited"

    assert (
        model.experiments["2"].estimator.hyperparameters["epochs"]
        != model.data["hyperparameters"]["epochs"]
    ), "Hyperparameter should have been overwritten"


def test_get_estimator():
    estimator = _get_estimator(
        estimator="tests.resources.estimators.DummyEstimator",
        model="model-1",
        experiment="experiment-1",
        properties=dict(),
        hyperparameters={"hp1": 123},
    )

    assert estimator.model == "model-1"
    assert estimator.experiment == "experiment-1"
    assert estimator.hyperparameters["hp1"] == 123


def test_get_estimator_invalid_estimator():
    with pytest.raises(InvalidEstimatorError):
        _get_estimator(
            estimator="Estimator",
            model="model-1",
            experiment="experiment-1",
            properties=dict(),
            hyperparameters=dict(),
        )

    with pytest.raises(InvalidEstimatorError):
        _get_estimator(
            estimator="invalid.module.Estimator",
            model="model-1",
            experiment="experiment-1",
            properties=dict(),
            hyperparameters=dict(),
        )

    with pytest.raises(InvalidEstimatorError):
        _get_estimator(
            estimator="tests.resources.estimators.Invalid",
            model="model-1",
            experiment="experiment-1",
            properties=dict(),
            hyperparameters=dict(),
        )


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


def test_estimator_missing_properties():
    with pytest.raises(EstimatorMissingPropertyError):
        Descriptor(
            {
                "models": {
                    "model-01": {
                        "experiments": {
                            "1": {
                                "estimator": {
                                    "classname": "leiah.estimators.TensorFlowEstimator"
                                }
                            }
                        }
                    }
                }
            }
        )


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


def test_process_training(descriptor):
    estimator = descriptor.models["model-01"].experiments["1"].estimator

    assert estimator.fitted is False
    descriptor.process(experiments="model-01.1")
    assert estimator.fitted is True


def test_process_tunning(descriptor):
    estimator = descriptor.models["model-01"].experiments["hpt-01"].estimator

    assert estimator.tuned is False
    descriptor.process(experiments="model-01.hpt-01")
    assert estimator.tuned is True


def test_process_invalid_experiment_type():
    model = Model(
        name="model1",
        data={
            "experiments": {
                "experiment1": {
                    "type": "invalid",
                    "estimator": {
                        "classname": "tests.resources.estimators.DummyEstimator"
                    },
                }
            }
        },
    )
    experiment = model.experiments["experiment1"]

    with pytest.raises(InvalidDescriptorError):
        experiment.process()
