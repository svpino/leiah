import pytest
from pathlib import Path

from leiah.descriptor import (
    Descriptor,
    Model,
)
from leiah.jobs import HyperparameterTuningJob, TrainingJob
from leiah.exceptions import DescriptorError
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


def test_jobs(descriptor):
    assert len(descriptor.models["model-01"].jobs) == 3
    assert len(descriptor.models["model-02"].jobs) == 1

    assert isinstance(descriptor.models["model-01"].jobs["1"], TrainingJob)
    assert isinstance(descriptor.models["model-01"].jobs["2"], TrainingJob)
    assert isinstance(
        descriptor.models["model-01"].jobs["hpt-01"], HyperparameterTuningJob
    )
    assert isinstance(descriptor.models["model-02"].jobs["1.0.1"], TrainingJob)


def test_jobs_estimator(descriptor):
    model = descriptor.models["model-01"]

    assert isinstance(model.jobs["1"].estimator, ExperimentEstimator)
    assert model.jobs["1"].estimator.sample == 123

    assert isinstance(model.jobs["2"].estimator, ModelEstimator)


def test_jobs_description(descriptor):
    model = descriptor.models["model-01"]
    assert model.jobs["1"].description == "Beautiful is better than ugly."
    assert model.jobs["2"].description is None


def test_jobs_estimator_hyperparameters(descriptor):
    estimator = descriptor.models["model-01"].jobs["1"].estimator
    assert len(estimator.hyperparameters) == 4
    assert "learning_rate" in estimator.hyperparameters
    assert "application" in estimator.hyperparameters
    assert "epochs" in estimator.hyperparameters
    assert "batch_size" in estimator.hyperparameters

    estimator = descriptor.models["model-01"].jobs["2"].estimator

    assert len(estimator.hyperparameters) == 3
    assert "application" in estimator.hyperparameters
    assert "epochs" in estimator.hyperparameters
    assert "batch_size" in estimator.hyperparameters


def test_jobs_estimator_hyperparameters_inheritance(descriptor):
    model = descriptor.models["model-01"]
    estimator = model.jobs["1"].estimator

    assert all(
        [h in estimator.hyperparameters for h in model.data["hyperparameters"].keys()]
    ), "Every model hyperparameter should be present in the estimator"

    assert (
        estimator.hyperparameters["epochs"] == model.data["hyperparameters"]["epochs"]
    ), "Hyperparameter value should have been inherited"

    assert (
        model.jobs["2"].estimator.hyperparameters["epochs"]
        != model.data["hyperparameters"]["epochs"]
    ), "Hyperparameter should have been overwritten"


def test_jobs_estimator_attributes_inheritance(descriptor):
    model = descriptor.models["model-01"]
    estimator = model.jobs["2"].estimator
    estimator.role == "role-name-experiment2"


def test_no_models(descriptor_base_path):
    descriptor_file_path = descriptor_base_path / "descriptor-02.yaml"
    descriptor = Descriptor(descriptor_file_path)

    assert len(descriptor.models) == 0


def test_descriptor_not_found():
    with pytest.raises(FileNotFoundError):
        Descriptor("unexistent")


def test_invalid_descriptor_non_yaml_file(descriptor_base_path):
    with pytest.raises(DescriptorError):
        Descriptor(descriptor_base_path / "invalid-descriptor-1.yaml")


def test_invalid_descriptor_missing_root(descriptor_base_path):
    with pytest.raises(DescriptorError):
        Descriptor(descriptor_base_path / "invalid-descriptor-2.yaml")


def test_invalid_descriptor_invalid_yaml(descriptor_base_path):
    with pytest.raises(DescriptorError):
        Descriptor(descriptor_base_path / "invalid-descriptor-3.yaml")


def test_invalid_descriptor_invalid_indentation(descriptor_base_path):
    with pytest.raises(DescriptorError):
        Descriptor(descriptor_base_path / "invalid-descriptor-4.yaml")


def test_invalid_descriptor_source():
    with pytest.raises(DescriptorError):
        Descriptor(123)


def test_estimator_missing_properties():
    with pytest.raises(DescriptorError):
        Descriptor(
            {
                "models": {
                    "model-01": {
                        "hyperparameter-tuning-jobs": {
                            "1": {"estimator": "leiah.estimators.TensorFlowEstimator"}
                        }
                    }
                }
            }
        )


def test_get_jobs_single_experiment(descriptor):
    jobs = descriptor._get_jobs(jobs="model-01.2")

    assert len(jobs) == 1
    assert jobs[0].identifier == "2"


def test_get_jobs_single_experiment_multiple_separators(descriptor):
    jobs = descriptor._get_jobs(jobs="model-02.1.0.1")

    assert len(jobs) == 1
    assert jobs[0].identifier == "1.0.1"


def test_get_jobs_invalid_experiment_name(descriptor):
    with pytest.raises(DescriptorError):
        descriptor._get_jobs(jobs="unexistent.1")


def test_get_jobs_invalid_experiment_identifier(descriptor):
    with pytest.raises(DescriptorError):
        descriptor._get_jobs(jobs="model-01.unexistent")


def test_get_jobs_multiple_jobs(descriptor):
    jobs = descriptor._get_jobs(jobs=["model-01.1", "model-02.1.0.1"])

    assert len(jobs) == 2
    assert jobs[0].identifier == "1"
    assert jobs[1].identifier == "1.0.1"


def test_get_jobs_from_model(descriptor):
    jobs = descriptor._get_jobs(jobs=["model-01", "model-02.1.0.1"])

    assert len(jobs) == 4
    assert jobs[0].identifier == "1"
    assert jobs[1].identifier == "2"
    assert jobs[2].identifier == "hpt-01"
    assert jobs[3].identifier == "1.0.1"


def test_get_all_jobs(descriptor):
    jobs = descriptor._get_jobs()
    assert len(jobs) == 4


def test_run_training_process(descriptor):
    estimator = descriptor.models["model-01"].jobs["1"].estimator

    assert estimator.fitted is False
    descriptor.run(jobs="model-01.1")
    assert estimator.fitted is True


def test_run_experiment_process(descriptor):
    estimator = descriptor.models["model-01"].jobs["hpt-01"].estimator

    assert estimator.tuned is False
    descriptor.run(jobs="model-01.hpt-01")
    assert estimator.tuned is True
