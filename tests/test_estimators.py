from dataclasses import dataclass, field
from leiah.estimators import Estimator


def test_estimator_get_training_job_name():
    estimator = Estimator(model="hello", experiment="world", hyperparameters=dict())
    assert estimator.get_training_job_name() == "hello-world"


def test_estimator_dict_attribute():
    @dataclass
    class FakeEstimator(Estimator):
        channels: dict = field(default_factory=dict)

    estimator = FakeEstimator(model="hello", experiment="world", hyperparameters=dict())
    assert len(estimator.channels.keys()) == 0

    estimator = FakeEstimator(
        model="hello",
        experiment="world",
        hyperparameters=dict(),
        channels={"channel1": 123},
    )
    assert len(estimator.channels.keys()) == 1
