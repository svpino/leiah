from dataclasses import dataclass, field


@dataclass
class DummyEstimator(object):
    pass
    hyperparameters: dict = field(default_factory=dict)


@dataclass
class ModelEstimator(object):
    role: str
    version: int
    train_instance_type: str
    train_max_run: int
    hyperparameters: dict = field(default_factory=dict)


@dataclass
class ExperimentEstimator(object):
    role: str
    version: int
    train_instance_type: str
    train_max_run: int
    sample: int
    hyperparameters: dict = field(default_factory=dict)
