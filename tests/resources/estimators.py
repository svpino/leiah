from dataclasses import dataclass, field
from leiah.estimators import Estimator


@dataclass
class DummyEstimator(Estimator):
    pass


@dataclass
class ModelEstimator(Estimator):
    role: str
    version: int
    train_instance_type: str
    train_max_run: int

    fitted: bool = False
    tuned: bool = False

    def fit(self):
        self.fitted = True

    def tune(self):
        self.tuned = True


@dataclass
class ExperimentEstimator(Estimator):
    role: str
    version: int
    train_instance_type: str
    train_max_run: int
    sample: int

    fitted: bool = False
    tuned: bool = False

    def fit(self):
        self.fitted = True

    def tune(self):
        self.tuned = True
