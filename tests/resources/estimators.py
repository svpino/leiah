from dataclasses import dataclass


@dataclass
class ModelEstimator(object):
    role: str
    version: int


@dataclass
class ExperimentEstimator(object):
    sample: int
