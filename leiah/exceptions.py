class LeiahError(Exception):
    pass


class InvalidDescriptorError(LeiahError):
    pass


class InvalidEstimatorError(LeiahError):
    def __init__(self, classname: str) -> None:
        super().__init__(f'Error creating estimator "{classname}"')


class ExperimentNotFoundError(LeiahError):
    def __init__(self, experiment: str) -> None:
        super().__init__(f'Experiment "{experiment}" was not found')
