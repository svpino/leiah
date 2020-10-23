from leiah.estimators import Estimator


class DummyEstimator(Estimator):
    def __init__(self, model, experiment, hyperparameters=None, ranges=None) -> None:
        super().__init__(
            model=model,
            experiment=experiment,
            hyperparameters=hyperparameters,
            ranges=ranges,
        )


class ModelEstimator(Estimator):
    def __init__(
        self,
        model: str,
        experiment: str,
        role: str,
        version: int,
        train_instance_type: str,
        train_max_run: int,
        hyperparameters=None,
        ranges=None,
    ) -> None:
        super().__init__(
            model=model,
            experiment=experiment,
            hyperparameters=hyperparameters,
            ranges=ranges,
        )

        self.role = role
        self.version = version
        self.train_instance_type = train_instance_type
        self.train_max_run = train_max_run

        self.fitted = False
        self.tuned = False

    def fit(self):
        self.fitted = True

    def tune(self):
        self.tuned = True


class ExperimentEstimator(Estimator):
    def __init__(
        self,
        model: str,
        experiment: str,
        role: str,
        version: int,
        sample: int,
        train_instance_type: str,
        train_max_run: int,
        hyperparameters=None,
        ranges=None,
    ) -> None:

        super().__init__(
            model=model,
            experiment=experiment,
            hyperparameters=hyperparameters,
            ranges=ranges,
        )

        self.role = role
        self.version = version
        self.sample = sample
        self.train_instance_type = train_instance_type
        self.train_max_run = train_max_run

        self.fitted = False
        self.tuned = False

    def fit(self):
        self.fitted = True

    def tune(self):
        self.tuned = True
