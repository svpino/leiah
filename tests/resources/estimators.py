from leiah.estimators import Estimator


class DummyEstimator(Estimator):
    def __init__(self, model, experiment, hyperparameters=None, ranges=None) -> None:
        super().__init__(
            model=model, experiment=experiment, hyperparameters=hyperparameters
        )

        self.tuned = False
        self.kwargs = None

    def tune(self, **kwargs):
        self.tuned = True
        self.kwargs = kwargs

    def get_sagemaker_estimator(self):
        return None

    def get_tuner_objective_metric_name(self):
        return "dummy"

    def get_tuner_metric_definitions(self):
        return [{"dummy": "value"}]


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
    ) -> None:
        super().__init__(
            model=model,
            experiment=experiment,
            hyperparameters=hyperparameters,
        )

        self.role = role
        self.version = version
        self.train_instance_type = train_instance_type
        self.train_max_run = train_max_run

        self.fitted = False
        self.tuned = False

    def fit(self):
        self.fitted = True

    def tune(self, **kwargs):
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
    ) -> None:

        super().__init__(
            model=model,
            experiment=experiment,
            hyperparameters=hyperparameters,
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

    def tune(self, **kwargs):
        self.tuned = True
