import sagemaker

from sagemaker.tensorflow import TensorFlow
from sagemaker.tuner import (
    HyperparameterTuner,
)


class Estimator(object):
    def __init__(self, model: str, experiment: str, hyperparameters: dict = None):
        self.model = model
        self.experiment = experiment
        self.hyperparameters = hyperparameters or dict()

    def fit(self):
        print(f"Fitting estimator {self.get_training_job_name()}...")

        sagemaker_estimator = self.get_sagemaker_estimator()
        return sagemaker_estimator.fit(self.channels, wait=False)

    def tune(self, max_jobs: int, max_parallel_jobs: int, hyperparameter_ranges: dict):
        print(f"Tuning estimator {self.get_tuning_job_name()}...")

        sagemaker_estimator = self.get_sagemaker_estimator()

        tuner = HyperparameterTuner(
            base_tuning_job_name=self.get_tuning_job_name(),
            estimator=sagemaker_estimator,
            objective_metric_name="val_loss",
            objective_type="Minimize",
            hyperparameter_ranges=hyperparameter_ranges,
            metric_definitions=[
                {"Name": "loss", "Regex": " loss: ([0-9\\.]+)"},
                {"Name": "accuracy", "Regex": " accuracy: ([0-9\\.]+)"},
                {"Name": "val_loss", "Regex": " val_loss: ([0-9\\.]+)"},
                {"Name": "val_accuracy", "Regex": " val_accuracy: ([0-9\\.]+)"},
            ],
            max_jobs=max_jobs,
            max_parallel_jobs=max_parallel_jobs,
        )

        return tuner.fit(self.channels)

    def get_training_job_name(self):
        return f"training-{self.model}-{self.experiment}"

    def get_tuning_job_name(self):
        return f"tuning-{self.model}-{self.experiment}"

    def get_sagemaker_estimator(self):
        raise NotImplementedError()


class TensorFlowEstimator(Estimator):
    def __init__(
        self,
        model: str,
        experiment: str,
        entry_point: str,
        train_instance_type: str,
        source_dir: str,
        model_uri: str,
        model_dir: str,
        code_location: str,
        output_path: str,
        hyperparameters: dict = None,
        train_max_run: int = 86400,
        py_version: str = "py37",
        framework_version: str = "2.3.0",
        train_instance_count: int = 1,
        train_volume_size: int = 10,
        debugger_hook_config: bool = False,
        channels: dict = None,
    ):
        super().__init__(
            model=model,
            experiment=experiment,
            hyperparameters=hyperparameters,
        )

        self.entry_point = entry_point
        self.train_instance_type = train_instance_type
        self.source_dir = source_dir
        self.model_uri = model_uri
        self.model_dir = model_dir
        self.code_location = code_location
        self.output_path = output_path
        self.train_max_run = train_max_run
        self.py_version = py_version
        self.framework_version = framework_version
        self.train_instance_count = train_instance_count
        self.train_volume_size = train_volume_size
        self.debugger_hook_config = debugger_hook_config
        self.channels = channels

    def get_sagemaker_estimator(self):
        sagemaker_estimator = TensorFlow(
            base_job_name=self.get_training_job_name(),
            source_dir=self.source_dir,
            entry_point=self.entry_point,
            role=sagemaker.get_execution_role(),
            hyperparameters=self.hyperparameters,
            train_instance_type=self.train_instance_type,
            train_instance_count=self.train_instance_count,
            py_version=self.py_version,
            framework_version=self.framework_version,
            debugger_hook_config=self.debugger_hook_config,
            model_uri=self.model_uri,
            model_dir=self.model_dir,
            code_location=self.code_location,
            output_path=self.output_path,
            train_max_run=self.train_max_run,
            train_volume_size=self.train_volume_size,
            script_mode=True,
        )

        return sagemaker_estimator
