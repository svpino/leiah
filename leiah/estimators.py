from dataclasses import dataclass
from sagemaker.tensorflow import TensorFlow


@dataclass
class Estimator(object):
    model: str
    experiment: str
    hyperparameters: dict

    def get_training_job_name(self):
        return f"{self.model}-{self.experiment}"

@dataclass
class TensorFlowEstimator(Estimator):
    entry_point: str
    train_instance_type: str
    source_dir: str
    model_uri: str
    model_dir: str
    code_location: str
    output_path: str
    train_max_run: int = 86400
    py_version: str = "py37"
    framework_version: str = "2.3.0"
    train_instance_count: int = 1
    train_volume_size: int = 10
    debugger_hook_config: bool = False

    def get_sagemaker_estimator(self, role):
        sagemaker_estimator = TensorFlow(
            base_job_name=self.get_training_job_name(),
            source_dir=self.source_dir,
            entry_point=self.entry_point,
            role=role,
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
