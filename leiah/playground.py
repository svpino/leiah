import logging

from leiah.descriptor import Descriptor


class Playground(object):
    def __init__(self, descriptor: Descriptor) -> None:
        self.descriptor = descriptor

    def process(self):
        for model in self.descriptor.models:
            logging.info(f"Processing model \"{model.name}\"...")
            for experiment in model.experiments:
                pass

        """
        estimator = self.get_estimator()

        training_input, validation_input, testing_input = self._get_inputs(self.brand_id)

        return estimator.fit(
            {
                "training": training_input,
                "validation": validation_input,
                "testing": testing_input,
            },
            wait=wait,
        )

        def get_estimator(self):
            estimator = TensorFlow(
                base_job_name=self.training_job_name,
                source_dir=self.source_dir,
                entry_point="training.py",
                role=self.role,
                hyperparameters=self.hyperparameters,
                train_instance_type=self.train_instance_type,
                train_instance_count=1,
                py_version="py37",
                framework_version="2.3.0",
                debugger_hook_config=False,
                model_uri=self.input_location,
                model_dir=self.output_location,
                code_location=self.output_location,
                output_path=self.output_location,
                train_max_run=self.train_max_run,
                script_mode=True,
            )

            if self.dataset_source == "FSxLustre":
                estimator.subnets = self.dataset_fsx_subnets
                estimator.security_group_ids = self.dataset_security_group_ids
            elif self.dataset_source == "S3":
                estimator.train_volume_size = self.train_volume_size
                
            return estimator
        """

