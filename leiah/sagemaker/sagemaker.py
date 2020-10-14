import yaml
from dataclasses import dataclass, field

from leiah.sagemaker.exceptions import InvalidDescriptorError


@dataclass
class Model(object):
    name: str
    hyperparameters: dict() = field(default_factory=dict)
    experiments: list[object] = field(default_factory=list)

    @classmethod
    def create(cls, name, data):
        model = Model(name)

        if "hyperparameters" in data:
            model.hyperparameters = data["hyperparameters"]

        if "experiments" in data:
            for identifier, data in data["experiments"].items():
                model.experiments.append(Experiment.create(model, identifier, data))

        return model


@dataclass
class Experiment(object):
    identifier: str
    description: str = None
    hyperparameters: dict() = field(default_factory=dict)

    @classmethod
    def create(cls, model: Model, identifier: str, data: dict()):
        experiment_type = data.get("type", "training")
        if experiment_type == "training":
            experiment = Training(model, identifier, data)
        elif experiment_type == "tuning":
            experiment = Tuning(model, identifier, data)
        else:
            raise InvalidDescriptorError(
                f'Experiment type "{experiment_type}" is not supported.'
            )

        experiment.description = data.get("description", None)

        experiment.hyperparameters.update(model.hyperparameters)
        if "hyperparameters" in data:
            experiment.hyperparameters.update(data["hyperparameters"])

        return experiment


@dataclass
class Training(Experiment):
    pass


@dataclass
class Tuning(Experiment):
    pass


class SageMaker(object):
    def __init__(self, descriptor_file_path) -> None:
        self.__models = []

        self._load_descriptor(descriptor_file_path)

    def train(self):
        pass

    def tune(self):
        pass

    def _load_descriptor(self, descriptor_file_path):
        try:
            with open(descriptor_file_path) as f:
                descriptor = yaml.load(f, Loader=yaml.FullLoader)
        except FileNotFoundError:
            raise
        else:
            self._parse_descriptor(descriptor)

    def _parse_descriptor(self, descriptor):
        if not isinstance(descriptor, dict):
            raise InvalidDescriptorError("The specified file is not a valid descriptor")

        try:
            models = descriptor["models"]
        except KeyError:
            raise InvalidDescriptorError(
                'Descriptor file is missing the root element "models".'
            )
        else:
            if models is None:
                return

            for name, data in models.items():
                self.__models.append(Model.create(name, data))

    @property
    def models(self):
        return self.__models