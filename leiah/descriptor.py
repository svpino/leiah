import yaml
from dataclasses import dataclass, field

from leiah.exceptions import InvalidDescriptorError


@dataclass
class Experiment(object):
    identifier: str
    description: str = None
    hyperparameters: dict() = field(default_factory=dict)

    @classmethod
    def create(cls, model, identifier: str, data: dict()):
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


class Model(object):
    def __init__(self, name: str, data: dict()) -> None:
        self.name = name
        self.hyperparameters = dict()
        self.experiments = []

        if "hyperparameters" in data:
            self.hyperparameters = data["hyperparameters"]

        if "experiments" in data:
            for identifier, data in data["experiments"].items():
                self.experiments.append(Experiment.create(self, identifier, data))


class Descriptor(object):
    def __init__(self, descriptor_file_path) -> None:
        self.__models = []
        self._load_descriptor(descriptor_file_path)

    def _load_descriptor(self, descriptor_file_path):
        try:
            with open(descriptor_file_path) as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
        except FileNotFoundError:
            raise
        else:
            self._parse_descriptor(data)

    def _parse_descriptor(self, data):
        if not isinstance(data, dict):
            raise InvalidDescriptorError("The specified file is not a valid descriptor")

        try:
            models = data["models"]
        except KeyError:
            raise InvalidDescriptorError(
                'Descriptor file is missing the root element "models".'
            )
        else:
            if models is None:
                return

            for name, data in models.items():
                self.__models.append(Model(name, data))

    @property
    def models(self):
        return self.__models
