from pathlib import Path
import yaml
import importlib

from dataclasses import dataclass
from yaml.scanner import ScannerError

from leiah.exceptions import (
    EstimatorMissingPropertyError, ExperimentNotFoundError,
    InvalidDescriptorError,
    InvalidEstimatorError,
)


@dataclass
class Experiment(object):
    model: object
    identifier: str
    estimator: object
    description: str = None

    @classmethod
    def create(cls, model, identifier: str, data: dict()):
        def get_properties():
            properties = dict()
            if "estimator" in model.data and "properties" in model.data["estimator"]:
                properties.update(model.data["estimator"]["properties"])

            if "estimator" in data and "properties" in data["estimator"]:
                properties.update(data["estimator"]["properties"])

            return properties

        def get_hyperparameters():
            hyperparameters = dict()
            if "hyperparameters" in model.data:
                hyperparameters.update(model.data["hyperparameters"])

            if "hyperparameters" in data:
                hyperparameters.update(data["hyperparameters"])

            return hyperparameters

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

        estimator_classname = (
            data["estimator"]["classname"]
            if "estimator" in data
            else model.data["estimator"]["classname"]
        )

        experiment.estimator = _get_estimator(
            estimator_classname,
            properties=get_properties(),
            hyperparameters=get_hyperparameters(),
        )

        return experiment


@dataclass
class Training(Experiment):
    def process(self):
        return self.estimator.fit()


@dataclass
class Tuning(Experiment):
    def process(self):
        return self.estimator.tune()


class Model(object):
    def __init__(self, name: str, data: dict()) -> None:
        self.name = name
        self.experiments = dict()
        self.data = data

        if "experiments" in data:
            for identifier, data in data["experiments"].items():
                identifier = str(identifier)
                self.experiments[identifier] = Experiment.create(
                    model=self, identifier=identifier, data=data
                )


class Descriptor(object):
    def __init__(self, descriptor) -> None:
        self.__models = dict()

        if isinstance(descriptor, dict):
            self._parse_descriptor(data=descriptor)
        elif isinstance(descriptor, str) or isinstance(descriptor, Path):
            self._load_descriptor(descriptor_file_path=descriptor)
        else:
            raise InvalidDescriptorError(
                "Invalid descriptor source. Must be a dictionary, or "
                "the path of the descriptor file."
            )

    def process(self, experiments=None):
        for experiment in self._get_experiments(experiments):
            experiment.process()

    def _get_experiments(self, experiments=None) -> list:
        if experiments is None:
            experiments = list(self.models.keys())
        elif isinstance(experiments, str):
            experiments = [experiments]

        result = []

        for name in experiments:
            identifier = name.split(".", 1)

            try:
                model = self.models[identifier[0]]
            except KeyError:
                raise ExperimentNotFoundError(name)

            if len(identifier) == 2:
                try:
                    experiment = model.experiments[identifier[1]]
                except KeyError:
                    raise ExperimentNotFoundError(name)
                else:
                    result.append(experiment)
            else:
                for experiment in model.experiments.values():
                    result.append(experiment)

        return result

    def _load_descriptor(self, descriptor_file_path) -> None:
        try:
            with open(descriptor_file_path) as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
        except FileNotFoundError:
            raise
        except ScannerError:
            raise InvalidDescriptorError("The specified file is not a valid descriptor")
        else:
            self._parse_descriptor(data)

    def _parse_descriptor(self, data: dict()) -> None:
        if not isinstance(data, dict):
            raise InvalidDescriptorError("The specified file is not a valid descriptor")

        try:
            descriptor_models = data["models"]
        except KeyError:
            raise InvalidDescriptorError(
                'Descriptor file is missing the root element "models".'
            )
        else:
            if descriptor_models is None:
                return

            for name, data in descriptor_models.items():
                self.__models[name] = Model(name, data)

    @property
    def models(self) -> dict:
        return self.__models


def _get_estimator(estimator, properties, hyperparameters):
    identifiers = estimator.split(".")
    class_name = identifiers[-1]
    module_name = ".".join(identifiers[:-1])

    try:
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)
    except ModuleNotFoundError:
        raise InvalidEstimatorError(estimator)
    except AttributeError:
        raise InvalidEstimatorError(estimator)
    else:
        try:
            return class_(**properties, hyperparameters=hyperparameters)
        except TypeError as e:
            raise EstimatorMissingPropertyError(estimator, e)
