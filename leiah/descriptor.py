from pathlib import Path
import yaml

from yaml.scanner import ScannerError
from leiah.experiments import TrainingExperiment, TuningExperiment
from leiah.exceptions import DescriptorError


class Model(object):
    def __init__(self, name: str, data: dict()) -> None:
        self.name = name
        self.experiments = dict()
        self.data = data

        if "experiments" in data:
            for identifier, data in data["experiments"].items():
                identifier = str(identifier)
                type = data.get("type", "training")

                if type == "training":
                    experiment = TrainingExperiment(
                        model=self, identifier=identifier, data=data
                    )
                elif type == "tuning":
                    experiment = TuningExperiment(
                        model=self, identifier=identifier, data=data
                    )
                else:
                    raise DescriptorError(f'Experiment type "{type}" is not supported.')

                self.experiments[identifier] = experiment


class Descriptor(object):
    def __init__(self, descriptor) -> None:
        self.__models = dict()

        if isinstance(descriptor, dict):
            self._parse_descriptor(data=descriptor)
        elif isinstance(descriptor, str) or isinstance(descriptor, Path):
            self._load_descriptor(descriptor_file_path=descriptor)
        else:
            raise DescriptorError(
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
                raise DescriptorError(f'Experiment "{name}" was not found')

            if len(identifier) == 2:
                try:
                    experiment = model.experiments[identifier[1]]
                except KeyError:
                    raise DescriptorError(f'Experiment "{name}" was not found')
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
            raise DescriptorError("The specified file is not a valid descriptor")
        else:
            self._parse_descriptor(data)

    def _parse_descriptor(self, data: dict()) -> None:
        if not isinstance(data, dict):
            raise DescriptorError("The specified file is not a valid descriptor")

        try:
            descriptor_models = data["models"]
        except KeyError:
            raise DescriptorError(
                'Descriptor file is missing the root element "models".'
            )
        else:
            if descriptor_models is None:
                return

            for name, data in descriptor_models.items():
                self.__models[str(name)] = Model(str(name), data)

    @property
    def models(self) -> dict:
        return self.__models


