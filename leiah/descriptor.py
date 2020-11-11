from pathlib import Path
import yaml
from yaml.parser import ParserError

from yaml.scanner import ScannerError
from leiah.processes import Training, Experiment
from leiah.exceptions import DescriptorError


class Model(object):
    def __init__(self, name: str, data: dict()) -> None:
        self.name = name
        self.data = data
        self.processes = dict()

        self._load_processes(
            self.data,
            "training",
            lambda model, identifier, data: Training(model, identifier, data),
        )

        self._load_processes(
            self.data,
            "experiments",
            lambda model, identifier, data: Experiment(model, identifier, data),
        )

    def _load_processes(self, data, section, factory_fn):
        if section not in data:
            return

        for identifier, data in data[section].items():
            identifier = str(identifier)

            self.processes[identifier] = factory_fn(
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
            raise DescriptorError(
                "Invalid descriptor source. Must be a dictionary, or "
                "the path of the descriptor file."
            )

    def process(self, processes=None):
        for process in self._get_processes(processes):
            process.run()

    def _get_processes(self, processes=None) -> list:
        if processes is None:
            processes = list(self.models.keys())
        elif isinstance(processes, str):
            processes = [processes]

        result = []

        for name in processes:
            identifier = name.split(".", 1)

            try:
                model = self.models[identifier[0]]
            except KeyError:
                raise DescriptorError(f'Process "{name}" was not found')

            if len(identifier) == 2:
                try:
                    process = model.processes[identifier[1]]
                except KeyError:
                    raise DescriptorError(f'Process "{name}" was not found')
                else:
                    result.append(process)
            else:
                for process in model.processes.values():
                    result.append(process)

        return result

    def _load_descriptor(self, descriptor_file_path) -> None:
        try:
            with open(descriptor_file_path) as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
        except FileNotFoundError:
            raise
        except ScannerError as e:
            raise DescriptorError(
                f"The specified file is not a valid descriptor. Error: {str(e)}"
            )
        except ParserError as e:
            raise DescriptorError(
                f"The specified file is not a valid descriptor. Error: {str(e)}"
            )
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
