import yaml

from pathlib import Path
from yaml.parser import ParserError
from yaml.scanner import ScannerError

from leiah.jobs import TrainingJob, HyperparameterTuningJob
from leiah.exceptions import DescriptorError


class Model(object):
    def __init__(self, name: str, data: dict()) -> None:
        self.name = name
        self.data = data
        self.jobs = dict()

        self._load_jobs(
            self.data,
            "training-jobs",
            lambda model, identifier, data: TrainingJob(model, identifier, data),
        )

        self._load_jobs(
            self.data,
            "hyperparameter-tuning-jobs",
            lambda model, identifier, data: HyperparameterTuningJob(
                model, identifier, data
            ),
        )

    def _load_jobs(self, data, section, factory_fn):
        if section not in data:
            return

        for identifier, data in data[section].items():
            identifier = str(identifier)

            self.jobs[identifier] = factory_fn(
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

    def run(self, jobs=None):
        for job in self._get_jobs(jobs):
            job.run()

    def _get_jobs(self, jobs=None) -> list:
        if jobs is None:
            jobs = list(self.models.keys())
        elif isinstance(jobs, str):
            jobs = [jobs]

        result = []

        for name in jobs:
            identifier = name.split(".", 1)

            try:
                model = self.models[identifier[0]]
            except KeyError:
                raise DescriptorError(f'Job "{name}" was not found')

            if len(identifier) == 2:
                try:
                    job = model.jobs[identifier[1]]
                except KeyError:
                    raise DescriptorError(f'Job "{name}" was not found')
                else:
                    result.append(job)
            else:
                for job in model.jobs.values():
                    result.append(job)

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
