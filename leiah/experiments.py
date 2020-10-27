import importlib

from sagemaker.tuner import (
    IntegerParameter,
    CategoricalParameter,
    ContinuousParameter,
)
from leiah.exceptions import DescriptorError


class Experiment(object):
    def __init__(self, model: object, identifier: str, data: dict) -> None:
        self.model = model
        self.identifier = identifier

        self._initialize(data)

    def _initialize(self, data):
        def get_properties():
            properties = dict()
            if (
                "estimator" in self.model.data
                and "properties" in self.model.data["estimator"]
            ):
                properties.update(self.model.data["estimator"]["properties"])

            if "estimator" in data and "properties" in data["estimator"]:
                properties.update(data["estimator"]["properties"])

            return properties

        def get_hyperparameters():
            hyperparameters = dict()
            if "hyperparameters" in self.model.data:
                hyperparameters.update(self.model.data["hyperparameters"])

            if "hyperparameters" in data:
                hyperparameters.update(data["hyperparameters"])

            return hyperparameters

        self.description = data.get("description", None)

        estimator_classname = (
            data["estimator"]["classname"]
            if "estimator" in data
            else self.model.data["estimator"]["classname"]
        )

        self.estimator = self._get_estimator(
            estimator_classname,
            model=self.model.name,
            experiment=self.identifier,
            properties=get_properties(),
            hyperparameters=get_hyperparameters(),
            ranges=data.get("ranges", None),
        )

    def _get_estimator(
        self, estimator, model, experiment, properties, hyperparameters, ranges
    ):
        identifiers = estimator.strip().split(".")
        class_name = identifiers[-1]
        module_name = ".".join(identifiers[:-1])

        if not module_name:
            raise DescriptorError(f'Error creating estimator "{estimator}"')

        try:
            module = importlib.import_module(module_name)
            class_ = getattr(module, class_name)
        except ModuleNotFoundError:
            raise DescriptorError(f'Error creating estimator "{estimator}"')
        except AttributeError:
            raise DescriptorError(f'Error creating estimator "{estimator}"')
        else:
            try:
                return class_(
                    model=model,
                    experiment=experiment,
                    hyperparameters=hyperparameters,
                    ranges=ranges,
                    **properties,
                )
            except TypeError as e:
                raise DescriptorError(
                    f'Error creating estimator "{estimator}". {str(e)}'
                )


class TrainingExperiment(Experiment):
    def process(self):
        self.estimator.fit()


class TuningExperiment(Experiment):
    def __init__(self, model: object, identifier: str, data: dict) -> None:
        super().__init__(model=model, identifier=identifier, data=data)

        self.hyperparameter_ranges = self._get_hyperparameter_ranges(
            data.get("hyperparameter_ranges", None)
        )

    def process(self):
        self.estimator.tune()

    def _get_hyperparameter_ranges(self, hyperparameter_ranges):
        if not hyperparameter_ranges:
            return dict()

        result = dict()
        for parameter, data in hyperparameter_ranges.items():
            if "type" not in data:
                raise DescriptorError(
                    f'Parameter "{parameter}" doesn\'t have a "type" specified'
                )

            if data["type"] == "categorical":
                result[parameter] = self._get_categorical_parameter(data)
            elif data["type"] == "integer":
                result[parameter] = self._get_integer_parameter(data)
            elif data["type"] == "continuous":
                result[parameter] = self._get_continuous_parameter(data)
            else:
                raise DescriptorError(
                    f"Parameter type \"{data['type']}\" is not supported"
                )

        return result

    def _get_categorical_parameter(self, data):
        if "values" not in data:
            raise DescriptorError(
                'The "values" attribute of a categorical parameter is required'
            )

        return CategoricalParameter(values=data["values"])

    def _get_integer_parameter(self, data):
        if "min_value" not in data:
            raise DescriptorError(
                'The "min_value" attribute of an integer parameter is required'
            )

        if "max_value" not in data:
            raise DescriptorError(
                'The "max_value" attribute of an integer parameter is required'
            )

        return IntegerParameter(
            min_value=data["min_value"],
            max_value=data["max_value"],
            scaling_type=data.get("scaling_type", "Auto"),
        )

    def _get_continuous_parameter(self, data):
        if "min_value" not in data:
            raise DescriptorError(
                'The "min_value" attribute of a continuous parameter is required'
            )

        if "max_value" not in data:
            raise DescriptorError(
                'The "max_value" attribute of a continuous parameter is required'
            )

        return ContinuousParameter(
            min_value=data["min_value"],
            max_value=data["max_value"],
            scaling_type=data.get("scaling_type", "Auto"),
        )
