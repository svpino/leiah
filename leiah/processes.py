import importlib

from sagemaker.tuner import (
    IntegerParameter,
    CategoricalParameter,
    ContinuousParameter,
)
from leiah.exceptions import DescriptorError


class Process(object):
    def __init__(self, model: object, identifier: str, data: dict) -> None:
        self.model = model
        self.identifier = identifier

        self._initialize(data)

    def _initialize(self, data):
        def get_properties():
            properties = dict()
            properties.update(self.model.data)
            properties.update(data)

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
            data["estimator"] if "estimator" in data else self.model.data["estimator"]
        )

        self.estimator = self._get_estimator(
            estimator_classname,
            model=self.model.name,
            process=self.identifier,
            properties=get_properties(),
            hyperparameters=get_hyperparameters(),
        )

    def _get_estimator(self, estimator, model, process, properties, hyperparameters):
        def remove_attribute(properties, attribute):
            if attribute in properties:
                del properties[attribute]

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
            remove_attribute(properties, "model")
            remove_attribute(properties, "process")
            remove_attribute(properties, "hyperparameters")

            try:
                return class_(
                    model=model,
                    process=process,
                    hyperparameters=hyperparameters,
                    **properties,
                )
            except TypeError as e:
                raise DescriptorError(
                    f'Error creating estimator "{estimator}". {str(e)}'
                )


class Training(Process):
    def run(self):
        self.estimator.fit()


class Experiment(Process):
    def __init__(self, model: object, identifier: str, data: dict) -> None:
        super().__init__(model=model, identifier=identifier, data=data)

        self.hyperparameter_ranges = self._get_hyperparameter_ranges(
            data.get("hyperparameter_ranges", None)
        )

        self.attributes = data
        self.attributes["hyperparameter_ranges"] = self._get_hyperparameter_ranges(
            data.get("hyperparameter_ranges", None)
        )

    def run(self):
        self.estimator.tune(**self.attributes)

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
