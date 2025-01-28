import abc
from src.common import shared_types
from dataclasses import dataclass, field
from typing import Sequence


class ConstantModel(abc.ABC):
    @abc.abstractmethod
    def predict(self) -> float: ...


class RulesBasedModel(abc.ABC):
    @abc.abstractmethod
    def predict(self, features: shared_types.FeatureVector) -> float: ...


@dataclass
class MachineLearningModel(RulesBasedModel):
    weights: Sequence[int | float] = field(default_factory=list)

    @abc.abstractmethod
    def update_weights(
        self, features: shared_types.FeatureVector, target: shared_types.Target
    ) -> None: ...


Model = ConstantModel | RulesBasedModel | MachineLearningModel
