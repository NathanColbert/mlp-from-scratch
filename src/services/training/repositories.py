import abc

from dataclasses import dataclass, field

from src.common import shared_types, models
from typing import Collection


class FeatureRepository(abc.ABC):
    @abc.abstractmethod
    def get_features(
        self, user_id: str, use_case: str
    ) -> shared_types.FeatureVector: ...

    @abc.abstractmethod
    def get_user_ids(self, use_case: str) -> Collection[str]: ...

    @abc.abstractmethod
    def get_target(self, user_id: str, use_case: str) -> float: ...


class ModelRepository(abc.ABC):
    @abc.abstractmethod
    def get_model(self, use_case: str) -> models.MachineLearningModel: ...


@dataclass
class InMemoryModelRepository(ModelRepository):
    registry: dict[shared_types.UseCase, models.MachineLearningModel] = field(
        default_factory=dict
    )

    def get_model(self, use_case: str) -> models.MachineLearningModel:
        if use_case not in self.registry:
            msg = f"No model found for use case '{use_case}' in the model registry."
            raise ValueError(msg)

        return self.registry[use_case]


@dataclass
class InMemoryFeatureRepository(FeatureRepository):
    features: shared_types.FeatureSets = field(default_factory=dict)
    targets: shared_types.Targets = field(default_factory=dict)

    def get_features(self, user_id: str, use_case: str) -> shared_types.FeatureVector:
        return self.features[use_case][user_id]

    def get_user_ids(self, use_case: str) -> Collection[str]:
        return self.features[use_case].keys()

    def get_target(self, user_id: str, use_case: str) -> float:
        return self.targets[use_case][user_id]
