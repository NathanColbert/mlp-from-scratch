import abc

from dataclasses import dataclass, field

from src.common import exceptions, shared_types, models


class FeatureRepository(abc.ABC):
    @abc.abstractmethod
    def get_features(
        self, user_id: str, use_case: str
    ) -> shared_types.FeatureVector: ...
    @abc.abstractmethod
    def add_features(
        self, use_case: str, feature_set: shared_types.FeatureSet
    ) -> None: ...


class ModelRepository(abc.ABC):
    @abc.abstractmethod
    def get_model(self, use_case: str) -> models.Model: ...
    @abc.abstractmethod
    def add_model(self, use_case: str, model: models.Model) -> None: ...


@dataclass
class InMemoryModelRepository(ModelRepository):
    registry: dict[shared_types.UseCase, models.Model] = field(default_factory=dict)

    def get_model(self, use_case: str) -> models.Model:
        if use_case not in self.registry:
            msg: str = f"Use case {use_case} not found in model registry."
            raise exceptions.ModelNotFound(msg)
        return self.registry[use_case]

    def add_model(self, use_case: str, model: models.Model) -> None:
        self.registry[use_case] = model


@dataclass
class InMemoryFeatureRepository(FeatureRepository):
    features: shared_types.FeatureSets = field(default_factory=dict)

    def get_features(self, user_id: str, use_case: str) -> shared_types.FeatureVector:
        if use_case not in self.features:
            msg: str = f"Use case {use_case} not found in features."
            raise exceptions.FeatureSetNotFound(msg)
        feature_set: shared_types.FeatureSet = self.features[use_case]

        if user_id not in feature_set:
            msg: str = (
                f"User ID: {user_id} not found in features for use case: {use_case}."
            )
            raise ValueError(msg)

        return feature_set[user_id]

    def add_features(self, use_case: str, feature_set: shared_types.FeatureSet) -> None:
        self.features[use_case] = feature_set
