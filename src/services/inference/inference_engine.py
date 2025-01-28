from dataclasses import dataclass, field
from src.common import shared_types
from src.common import models
from src.services.inference.repositories import (
    FeatureRepository,
    ModelRepository,
    InMemoryFeatureRepository,
    InMemoryModelRepository,
)


@dataclass
class InferenceEngine:
    feature_repository: FeatureRepository = field(
        default_factory=InMemoryFeatureRepository
    )
    model_repository: ModelRepository = field(default_factory=InMemoryModelRepository)

    def get_prediction(self, user_id: str, use_case: str) -> float:
        model: models.Model = self.model_repository.get_model(use_case=use_case)

        if isinstance(model, models.RulesBasedModel):
            features: shared_types.FeatureVector = self.feature_repository.get_features(
                user_id=user_id, use_case=use_case
            )
            return model.predict(features=features)
        return model.predict()

    def add_model(self, use_case: shared_types.UseCase, model: models.Model) -> None:
        self.model_repository.add_model(use_case=use_case, model=model)

    def add_feature_set(
        self, use_case: shared_types.UseCase, feature_set: shared_types.FeatureSet
    ) -> None:
        self.feature_repository.add_features(use_case=use_case, feature_set=feature_set)
