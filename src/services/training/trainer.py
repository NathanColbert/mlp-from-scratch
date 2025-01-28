from dataclasses import dataclass, field
from typing import Collection

from src.common.models import MachineLearningModel
from src.services.training import repositories


@dataclass
class Trainer:
    feature_repository: repositories.FeatureRepository = field(
        default_factory=repositories.InMemoryFeatureRepository
    )
    model_repository: repositories.ModelRepository = field(
        default_factory=repositories.InMemoryModelRepository
    )

    def train(self, use_case: str) -> None:
        model: MachineLearningModel = self.model_repository.get_model(use_case=use_case)
        user_ids: Collection[str] = self.feature_repository.get_user_ids(
            use_case=use_case
        )

        for user_id in user_ids:
            features: dict[str, int | float] = self.feature_repository.get_features(
                use_case=use_case, user_id=user_id
            )
            target: int | float = self.feature_repository.get_target(
                use_case=use_case, user_id=user_id
            )
            model.update_weights(features=features, target=target)
