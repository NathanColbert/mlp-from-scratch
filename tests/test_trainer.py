from typing import Sequence
import pytest

from dataclasses import dataclass, field
from src.common.models import MachineLearningModel
from src.services.training.repositories import (
    InMemoryFeatureRepository,
    InMemoryModelRepository,
    ModelRepository,
)
from src.services.training.trainer import Trainer
from src.common import shared_types


@dataclass
class StubModel(MachineLearningModel):
    expected_weights: Sequence[int | float] = field(default_factory=list)
    update_calls: list[tuple[shared_types.FeatureVector, shared_types.Target]] = field(
        default_factory=list
    )

    def update_weights(
        self, features: shared_types.FeatureVector, target: shared_types.Target
    ) -> None:
        self.update_calls.append((features, target))
        self.weights = self.expected_weights

    def predict(self, features: shared_types.FeatureVector) -> float:
        raise NotImplementedError()


class TestTrainer:
    def test_trains_on_all_users(self) -> None:
        """Verify trainer processes all users in the training set."""
        stub_use_case = "stub_use_case"

        stub_feature_repository = InMemoryFeatureRepository(
            features={
                stub_use_case: {
                    "user_1": {"feature1": 1.0},
                    "user_2": {"feature1": 2.0},
                }
            },
            targets={stub_use_case: {"user_1": 0.0, "user_2": 1.0}},
        )

        stub_model = StubModel()
        stub_model_registry: ModelRepository = InMemoryModelRepository(
            registry={stub_use_case: stub_model}
        )

        trainer = Trainer(
            model_repository=stub_model_registry,
            feature_repository=stub_feature_repository,
        )
        trainer.train(use_case=stub_use_case)

        assert len(stub_model.update_calls) == 2
        assert stub_model.update_calls[0] == ({"feature1": 1.0}, 0.0)
        assert stub_model.update_calls[1] == ({"feature1": 2.0}, 1.0)

    def test_train_pushes_updated_model_to_model_repository(self) -> None:
        stub_use_case = "stub_use_case"

        stub_feature_repository = InMemoryFeatureRepository(
            features={
                stub_use_case: {
                    "user_1": {"feature1": 1.0},
                    "user_2": {"feature1": 2.0},
                }
            },
            targets={stub_use_case: {"user_1": 0.0, "user_2": 1.0}},
        )
        expected_weights = [4, 5, 6]

        stub_model: MachineLearningModel = StubModel(
            weights=[1, 2, 3], expected_weights=expected_weights
        )
        stub_model_registry: ModelRepository = InMemoryModelRepository(
            registry={stub_use_case: stub_model}
        )

        trainer = Trainer(
            model_repository=stub_model_registry,
            feature_repository=stub_feature_repository,
        )
        trainer.train(use_case=stub_use_case)
        actual_weights: Sequence[int | float] = stub_model_registry.get_model(
            use_case=stub_use_case
        ).weights

        assert actual_weights == expected_weights

    def test_raises_error_for_unknown_use_case(self) -> None:
        """Test that training with an unknown use case raises an error."""
        stub_trainer = Trainer(
            model_repository=InMemoryModelRepository(),
            feature_repository=InMemoryFeatureRepository(),
        )

        with pytest.raises(expected_exception=ValueError, match="No model found.*"):
            stub_trainer.train(use_case="unknown_usecase")
