from src.common import models
from src.services.inference.inference_engine import InferenceEngine
from src.services.inference.repositories import (
    InMemoryFeatureRepository,
    InMemoryModelRepository,
    ModelRepository,
)
from src.common.shared_types import FeatureVector, FeatureSet
from src.common.models import ConstantModel


class TestGetPrediction:
    def test_returns_expected_constant_value(self) -> None:
        stub_user = "unused"
        stub_use_case = "unused"
        expected_prediction = 0.85

        class StubModel(ConstantModel):
            def predict(self) -> float:
                return expected_prediction

        stub_model_repository = InMemoryModelRepository(
            registry={stub_use_case: StubModel()}
        )
        inference_engine = InferenceEngine(
            model_repository=stub_model_repository,
            feature_repository=InMemoryFeatureRepository(),
        )
        actual_prediction: float = inference_engine.get_prediction(
            user_id=stub_user, use_case=stub_use_case
        )

        assert actual_prediction == expected_prediction

    def test_produces_different_predictions_for_different_use_cases(self) -> None:
        stub_use_case_1 = "stub_use_case_1"
        stub_use_case_2 = "stub_use_case_2"
        stub_user_id = "stub_user_id"

        expected_prediction_use_case_1 = 0.111
        expected_prediction_use_case_2 = 0.9

        class StubModel1(ConstantModel):
            def predict(self) -> float:
                return expected_prediction_use_case_1

        class StubModel2(ConstantModel):
            def predict(self) -> float:
                return expected_prediction_use_case_2

        stub_model_repository: ModelRepository = InMemoryModelRepository(
            registry={
                stub_use_case_1: StubModel1(),
                stub_use_case_2: StubModel2(),
            }
        )

        inference_engine = InferenceEngine(
            model_repository=stub_model_repository,
            feature_repository=InMemoryFeatureRepository(),
        )

        prediction_1: float = inference_engine.get_prediction(
            user_id=stub_user_id,
            use_case=stub_use_case_1,
        )
        prediction_2: float = inference_engine.get_prediction(
            user_id=stub_user_id,
            use_case=stub_use_case_2,
        )

        assert prediction_1 == expected_prediction_use_case_1
        assert prediction_2 == expected_prediction_use_case_2

    def test_returns_expected_predictions_for_different_users(self) -> None:
        stub_use_case = "unused"
        stub_user_1 = "stub_user_1"
        stub_user_2 = "stub_user_2"
        stub_feature = "stub_feature"
        expected_prediction_user_1 = 0.5
        expected_prediction_user_2 = 0.75
        stub_features: dict[str, FeatureSet] = {
            stub_use_case: {
                stub_user_1: {stub_feature: 123},
                stub_user_2: {stub_feature: 1234},
            }
        }

        class StubModel(models.RulesBasedModel):
            def predict(self, features: FeatureVector) -> float:
                if features[stub_feature] == 123:
                    return expected_prediction_user_1
                elif features[stub_feature] == 1234:
                    return expected_prediction_user_2
                raise

        stub_feature_repository = InMemoryFeatureRepository(features=stub_features)
        stub_model_repository = InMemoryModelRepository(
            registry={stub_use_case: StubModel()}
        )

        inference_engine = InferenceEngine(
            model_repository=stub_model_repository,
            feature_repository=stub_feature_repository,
        )

        prediction_1: float = inference_engine.get_prediction(
            user_id=stub_user_1,
            use_case=stub_use_case,
        )
        prediction_2: float = inference_engine.get_prediction(
            user_id=stub_user_2,
            use_case=stub_use_case,
        )

        assert prediction_1 == expected_prediction_user_1
        assert prediction_2 == expected_prediction_user_2
