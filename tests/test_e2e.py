from src.orchestration import (
    command_handler,
    commands,
    handlers,
    message_bus,
    translator,
)
from src.common import shared_types, models
from src.common.models import MachineLearningModel
from src.services.inference.inference_engine import InferenceEngine
from src.services.model_registry.model_registry import InMemoryModelRegistry
from src.services.features.feature_store import InMemoryFeatureStore


def test_inference_engine_can_use_trained_model_for_prediction() -> None:
    stub_user_id = "stub_user_id"
    stub_use_case = "stub_use_case"
    stub_feature_name = "stub_feature_name"

    # Set up feature store
    stub_current_feature_value = 1
    stub_historical_feature_value = 0
    stub_target = 5
    stub_current_features: shared_types.FeatureVector = {
        stub_feature_name: stub_current_feature_value
    }
    stub_current_feature_set: dict[str, shared_types.FeatureSet] = {
        stub_use_case: {stub_user_id: stub_current_features}
    }
    stub_historical_features: shared_types.FeatureVector = {
        stub_feature_name: stub_historical_feature_value
    }
    stub_historical_feature_set: dict[str, shared_types.FeatureSet] = {
        stub_use_case: {stub_user_id: stub_historical_features}
    }
    stub_targets: shared_types.Targets = {stub_use_case: {stub_user_id: stub_target}}
    feature_store = InMemoryFeatureStore(
        current_features=stub_current_feature_set,
        historical_features=stub_historical_feature_set,
        targets=stub_targets,
    )

    expected_prediction: float = 0.42

    # Setup Model Registry
    class StubModel(MachineLearningModel):
        """
        Naive model:

        update_weights = each feature + target
        prediction = pairwise multiplication of weights and features.

        """

        def update_weights(
            self, features: shared_types.FeatureVector, target: shared_types.Target
        ) -> None: ...

        def predict(self, features: shared_types.FeatureVector) -> float:
            return expected_prediction

    stub_registry: dict[shared_types.UseCase, models.Model] = {
        stub_use_case: StubModel()
    }
    model_registry = InMemoryModelRegistry(registry=stub_registry)

    # Setup Inference Engine
    inference_engine = InferenceEngine()

    message_translator = translator.MessageTranslator()
    handler = command_handler.CommandHandler(
        handlers={
            commands.GetPrediction: lambda c: handlers.get_prediction(
                cmd=c, predictor=inference_engine
            ),
            commands.PublishModelForInference: lambda c: handlers.publish_model_for_inference(
                cmd=c, model_registry=model_registry
            ),
            commands.AddModelForInference: lambda c: handlers.add_model_for_inference(
                cmd=c, model_registry=inference_engine
            ),
            commands.PublishInferenceFeatures: lambda c: handlers.publish_features_for_inference(
                cmd=c, feature_store=feature_store
            ),
            commands.AddFeaturesForInference: lambda c: handlers.add_features_for_inference(
                cmd=c, feature_repository=inference_engine
            ),
        }
    )

    command = commands.GetPrediction(use_case=stub_use_case, user_id=stub_user_id)

    bus = message_bus.MessageBus(translator=message_translator, handler=handler)

    actual_prediction: float | None = bus.dispatch(command=command)

    assert actual_prediction == expected_prediction
