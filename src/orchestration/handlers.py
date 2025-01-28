from typing import Protocol
from src.common import shared_types
from src.common import exceptions, models
from src.orchestration import commands, messages


class CanGetPrediction(Protocol):
    def get_prediction(self, user_id: str, use_case: str) -> float: ...


class CanGetModel(Protocol):
    def get_model(self, use_case: str) -> models.Model: ...


class CanAddModel(Protocol):
    def add_model(
        self, use_case: shared_types.UseCase, model: models.Model
    ) -> None: ...


class CanGetCurrentFeatureSet(Protocol):
    def get_current_feature_set(
        self,
        use_case: shared_types.UseCase,
    ) -> shared_types.FeatureSet: ...


class CanAddFeatureSet(Protocol):
    def add_feature_set(
        self, use_case: shared_types.UseCase, feature_set: shared_types.FeatureSet
    ) -> None: ...


def get_prediction(cmd: commands.GetPrediction, predictor: CanGetPrediction):
    try:
        prediction: float = predictor.get_prediction(
            user_id=cmd.user_id, use_case=cmd.use_case
        )
    except exceptions.ModelNotFound:
        return messages.ModelNotFound(use_case=cmd.use_case)
    except exceptions.FeatureSetNotFound:
        return messages.InferenceMissingFeatures(use_case=cmd.use_case)
    return messages.NewPrediction(prediction=prediction)


def publish_model_for_inference(
    cmd: commands.PublishModelForInference, model_registry: CanGetModel
) -> messages.NewModelForInference:
    model: models.Model = model_registry.get_model(use_case=cmd.use_case)
    return messages.NewModelForInference(use_case=cmd.use_case, model=model)


def add_model_for_inference(
    cmd: commands.AddModelForInference, model_registry: CanAddModel
) -> None:
    model_registry.add_model(use_case=cmd.use_case, model=cmd.model)


def publish_features_for_inference(
    cmd: commands.PublishInferenceFeatures, feature_store: CanGetCurrentFeatureSet
) -> messages.NewFeaturesForInference:
    feature_set: shared_types.FeatureSet = feature_store.get_current_feature_set(
        use_case=cmd.use_case
    )
    return messages.NewFeaturesForInference(
        use_case=cmd.use_case, feature_set=feature_set
    )


def add_features_for_inference(
    cmd: commands.AddFeaturesForInference, feature_repository: CanAddFeatureSet
) -> None:
    feature_repository.add_feature_set(
        use_case=cmd.use_case, feature_set=cmd.feature_set
    )
