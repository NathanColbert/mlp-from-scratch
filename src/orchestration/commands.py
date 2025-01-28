from src.orchestration import messages
from src.common import shared_types, models
from dataclasses import dataclass


class Command(messages.Message):
    """What you want to do."""

    ...


@dataclass
class GetPrediction(Command):
    use_case: shared_types.UseCase
    user_id: shared_types.UserId


@dataclass
class PublishModelForInference(Command):
    use_case: shared_types.UseCase


@dataclass
class AddModelForInference(Command):
    use_case: shared_types.UseCase
    model: models.Model


@dataclass
class AddFeaturesForInference(Command):
    use_case: shared_types.UseCase
    feature_set: shared_types.FeatureSet


@dataclass
class TrainModel(Command):
    use_case: shared_types.UseCase


@dataclass
class AddModelToRegistry(Command):
    use_case: shared_types.UseCase
    model: models.Model


@dataclass
class PublishTrainingFeatures(Command):
    use_case: shared_types.UseCase


@dataclass
class PublishInferenceFeatures(Command):
    use_case: shared_types.UseCase
