from dataclasses import dataclass
from src.common import models
from src.common import shared_types


class Message: ...


class Command(Message):
    """What you want to do."""

    ...


class Event(Message):
    """What happened."""

    ...


class Error(Message):
    """What went wrong."""

    ...


@dataclass
class NewPrediction(Event):
    prediction: float


@dataclass
class ModelNotFound(Error):
    use_case: shared_types.UseCase


@dataclass
class NewModelForInference(Event):
    use_case: shared_types.UseCase
    model: models.Model


@dataclass
class InferenceMissingFeatures(Error):
    use_case: shared_types.UseCase


@dataclass
class NewFeaturesForInference(Event):
    use_case: shared_types.UseCase
    feature_set: shared_types.FeatureSet


@dataclass
class ModelRequiresTraining(Error):
    use_case: shared_types.UseCase


@dataclass
class NewTrainedModel(Event):
    use_case: shared_types.UseCase
    model: models.Model


@dataclass
class ModelTrainingMissingFeatures(Error):
    use_case: shared_types.UseCase


@dataclass
class NewTrainingFeatures(Event):
    feature_set: shared_types.FeatureSet


@dataclass
class NewInferenceFeatures(Event):
    feature_set: shared_types.FeatureSet
