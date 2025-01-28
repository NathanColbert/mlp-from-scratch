from collections.abc import Collection
from src.common import shared_types
from dataclasses import dataclass, field


@dataclass
class InMemoryFeatureStore:
    """
    example feature store structure:
    {
        use_case: {
            user_id: {
                feature_1: value,
                ...
            },
            ...
        },
        ...
    }
    example target structure
    {
        use_case: {
            user_id:  value,
                ...
            },
        },
        ...
    }
    """

    current_features: dict[str, shared_types.FeatureSet] = field(default_factory=dict)
    historical_features: dict[str, shared_types.FeatureSet] = field(
        default_factory=dict
    )
    targets: dict[str, dict[str, int | float]] = field(default_factory=dict)

    def get_current_features(
        self, user_id: str, use_case: str
    ) -> shared_types.FeatureVector:
        return self.current_features[use_case][user_id]

    def get_current_feature_set(self, use_case: str) -> shared_types.FeatureSet:
        return self.current_features[use_case]

    def get_historical_features(
        self, user_id: str, use_case: str
    ) -> shared_types.FeatureVector:
        return self.historical_features[use_case][user_id]

    def get_user_ids(self, use_case: str) -> Collection[str]:
        return self.historical_features[use_case].keys()

    def get_target(self, user_id: str, use_case: str) -> int | float:
        return self.targets[use_case][user_id]
