import pytest
from src.common import shared_types
from src.services.features.feature_store import InMemoryFeatureStore


class TestFeatureStore:
    def test_get_current_features_returns_feature_vector(self) -> None:
        """Test that get_current_features returns the correct feature vector for a user."""
        stub_use_case = "stub_use_case"
        stub_user_id = "stub_user"
        stub_features: shared_types.FeatureVector = {
            "watch_count": 10,
            "rating_avg": 4.5,
        }

        feature_store = InMemoryFeatureStore(
            current_features={stub_use_case: {stub_user_id: stub_features}}
        )

        actual_features: shared_types.FeatureVector = (
            feature_store.get_current_features(
                user_id=stub_user_id, use_case=stub_use_case
            )
        )

        assert actual_features == stub_features

    def test_get_historical_features_returns_feature_vector(self) -> None:
        """Test that get_historical_features returns the correct feature vector for a user."""
        stub_use_case = "stub_use_case"
        stub_user_id = "stub_user"
        stub_features: shared_types.FeatureVector = {
            "watch_count": 2,
            "rating_avg": 3.0,
        }

        feature_store = InMemoryFeatureStore(
            historical_features={stub_use_case: {stub_user_id: stub_features}}
        )

        actual_features: shared_types.FeatureVector = (
            feature_store.get_historical_features(
                user_id=stub_user_id, use_case=stub_use_case
            )
        )

        assert actual_features == stub_features

    def test_current_and_historical_features_differ(self) -> None:
        """Test that current and historical features can be different for the same user."""
        stub_use_case = "movie_recommendation"
        stub_user_id = "stub_user"

        stub_current_features: shared_types.FeatureVector = {
            "watch_count": 10,
            "rating_avg": 4.5,
        }
        stub_historical_features: shared_types.FeatureVector = {
            "watch_count": 2,
            "rating_avg": 3.0,
        }

        feature_store = InMemoryFeatureStore(
            current_features={stub_use_case: {stub_user_id: stub_current_features}},
            historical_features={
                stub_use_case: {stub_user_id: stub_historical_features}
            },
        )

        current: shared_types.FeatureVector = feature_store.get_current_features(
            user_id=stub_user_id, use_case=stub_use_case
        )
        historical: shared_types.FeatureVector = feature_store.get_historical_features(
            user_id=stub_user_id, use_case=stub_use_case
        )

        assert current != historical
        assert current == stub_current_features
        assert historical == stub_historical_features

    def test_get_user_ids_returns_users_from_historical_features(self) -> None:
        """Test that get_user_ids returns users from historical features."""
        stub_use_case = "stub_use_case"
        stub_users: set[str] = {"user1", "user2"}

        feature_store = InMemoryFeatureStore(
            historical_features={
                stub_use_case: {user_id: {"feature1": 1.0} for user_id in stub_users}
            }
        )

        actual_users = set(feature_store.get_user_ids(use_case=stub_use_case))

        assert actual_users == stub_users

    def test_raises_key_error_for_missing_current_features(self) -> None:
        """Test that accessing missing current features raises KeyError."""
        feature_store = InMemoryFeatureStore()

        with pytest.raises(expected_exception=KeyError):
            feature_store.get_current_features(
                user_id="stub_user", use_case="missing_use_case"
            )

    def test_raises_key_error_for_missing_historical_features(self) -> None:
        """Test that accessing missing historical features raises KeyError."""
        feature_store = InMemoryFeatureStore()

        with pytest.raises(expected_exception=KeyError):
            feature_store.get_historical_features(
                user_id="stub_user", use_case="missing_use_case"
            )
