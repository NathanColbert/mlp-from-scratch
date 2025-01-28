from src.common import models
from src.services.model_registry.model_registry import InMemoryModelRegistry
import pytest


class StubModel(models.ConstantModel):
    def predict(self) -> float:
        return 0.5


class TestInMemoryModelRegistry:
    def test_returns_model_for_existing_use_case(self) -> None:
        stub_use_case = "stub_use_case"
        stub_model = StubModel()
        registry = InMemoryModelRegistry(registry={stub_use_case: stub_model})

        actual_model = registry.get_model(use_case=stub_use_case)

        assert actual_model == stub_model

    def test_raises_error_for_unknown_use_case(self) -> None:
        registry = InMemoryModelRegistry()

        with pytest.raises(ValueError, match="No model found.*"):
            registry.get_model(use_case="unknown")

    def test_handles_multiple_use_cases(self) -> None:
        stub_use_case_1 = "stub_use_case_1"
        stub_use_case_2 = "stub_use_case_2"
        stub_model_1 = StubModel()
        stub_model_2 = StubModel()

        registry = InMemoryModelRegistry(
            registry={stub_use_case_1: stub_model_1, stub_use_case_2: stub_model_2}
        )

        assert registry.get_model(use_case=stub_use_case_1) == stub_model_1
        assert registry.get_model(use_case=stub_use_case_2) == stub_model_2
