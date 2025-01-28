from src.common import shared_types, models
from dataclasses import dataclass, field


@dataclass
class InMemoryModelRegistry:
    registry: dict[shared_types.UseCase, models.Model] = field(default_factory=dict)

    def get_model(self, use_case: str) -> models.Model:
        if use_case not in self.registry:
            raise ValueError(
                f"No model found for use case '{use_case}' in the model registry."
            )

        return self.registry[use_case]
