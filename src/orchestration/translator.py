from dataclasses import dataclass

from src.orchestration import commands
from src.orchestration import messages


@dataclass
class MessageTranslator:
    def get_next_command(
        self, message: messages.Event | messages.Error
    ) -> commands.Command:
        match message:
            case messages.ModelNotFound():
                return commands.PublishModelForInference(use_case=message.use_case)
            case messages.NewModelForInference():
                return commands.AddModelForInference(
                    use_case=message.use_case, model=message.model
                )

            case messages.InferenceMissingFeatures():
                return commands.PublishInferenceFeatures(use_case=message.use_case)

            case messages.NewFeaturesForInference():
                return commands.AddFeaturesForInference(
                    use_case=message.use_case, feature_set=message.feature_set
                )
            case _:
                msg = f"No command available for message: {message}"
                raise NotImplementedError(msg)
