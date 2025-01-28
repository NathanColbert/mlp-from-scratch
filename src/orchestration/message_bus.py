from collections import deque
from dataclasses import dataclass, field

from src.orchestration import commands
from src.orchestration import messages
from src.orchestration.command_handler import CommandHandler
from src.orchestration.translator import MessageTranslator


@dataclass
class MessageBus:
    translator: MessageTranslator
    handler: CommandHandler
    log: list[messages.Message] = field(default_factory=list)

    def dispatch(self, command: commands.Command) -> float | None:
        remaining_queue: deque[commands.Command] = deque([command])

        while remaining_queue:
            current_command: commands.Command = remaining_queue.popleft()
            self.log.append(current_command)

            response: messages.Event | messages.Error | None = self.handler.handle(
                command=current_command
            )

            if response:
                if isinstance(response, messages.NewPrediction):
                    return response.prediction

                self.log.append(response)
                next_command = self.translator.get_next_command(response)
                if isinstance(response, messages.Error):
                    remaining_queue.appendleft(current_command)
                remaining_queue.appendleft(next_command)

        return None
