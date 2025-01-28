from dataclasses import dataclass
from typing import Callable

from src.orchestration import commands, messages


@dataclass
class CommandHandler:
    handlers: dict[
        type[commands.Command],
        Callable[[commands.Command], messages.Event | messages.Error | None],
    ]

    def handle(
        self, command: commands.Command
    ) -> messages.Event | messages.Error | None:
        handler = self.get_handler(command=command)
        return handler(command)

    def get_handler(
        self, command: commands.Command
    ) -> Callable[[commands.Command], messages.Event | messages.Error | None]:
        handler = self.handlers.get(type(command))
        if not handler:
            msg = f"No handler exists for command: {command}"
            raise NotImplementedError(msg)
        return handler
