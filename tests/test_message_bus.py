from dataclasses import dataclass
from src.orchestration import commands
from src.orchestration import messages
from src.orchestration.command_handler import CommandHandler
from src.orchestration.translator import MessageTranslator
from src.orchestration.message_bus import MessageBus


class TestMessageBus:
    def test_commands_are_handled_entirely_before_queue_proceeds(self):
        @dataclass
        class FirstCommand(commands.Command): ...

        @dataclass
        class FirstCommandComplete(commands.Command): ...

        @dataclass
        class ErrorResolvingCommand(commands.Command): ...

        @dataclass
        class FirstError(messages.Error): ...

        @dataclass
        class FirstEvent(messages.Event): ...

        event_generator = (event for event in (FirstError(), FirstEvent()))

        def stub_command_handler(cmd: FirstCommand):
            return next(event_generator)

        stub_handler = CommandHandler(
            handlers={
                FirstCommand: stub_command_handler,
                ErrorResolvingCommand: lambda x: None,
                FirstCommandComplete: lambda x: None,
            }
        )

        class StubTranslator(MessageTranslator):
            def get_next_command(
                self, message: messages.Event | messages.Error
            ) -> commands.Command:
                match message:
                    case FirstError():
                        return ErrorResolvingCommand()
                    case FirstEvent():
                        return FirstCommandComplete()
                    case _:
                        raise NotImplementedError

        bus = MessageBus(translator=StubTranslator(), handler=stub_handler)

        bus.dispatch(command=FirstCommand())

        assert bus.log == [
            FirstCommand(),
            FirstError(),
            ErrorResolvingCommand(),
            FirstCommand(),
            FirstEvent(),
            FirstCommandComplete(),
        ]
