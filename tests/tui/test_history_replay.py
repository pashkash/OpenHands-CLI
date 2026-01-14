"""Tests for conversation history replay (Issue #204)."""

from unittest.mock import Mock

from openhands_cli.tui.textual_app import OpenHandsApp


class TestExtractUserMessageText:
    """Tests for _extract_user_message_text method."""

    def test_extracts_string_content(self):
        """String content is returned directly."""
        app = OpenHandsApp.__new__(OpenHandsApp)

        event = Mock()
        event.llm_message = Mock()
        event.llm_message.content = "Hello, world!"

        result = app._extract_user_message_text(event)
        assert result == "Hello, world!"

    def test_extracts_text_from_text_content_list(self):
        """TextContent parts with .text attribute are extracted."""
        app = OpenHandsApp.__new__(OpenHandsApp)

        text_part = Mock()
        text_part.text = "First part"

        text_part2 = Mock()
        text_part2.text = "Second part"

        event = Mock()
        event.llm_message = Mock()
        event.llm_message.content = [text_part, text_part2]

        result = app._extract_user_message_text(event)
        assert result == "First part\nSecond part"

    def test_extracts_string_parts_from_list(self):
        """Plain string parts in list are extracted."""
        app = OpenHandsApp.__new__(OpenHandsApp)

        event = Mock()
        event.llm_message = Mock()
        event.llm_message.content = ["Hello", "World"]

        result = app._extract_user_message_text(event)
        assert result == "Hello\nWorld"

    def test_skips_image_content_parts(self):
        """ImageContent parts (no .text) are skipped."""
        app = OpenHandsApp.__new__(OpenHandsApp)

        text_part = Mock()
        text_part.text = "Text message"

        image_part = Mock(spec=[])  # No .text attribute

        event = Mock()
        event.llm_message = Mock()
        event.llm_message.content = [text_part, image_part]

        result = app._extract_user_message_text(event)
        assert result == "Text message"

    def test_returns_none_for_empty_content(self):
        """Empty content returns None."""
        app = OpenHandsApp.__new__(OpenHandsApp)

        event = Mock()
        event.llm_message = Mock()
        event.llm_message.content = []

        result = app._extract_user_message_text(event)
        assert result is None

    def test_returns_none_for_no_llm_message(self):
        """No llm_message returns None."""
        app = OpenHandsApp.__new__(OpenHandsApp)

        event = Mock()
        event.llm_message = None

        result = app._extract_user_message_text(event)
        assert result is None

    def test_returns_none_for_none_content(self):
        """None content returns None."""
        app = OpenHandsApp.__new__(OpenHandsApp)

        event = Mock()
        event.llm_message = Mock()
        event.llm_message.content = None

        result = app._extract_user_message_text(event)
        assert result is None


class TestMountUserMessage:
    """Tests for _mount_user_message method."""

    def test_mounts_static_widget_with_correct_format(self):
        """User message is mounted as Static widget with '> ' prefix."""
        from textual.widgets import Static

        app = OpenHandsApp.__new__(OpenHandsApp)
        app.main_display = Mock()

        app._mount_user_message("Hello, agent!")

        app.main_display.mount.assert_called_once()
        widget = app.main_display.mount.call_args[0][0]
        # Check it's a Static widget with user-message class
        assert isinstance(widget, Static)
        assert "user-message" in widget.classes


class TestReplayConversationHistory:
    """Tests for _replay_conversation_history method."""

    def test_skips_replay_when_no_conversation(self):
        """No replay when runner has no conversation."""
        app = OpenHandsApp.__new__(OpenHandsApp)

        runner = Mock()
        runner.conversation = None
        visualizer = Mock()

        app._replay_conversation_history(runner, visualizer)

        visualizer.on_event.assert_not_called()

    def test_skips_replay_when_no_state(self):
        """No replay when conversation has no state."""
        app = OpenHandsApp.__new__(OpenHandsApp)

        runner = Mock()
        runner.conversation = Mock()
        runner.conversation.state = None
        visualizer = Mock()

        app._replay_conversation_history(runner, visualizer)

        visualizer.on_event.assert_not_called()

    def test_skips_replay_when_no_events(self):
        """No replay when state has no events."""
        app = OpenHandsApp.__new__(OpenHandsApp)

        runner = Mock()
        runner.conversation = Mock()
        runner.conversation.state = Mock()
        runner.conversation.state.events = []
        visualizer = Mock()

        app._replay_conversation_history(runner, visualizer)

        visualizer.on_event.assert_not_called()

    def test_replays_user_messages_as_static_widgets(self):
        """User MessageEvents are rendered as Static widgets."""
        from openhands.sdk.event import MessageEvent

        app = OpenHandsApp.__new__(OpenHandsApp)
        app.call_from_thread = Mock(side_effect=lambda f, *args: f(*args))
        app.main_display = Mock()

        # Create a user message event
        user_event = Mock(spec=MessageEvent)
        user_event.llm_message = Mock()
        user_event.llm_message.role = "user"
        user_event.llm_message.content = "Hello!"

        runner = Mock()
        runner.conversation = Mock()
        runner.conversation.state = Mock()
        runner.conversation.state.events = [user_event]
        visualizer = Mock()

        app._replay_conversation_history(runner, visualizer)

        # User message should be mounted, not sent to visualizer
        app.main_display.mount.assert_called_once()
        visualizer.on_event.assert_not_called()

    def test_replays_agent_events_via_visualizer(self):
        """Non-user events are rendered via visualizer."""
        from openhands.sdk.event import ActionEvent

        app = OpenHandsApp.__new__(OpenHandsApp)
        app.call_from_thread = Mock(side_effect=lambda f, *args: f(*args))

        # Create an agent action event
        agent_event = Mock(spec=ActionEvent)

        runner = Mock()
        runner.conversation = Mock()
        runner.conversation.state = Mock()
        runner.conversation.state.events = [agent_event]
        visualizer = Mock()

        app._replay_conversation_history(runner, visualizer)

        # Agent event should go to visualizer
        visualizer.on_event.assert_called_once_with(agent_event)

    def test_replays_mixed_events_in_order(self):
        """Mixed user and agent events are replayed in order."""
        from openhands.sdk.event import ActionEvent, MessageEvent

        app = OpenHandsApp.__new__(OpenHandsApp)
        app.call_from_thread = Mock(side_effect=lambda f, *args: f(*args))
        app.main_display = Mock()

        # Create mixed events
        user_event = Mock(spec=MessageEvent)
        user_event.llm_message = Mock()
        user_event.llm_message.role = "user"
        user_event.llm_message.content = "Hello!"

        agent_event = Mock(spec=ActionEvent)

        user_event2 = Mock(spec=MessageEvent)
        user_event2.llm_message = Mock()
        user_event2.llm_message.role = "user"
        user_event2.llm_message.content = "Follow up"

        runner = Mock()
        runner.conversation = Mock()
        runner.conversation.state = Mock()
        runner.conversation.state.events = [user_event, agent_event, user_event2]
        visualizer = Mock()

        app._replay_conversation_history(runner, visualizer)

        # 2 user messages mounted
        assert app.main_display.mount.call_count == 2
        # 1 agent event to visualizer
        visualizer.on_event.assert_called_once_with(agent_event)

    def test_skips_assistant_message_events(self):
        """Assistant MessageEvents go to visualizer, not as user messages."""
        from openhands.sdk.event import MessageEvent

        app = OpenHandsApp.__new__(OpenHandsApp)
        app.call_from_thread = Mock(side_effect=lambda f, *args: f(*args))
        app.main_display = Mock()

        # Create an assistant message event
        assistant_event = Mock(spec=MessageEvent)
        assistant_event.llm_message = Mock()
        assistant_event.llm_message.role = "assistant"
        assistant_event.llm_message.content = "I can help!"

        runner = Mock()
        runner.conversation = Mock()
        runner.conversation.state = Mock()
        runner.conversation.state.events = [assistant_event]
        visualizer = Mock()

        app._replay_conversation_history(runner, visualizer)

        # Assistant message should go to visualizer
        app.main_display.mount.assert_not_called()
        visualizer.on_event.assert_called_once_with(assistant_event)
