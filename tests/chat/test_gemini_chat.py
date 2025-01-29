"""Test suite for the GeminiChatService in the thinkhub.chat package."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
import tenacity
from PIL import Image

from thinkhub.chat.exceptions import InvalidInputDataError, TokenLimitExceededError
from thinkhub.chat.gemini_chat import GeminiChatService
from thinkhub.transcription.exceptions import MissingAPIKeyError


@pytest.fixture
def mock_genai():
    """Fixture that provides a mock of the genai library."""
    with patch("thinkhub.chat.gemini_chat.genai") as mock:
        # Set up mock model info
        mock_model_info = Mock()
        mock_model_info.input_token_limit = 1000
        mock.get_model.return_value = mock_model_info
        # Configure genai
        mock.configure = Mock()
        yield mock


@pytest.fixture
def mock_image():
    """Fixture that provides a mock for PIL.Image.open."""
    with patch("PIL.Image.open") as mock:
        mock_img = Mock(spec=Image.Image)
        mock.return_value = mock_img
        yield mock


@pytest.fixture
def chat_service(mock_genai):
    """Fixture that provides an instance of GeminiChatService with a mocked environment."""
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test_key"}):
        service = GeminiChatService()
        # Mock the get_model call to avoid API calls
        mock_model_info = Mock()
        mock_model_info.input_token_limit = 1000
        mock_genai.get_model.return_value = mock_model_info
        return service


class TestGeminiChatServiceInit:
    """Tests for the GeminiChatService initialization logic."""

    def test_init_with_api_key(self):
        """Test that the chat service can be initialized with an explicit API key."""
        service = GeminiChatService(api_key="test_key")
        assert service.api_key == "test_key"
        assert service.model_name == "gemini-1.5-flash"

    def test_init_without_api_key(self):
        """Test that initializing without an API key raises MissingAPIKeyError."""
        with patch.dict("os.environ", clear=True):
            with pytest.raises(MissingAPIKeyError):
                GeminiChatService()

    def test_init_with_custom_model(self):
        """Test that the chat service can be initialized with a custom model name."""
        service = GeminiChatService(model="custom-model", api_key="test_key")
        assert service.model_name == "custom-model"


class TestChatSessionManagement:
    """Tests for GeminiChatService session creation and token-based cleanup."""

    @pytest.mark.asyncio
    async def test_ensure_chat_session_creation(self, chat_service, mock_genai):
        """Test that a chat session is created if none exists yet."""
        mock_model = Mock()
        mock_chat = Mock()
        mock_model.start_chat.return_value = mock_chat
        mock_genai.GenerativeModel.return_value = mock_model

        await chat_service._ensure_chat_session("Test system prompt")
        assert chat_service.chat_session is not None
        mock_model.start_chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_chat_session_token_management(self, chat_service, mock_genai):
        """Test that excessive tokens in the history trigger removal of older messages."""
        # Setup mock chat session with history
        mock_model = Mock()
        mock_chat = Mock()
        initial_history = [
            {"role": "model", "parts": "system"},
            {"role": "user", "parts": "message1"},
            {"role": "model", "parts": "response1"},
        ]
        mock_chat.history = initial_history.copy()
        mock_model.start_chat.return_value = mock_chat
        mock_genai.GenerativeModel.return_value = mock_model

        # Mock token counting to trigger history cleanup
        token_counts = []

        async def mock_count_tokens(contents):
            token_counts.append(contents)
            # Return 1500 if the history length is 3 (force a pop),
            # but 900 if there are only 2 messages, so we stop popping.
            if len(contents) == 3:
                return 1500
            return 900

        chat_service._count_tokens = mock_count_tokens
        chat_service.chat_session = mock_chat  # Set the chat session directly

        await chat_service._ensure_chat_session("Test system prompt")

        # Check that the history has been properly managed
        assert len(token_counts) > 0, "Token counting should have been called"
        assert len(mock_chat.history) == 2
        assert (
            mock_chat.history[0]["role"] == "model"
            and mock_chat.history[0]["parts"] == "system"
        )
        assert (
            mock_chat.history[1]["role"] == "user"
            and mock_chat.history[1]["parts"] == "message1"
        ) or (
            mock_chat.history[1]["role"] == "model"
            and mock_chat.history[1]["parts"] == "response1"
        )


class TestInputProcessing:
    """Tests for GeminiChatService input validation and multi-modal preparation."""

    def test_validate_image_input_valid(self, chat_service):
        """Test that valid image-path dictionaries pass the validation function."""
        valid_input = [{"image_path": "path/to/image.jpg"}]
        assert chat_service._validate_image_input(valid_input) is True

    def test_validate_image_input_invalid(self, chat_service):
        """Test various invalid inputs that should fail validation."""
        invalid_inputs = [
            None,
            "not a list",
            [{"wrong_key": "value"}],
            [{"image_path": "path1"}, {"wrong_key": "path2"}],
        ]
        for invalid_input in invalid_inputs:
            assert chat_service._validate_image_input(invalid_input) is False

    @pytest.mark.asyncio
    async def test_prepare_image_input_list(self, chat_service, mock_image):
        """Test successful multi-modal input preparation with a single image."""
        image_data = [{"image_path": "test.jpg"}]
        result = await chat_service._prepare_image_input_list(image_data, "Test prompt")
        assert len(result) == 2  # Prompt + 1 image
        assert result[0] == "Test prompt"
        mock_image.assert_called_once_with("test.jpg")

    @pytest.mark.asyncio
    async def test_prepare_image_input_list_error(self, chat_service, mock_image):
        """Test that OSError in image opening raises InvalidInputDataError."""
        mock_image.side_effect = OSError("Failed to open image")
        image_data = [{"image_path": "test.jpg"}]
        with pytest.raises(InvalidInputDataError):
            await chat_service._prepare_image_input_list(image_data, "Test prompt")


class TestChatResponseStreaming:
    """Tests for streaming responses from GeminiChatService."""

    @pytest.mark.asyncio
    async def test_stream_chat_response_text(self, chat_service, mock_genai):
        """Test that normal text input is streamed properly."""
        mock_response = AsyncMock()
        mock_response.__aiter__.return_value = [Mock(text="Hello"), Mock(text=" World")]
        chat_service._safe_api_call = AsyncMock(return_value=mock_response)
        chat_service._count_tokens = AsyncMock(return_value=100)

        responses = []
        async for chunk in chat_service.stream_chat_response("Test input"):
            responses.append(chunk)

        assert responses == ["Hello", " World"]

    @pytest.mark.asyncio
    async def test_stream_chat_response_token_limit(self, chat_service, mock_genai):
        """Test that exceeding the token limit raises TokenLimitExceededError."""
        mock_model = Mock()
        mock_chat = Mock()
        mock_chat.history = []
        mock_model.start_chat.return_value = mock_chat
        mock_genai.GenerativeModel.return_value = mock_model
        chat_service.chat_session = mock_chat

        # Mock token counting
        chat_service._count_tokens = AsyncMock(return_value=2000)

        # Mock model info
        mock_model_info = Mock()
        mock_model_info.input_token_limit = 1000
        mock_genai.get_model.return_value = mock_model_info

        with pytest.raises(TokenLimitExceededError):
            async for _ in chat_service.stream_chat_response("Test input"):
                pass

    @pytest.mark.asyncio
    async def test_stream_chat_response_empty_input(self, chat_service):
        """Test that empty input yields no response chunks."""
        responses = []
        async for chunk in chat_service.stream_chat_response(""):
            responses.append(chunk)
        assert responses == []

    @pytest.mark.asyncio
    async def test_stream_chat_response_api_error(self, chat_service, mock_genai):
        """Test that an API error yields an error message in the stream."""
        mock_model = Mock()
        mock_chat = Mock()
        mock_chat.history = []
        mock_model.start_chat.return_value = mock_chat
        mock_genai.GenerativeModel.return_value = mock_model
        chat_service.chat_session = mock_chat

        chat_service._count_tokens = AsyncMock(return_value=100)

        # Mock model info
        mock_model_info = Mock()
        mock_model_info.input_token_limit = 1000
        mock_genai.get_model.return_value = mock_model_info

        # Mock the API call to raise an exception
        chat_service._safe_api_call = AsyncMock(side_effect=Exception("API Error"))

        responses = []
        async for chunk in chat_service.stream_chat_response("Test input"):
            responses.append(chunk)

        assert responses == ["[Error: API Error]"]


class TestTokenCounting:
    """Tests for counting tokens in GeminiChatService."""

    @pytest.mark.asyncio
    async def test_count_tokens(self, chat_service, mock_genai):
        """Test that token counting uses the generative model's count_tokens_async."""
        mock_response = Mock()
        mock_response.total_tokens = 50
        mock_model = Mock()
        mock_model.count_tokens_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        token_count = await chat_service._count_tokens("Test content")
        assert token_count == 50
        mock_model.count_tokens_async.assert_called_once_with("Test content")


class TestIntegration:
    """Integration-style tests for GeminiChatService."""

    @pytest.mark.asyncio
    async def test_full_chat_flow(self, chat_service, mock_genai, mock_image):
        """Test a simple end-to-end chat flow with text and image input."""
        chat_service._count_tokens = AsyncMock(return_value=100)
        mock_response = AsyncMock()
        mock_response.__aiter__.return_value = [Mock(text="Response chunk")]
        chat_service._safe_api_call = AsyncMock(return_value=mock_response)

        # Test with text input
        responses = []
        async for chunk in chat_service.stream_chat_response("Hello"):
            responses.append(chunk)
        assert responses == ["Response chunk"]

        # Test with image input
        image_input = [{"image_path": "test.jpg"}]
        responses = []
        async for chunk in chat_service.stream_chat_response(image_input):
            responses.append(chunk)
        assert responses == ["Response chunk"]


class TestAdditionalScenarios:
    """Additional scenarios for expanded test coverage."""

    @pytest.mark.asyncio
    async def test_prepare_image_input_list_multiple_images(
        self, chat_service, mock_image
    ):
        """Test multi-image input preparation."""
        image_data = [
            {"image_path": "test1.jpg"},
            {"image_path": "test2.jpg"},
        ]
        result = await chat_service._prepare_image_input_list(image_data, "Test prompt")
        # Expect: prompt + 2 images
        assert len(result) == 3
        assert result[0] == "Test prompt"
        assert mock_image.call_count == 2
        mock_image.assert_any_call("test1.jpg")
        mock_image.assert_any_call("test2.jpg")

    @pytest.mark.asyncio
    async def test_safe_api_call_retry_success(self, chat_service):
        """Test Tenacity retry logic when the final attempt succeeds."""
        mock_chat_session = Mock()
        mock_chat_session.send_message_async = AsyncMock(
            side_effect=[
                Exception("Transient failure 1"),
                Exception("Transient failure 2"),
                "Final success",
            ]
        )
        chat_service.chat_session = mock_chat_session

        result = await chat_service._safe_api_call("Test prompt")
        assert result == "Final success"
        assert mock_chat_session.send_message_async.call_count == 3

    @pytest.mark.asyncio
    async def test_safe_api_call_retry_fail(self, chat_service):
        """Test Tenacity retry logic when all attempts fail, resulting in RetryError."""
        mock_chat_session = Mock()
        mock_chat_session.send_message_async = AsyncMock(
            side_effect=[
                Exception("Fail 1"),
                Exception("Fail 2"),
                Exception("Fail 3"),
            ]
        )
        chat_service.chat_session = mock_chat_session

        # Expect a RetryError after 3 attempts
        with pytest.raises(tenacity.RetryError) as exc_info:
            await chat_service._safe_api_call("Test prompt")

        # Verify the final *underlying* exception is "Fail 3"
        assert "Fail 3" in str(exc_info.value.last_attempt.exception())

    @pytest.mark.asyncio
    async def test_concurrent_usage(self, chat_service, mock_genai):
        """Test concurrent calls to stream_chat_response."""
        chat_service._count_tokens = AsyncMock(return_value=100)

        async def async_generator_1():
            yield Mock(text="Concurrent response 1 part A")
            yield Mock(text="Concurrent response 1 part B")

        async def async_generator_2():
            yield Mock(text="Concurrent response 2 part A")

        call_count = 0

        async def side_effect(prompt, stream=True):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return async_generator_1()
            return async_generator_2()

        chat_service._safe_api_call = AsyncMock(side_effect=side_effect)

        async def gather_responses(input_data):
            responses = []
            async for chunk in chat_service.stream_chat_response(input_data):
                responses.append(chunk)
            return responses

        results = await asyncio.gather(
            gather_responses("Hello from task1"),
            gather_responses("Hello from task2"),
        )

        assert results[0] == [
            "Concurrent response 1 part A",
            "Concurrent response 1 part B",
        ]
        assert results[1] == ["Concurrent response 2 part A"]

    @pytest.mark.asyncio
    async def test_stream_chat_response_at_token_limit(self, chat_service, mock_genai):
        """Test streaming when token count is exactly at the limit."""
        mock_model = Mock()
        mock_chat = Mock()
        mock_chat.history = []
        mock_model.start_chat.return_value = mock_chat
        mock_genai.GenerativeModel.return_value = mock_model
        chat_service.chat_session = mock_chat

        chat_service._count_tokens = AsyncMock(return_value=1000)

        mock_model_info = Mock()
        mock_model_info.input_token_limit = 1000
        mock_genai.get_model.return_value = mock_model_info

        mock_response = AsyncMock()
        mock_response.__aiter__.return_value = [Mock(text="OK")]
        chat_service._safe_api_call = AsyncMock(return_value=mock_response)

        responses = []
        async for chunk in chat_service.stream_chat_response("Prompt at limit"):
            responses.append(chunk)
        assert responses == ["OK"]

    @pytest.mark.asyncio
    async def test_stream_chat_response_just_below_limit(
        self, chat_service, mock_genai
    ):
        """Test streaming when token count is just below the limit."""
        mock_model = Mock()
        mock_chat = Mock()
        mock_chat.history = []
        mock_model.start_chat.return_value = mock_chat
        mock_genai.GenerativeModel.return_value = mock_model
        chat_service.chat_session = mock_chat

        chat_service._count_tokens = AsyncMock(return_value=999)

        mock_model_info = Mock()
        mock_model_info.input_token_limit = 1000
        mock_genai.get_model.return_value = mock_model_info

        mock_response = AsyncMock()
        mock_response.__aiter__.return_value = [Mock(text="Below limit")]
        chat_service._safe_api_call = AsyncMock(return_value=mock_response)

        responses = []
        async for chunk in chat_service.stream_chat_response("Prompt just below limit"):
            responses.append(chunk)
        assert responses == ["Below limit"]
