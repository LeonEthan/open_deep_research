"""Tests for DeepSeek and Qwen model configuration."""

import pytest
from open_deep_research.configuration import Configuration
from open_deep_research.utils import (
    get_api_key_for_model,
    get_model_token_limit,
    is_model_supports_structured_output,
)


def test_deepseek_default_config():
    """Test that DeepSeek models are set as defaults."""
    config = Configuration()
    
    # Check that all default models use DeepSeek
    assert config.summarization_model.startswith("deepseek:")
    assert config.research_model.startswith("deepseek:")
    assert config.compression_model.startswith("deepseek:")
    assert config.final_report_model.startswith("deepseek:")
    
    # Check specific model assignments
    assert config.summarization_model == "deepseek:deepseek-chat"
    assert config.research_model == "deepseek:deepseek-chat"
    assert config.compression_model == "deepseek:deepseek-chat"
    assert config.final_report_model == "deepseek:deepseek-reasoner"


def test_qwen_compatibility():
    """Test that Qwen models can be configured."""
    config = Configuration(
        research_model="qwen:qwen-plus",
        final_report_model="qwen:qwen-max"
    )
    
    # Verify Qwen configuration
    assert config.research_model == "qwen:qwen-plus"
    assert config.final_report_model == "qwen:qwen-max"


def test_api_key_retrieval():
    """Test API key retrieval for DeepSeek and Qwen models."""
    import os

    # Mock config for testing
    mock_config = {
        "configurable": {
            "apiKeys": {
                "DEEPSEEK_API_KEY": "test_deepseek_key",
                "QWEN_API_KEY": "test_qwen_key",
                "OPENAI_API_KEY": "test_openai_key"
            }
        }
    }

    # Set environment variable to get keys from config
    original_env = os.environ.get("GET_API_KEYS_FROM_CONFIG")
    os.environ["GET_API_KEYS_FROM_CONFIG"] = "true"

    try:
        # Test DeepSeek API key retrieval
        deepseek_key = get_api_key_for_model("deepseek:deepseek-chat", mock_config)
        assert deepseek_key == "test_deepseek_key"

        # Test Qwen API key retrieval
        qwen_key = get_api_key_for_model("qwen:qwen-plus", mock_config)
        assert qwen_key == "test_qwen_key"

        # Test OpenAI API key retrieval (for backward compatibility)
        openai_key = get_api_key_for_model("openai:gpt-4", mock_config)
        assert openai_key == "test_openai_key"

    finally:
        # Restore original environment variable
        if original_env is not None:
            os.environ["GET_API_KEYS_FROM_CONFIG"] = original_env
        else:
            os.environ.pop("GET_API_KEYS_FROM_CONFIG", None)


def test_model_token_limits():
    """Test token limit configuration for DeepSeek and Qwen models."""
    # Test DeepSeek models
    assert get_model_token_limit("deepseek:deepseek-chat") == 32768
    assert get_model_token_limit("deepseek:deepseek-coder") == 32768
    assert get_model_token_limit("deepseek:deepseek-reasoner") == 65536
    assert get_model_token_limit("deepseek:deepseek-r1") == 65536
    
    # Test Qwen models
    assert get_model_token_limit("qwen:qwen-turbo") == 32768
    assert get_model_token_limit("qwen:qwen-plus") == 32768
    assert get_model_token_limit("qwen:qwen-max") == 32768
    assert get_model_token_limit("qwen:qwen-long") == 1000000
    
    # Test unknown model
    assert get_model_token_limit("unknown:model") is None


def test_structured_output_support():
    """Test structured output support detection."""
    # DeepSeek R1 series should return False (weaker function calling)
    assert is_model_supports_structured_output("deepseek:deepseek-r1") is False
    assert is_model_supports_structured_output("deepseek:deepseek-r1-distill-llama-70b") is False
    
    # Other DeepSeek models should return True
    assert is_model_supports_structured_output("deepseek:deepseek-chat") is True
    assert is_model_supports_structured_output("deepseek:deepseek-coder") is True
    assert is_model_supports_structured_output("deepseek:deepseek-reasoner") is True
    
    # Qwen models should return True
    assert is_model_supports_structured_output("qwen:qwen-plus") is True
    assert is_model_supports_structured_output("qwen:qwen-max") is True
    
    # Other models should default to True
    assert is_model_supports_structured_output("openai:gpt-4") is True
    assert is_model_supports_structured_output("anthropic:claude-3-5-sonnet") is True


def test_configuration_from_env():
    """Test configuration creation from environment variables."""
    import os
    
    # Set environment variables
    os.environ["SUMMARIZATION_MODEL"] = "qwen:qwen-turbo"
    os.environ["RESEARCH_MODEL"] = "deepseek:deepseek-coder"
    
    try:
        config = Configuration.from_runnable_config({})
        
        # Check that environment variables override defaults
        assert config.summarization_model == "qwen:qwen-turbo"
        assert config.research_model == "deepseek:deepseek-coder"
        
    finally:
        # Clean up environment variables
        os.environ.pop("SUMMARIZATION_MODEL", None)
        os.environ.pop("RESEARCH_MODEL", None)


if __name__ == "__main__":
    pytest.main([__file__])
