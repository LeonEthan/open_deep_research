#!/usr/bin/env python3
"""Test script to verify Qwen model compatibility."""

import os

from src.open_deep_research.utils import create_configurable_model

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is optional for this test
    pass


def test_qwen_model_creation():
    """Test creating a Qwen model using create_configurable_model."""
    print("Testing Qwen model compatibility...")
    
    # Create configurable model
    model = create_configurable_model()
    
    # Test with Qwen model configuration
    qwen_config = {
        "configurable": {
            "model": "qwen:qwen-max",
            "max_tokens": 1000
        }
    }
    
    try:
        # Configure the model for Qwen
        configured_model = model.with_config(qwen_config)
        print(f"✅ Successfully created Qwen model: {type(configured_model)}")
        
        # Test a simple invocation (if API key is available)
        if os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY"):
            print("Testing model invocation...")
            response = configured_model.invoke("Hello, how are you?")
            print(f"✅ Model response: {response.content[:100]}...")
        else:
            print("⚠️  No API key found, skipping invocation test")
            
    except Exception as e:
        print(f"❌ Error creating/testing Qwen model: {e}")
        return False
    
    return True


def test_openai_model_fallback():
    """Test that OpenAI models still work with create_configurable_model."""
    print("\nTesting OpenAI model fallback...")
    
    # Create configurable model
    model = create_configurable_model()
    
    # Test with OpenAI model configuration
    openai_config = {
        "configurable": {
            "model": "openai:gpt-4o-mini",
            "max_tokens": 100
        }
    }
    
    try:
        # Configure the model for OpenAI
        configured_model = model.with_config(openai_config)
        print(f"✅ Successfully created OpenAI model: {type(configured_model)}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating OpenAI model: {e}")
        return False


if __name__ == "__main__":
    print("🔬 Testing Qwen Model Compatibility")
    print("=" * 40)
    
    # Run tests
    qwen_success = test_qwen_model_creation()
    openai_success = test_openai_model_fallback()
    
    print("\n" + "=" * 40)
    print("📊 Test Results:")
    print(f"  Qwen Model Support: {'✅ PASS' if qwen_success else '❌ FAIL'}")
    print(f"  OpenAI Fallback: {'✅ PASS' if openai_success else '❌ FAIL'}")
    
    if qwen_success and openai_success:
        print("\n🎉 All tests passed! Qwen compatibility is working.")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")