#!/usr/bin/env python3
"""Test script to verify legacy compatibility."""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def test_legacy_import():
    """Test that legacy files can still import our new function."""
    print("Testing legacy import compatibility...")
    
    try:
        # Test importing from legacy files
        from src.legacy.multi_agent import graph as legacy_graph
        print("✅ Legacy multi_agent import successful")
        
        # Test that the graph can be created
        if legacy_graph:
            print("✅ Legacy graph creation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Legacy import failed: {e}")
        return False


def test_main_module_import():
    """Test that main module still works."""
    print("\nTesting main module import...")
    
    try:
        from src.open_deep_research.deep_researcher import graph as main_graph
        print("✅ Main module import successful")
        
        if main_graph:
            print("✅ Main graph creation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Main module import failed: {e}")
        return False


if __name__ == "__main__":
    print("🔬 Testing Legacy Compatibility")
    print("=" * 40)
    
    # Run tests
    legacy_success = test_legacy_import()
    main_success = test_main_module_import()
    
    print("\n" + "=" * 40)
    print("📊 Test Results:")
    print(f"  Legacy Compatibility: {'✅ PASS' if legacy_success else '❌ FAIL'}")
    print(f"  Main Module: {'✅ PASS' if main_success else '❌ FAIL'}")
    
    if legacy_success and main_success:
        print("\n🎉 All compatibility tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")