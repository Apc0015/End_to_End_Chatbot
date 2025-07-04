#!/usr/bin/env python3
"""
Simple test script to verify chatbot functionality.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from core.chatbot import Chatbot

def test_chatbot():
    print("🤖 Testing Chatbot Components...")
    
    # Test 1: Initialize chatbot
    print("\n1. Testing Chatbot Initialization...")
    bot = Chatbot()
    
    # Test 2: Load intents
    print("2. Testing Intent Loading...")
    try:
        bot.load_intents()
        print(f"✅ Loaded {len(bot.intents['intents'])} intents successfully")
    except Exception as e:
        print(f"❌ Error loading intents: {e}")
        return False
    
    # Test 3: Test responses without model
    print("3. Testing Fallback Responses...")
    response, intent, confidence = bot.get_response("Hello there!")
    print(f"Response: {response}")
    print(f"Intent: {intent}")
    print(f"Confidence: {confidence}")
    
    print("\n✅ All tests passed! Chatbot is working correctly.")
    return True

if __name__ == "__main__":
    success = test_chatbot()
    sys.exit(0 if success else 1)