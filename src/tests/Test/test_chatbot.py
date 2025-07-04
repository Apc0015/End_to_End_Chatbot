import json
from chatbot import Chatbot

def test_chatbot():
    # Initialize chatbot
    bot = Chatbot()
    bot.initialize()
    
    # Test cases with expected intents
    test_cases = [
        ("hello", "greeting"),
        ("hi there", "greeting"),
        ("goodbye", "goodbye"),
        ("bye", "goodbye"),
        ("can you help me", "help"),
        ("what can you do", "help"),
        ("thank you", "thanks"),
        ("thanks a lot", "thanks"),
        ("who are you", "questions"),
        ("what is your name", "questions"),
        ("how's the weather", "weather"),
        ("is it raining", "weather"),
        ("random text that should not match", "fallback")
    ]
    
    print("Testing Chatbot Intent Recognition")
    print("=" * 50)
    
    correct_predictions = 0
    total_tests = len(test_cases)
    
    for user_input, expected_intent in test_cases:
        response, predicted_intent, confidence = bot.get_response(user_input)
        
        # Check if prediction is correct (allowing fallback for unknown inputs)
        is_correct = (predicted_intent == expected_intent or 
                     (expected_intent == "fallback" and predicted_intent == "fallback"))
        
        if is_correct:
            correct_predictions += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"{status} Input: '{user_input}'")
        print(f"   Expected: {expected_intent} | Predicted: {predicted_intent}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Response: {response}")
        print()
    
    accuracy = (correct_predictions / total_tests) * 100
    print(f"Test Results: {correct_predictions}/{total_tests} correct ({accuracy:.1f}%)")
    
    return accuracy

def interactive_test():
    print("\nInteractive Testing Mode")
    print("Type messages to test the chatbot. Type 'quit' to exit.")
    print("-" * 50)
    
    bot = Chatbot()
    bot.initialize()
    
    while True:
        user_input = input("Test Input: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            break
        
        if user_input:
            response, intent, confidence = bot.get_response(user_input)
            print(f"Response: {response}")
            print(f"Intent: {intent} (Confidence: {confidence:.3f})")
            print()

def main():
    print("Chatbot Testing Suite")
    print("=" * 30)
    
    # Run automated tests
    accuracy = test_chatbot()
    
    # Ask for interactive testing
    while True:
        choice = input("\nRun interactive test? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            interactive_test()
            break
        elif choice in ['n', 'no']:
            break
        else:
            print("Please enter 'y' or 'n'")
    
    print("Testing complete!")

if __name__ == "__main__":
    main()