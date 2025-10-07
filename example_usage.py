#!/usr/bin/env python3
"""
Example usage of the DTW speech recognition system
Demonstrates programmatic usage of the DTWRecognizer class
"""

from dtw_recognizer import DTWRecognizer
import sys

def example_basic_usage():
    """Basic usage example"""
    print("="*60)
    print("Example: Basic Usage")
    print("="*60)
    
    # Initialize recognizer
    recognizer = DTWRecognizer(data_dir='data')
    
    # Load templates
    recognizer.load_templates()
    
    # Check if templates were loaded
    if not recognizer.templates:
        print("Error: No templates found. Run generate_sample_data.py first.")
        return
    
    # Show loaded templates
    print("\nLoaded templates by vowel:")
    for vowel, templates in recognizer.templates.items():
        print(f"  {vowel}: {len(templates)} templates")
    
    # Show supported formats
    print(f"\nSupported audio formats: {', '.join(recognizer.get_supported_formats())}")

def example_single_classification():
    """Example of classifying a single file"""
    print("\n" + "="*60)
    print("Example: Single File Classification")
    print("="*60)
    
    recognizer = DTWRecognizer(data_dir='data')
    recognizer.load_templates()
    
    if not recognizer.templates:
        print("Error: No templates found.")
        return
    
    # Classify a single test file (try different formats)
    test_files = [
        'data/a/uji_p1.wav',
        'data/a/uji_p1.mp3',  # If available
        'data/a/uji_p1.flac'  # If available
    ]
    
    for test_file in test_files:
        try:
            print(f"\nTrying to classify: {test_file}")
            
            # Check if file exists
            import os
            if not os.path.exists(test_file):
                print(f"  File not found: {test_file}")
                continue
                
            test_features = recognizer.extract_mfcc_features(test_file)
            predicted, distance, all_distances = recognizer.classify(test_features)
            
            print(f"  Predicted vowel: {predicted}")
            print(f"  Distance: {distance:.2f}")
            print("  Distances to all vowels:")
            for vowel, dist in sorted(all_distances.items(), key=lambda x: x[1]):
                print(f"    {vowel}: {dist:.2f}")
            break  # Exit after first successful classification
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print("No test files could be processed")

def example_custom_evaluation():
    """Example of custom evaluation with specific parameters"""
    print("\n" + "="*60)
    print("Example: Custom Evaluation")
    print("="*60)
    
    recognizer = DTWRecognizer(data_dir='data')
    
    # Run evaluation with custom threshold
    results = recognizer.evaluate(unknown_threshold=1500.0)
    
    if results:
        print("\nDetailed Results:")
        print(f"  Closed-set: {results['closed_set']['correct']}/{results['closed_set']['total']}")
        print(f"  Open-set: {results['open_set']['correct']}/{results['open_set']['total']}")
        print(f"  Average: {results['average_accuracy']:.2f}%")

def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("DTW SPEECH RECOGNITION - USAGE EXAMPLES")
    print("="*60)
    print("\nMake sure to run generate_sample_data.py first!\n")
    
    # Run examples
    example_basic_usage()
    example_single_classification()
    example_custom_evaluation()

if __name__ == '__main__':
    main()
