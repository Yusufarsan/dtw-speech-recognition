#!/usr/bin/env python3
"""
DTW-based Speech Recognition System for Vowel Recognition
Uses MFCC features (13 MFCC + Δ + ΔΔ = 39D) and DTW for classification
"""

import os
import glob
import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc, delta
from dtw import dtw


class DTWRecognizer:
    """
    Speech recognition system using DTW algorithm
    """
    
    def __init__(self, data_dir='data'):
        """
        Initialize the DTW recognizer
        
        Args:
            data_dir: Directory containing vowel subdirectories
        """
        self.data_dir = data_dir
        self.vowels = ['a', 'i', 'u', 'e', 'o']
        self.templates = {}
        
    def extract_mfcc_features(self, audio_file):
        """
        Extract 39D MFCC features (13 MFCC + Δ + ΔΔ)
        
        Args:
            audio_file: Path to WAV audio file
            
        Returns:
            39D MFCC features array
        """
        # Read audio file
        sample_rate, signal = wavfile.read(audio_file)
        
        # Extract 13 MFCCs
        mfcc_features = mfcc(signal, sample_rate, numcep=13, nfilt=26, 
                            nfft=512, winfunc=np.hamming)
        
        # Compute delta (first derivative)
        delta_features = delta(mfcc_features, 2)
        
        # Compute delta-delta (second derivative)
        delta_delta_features = delta(delta_features, 2)
        
        # Concatenate all features to get 39D feature vector
        features_39d = np.hstack([mfcc_features, delta_features, delta_delta_features])
        
        return features_39d
    
    def load_templates(self):
        """
        Load template files (template_p1.wav to template_p5.wav) for each vowel
        """
        print("Loading templates...")
        for vowel in self.vowels:
            vowel_dir = os.path.join(self.data_dir, vowel)
            template_files = sorted(glob.glob(os.path.join(vowel_dir, 'template_p*.wav')))
            
            if not template_files:
                print(f"Warning: No template files found for vowel '{vowel}'")
                continue
            
            self.templates[vowel] = []
            for template_file in template_files:
                try:
                    features = self.extract_mfcc_features(template_file)
                    self.templates[vowel].append({
                        'file': os.path.basename(template_file),
                        'features': features
                    })
                    print(f"  Loaded {os.path.basename(template_file)} for vowel '{vowel}'")
                except Exception as e:
                    print(f"  Error loading {template_file}: {e}")
        
        print(f"\nTotal templates loaded: {sum(len(v) for v in self.templates.values())}")
    
    def classify(self, test_features):
        """
        Classify a test sample using DTW
        
        Args:
            test_features: MFCC features of test sample
            
        Returns:
            Tuple of (predicted_vowel, min_distance, distances_dict)
        """
        distances = {}
        
        for vowel, templates_list in self.templates.items():
            vowel_distances = []
            
            for template in templates_list:
                # Compute DTW distance
                alignment = dtw(test_features, template['features'], 
                              dist_method='euclidean')
                vowel_distances.append(alignment.distance)
            
            # Use minimum distance among all templates for this vowel
            if vowel_distances:
                distances[vowel] = min(vowel_distances)
        
        # Find vowel with minimum distance
        if distances:
            predicted_vowel = min(distances, key=distances.get)
            min_distance = distances[predicted_vowel]
            return predicted_vowel, min_distance, distances
        
        return None, float('inf'), {}
    
    def evaluate_closed_set(self):
        """
        Evaluate closed-set recognition (test files from same vowels as templates)
        
        Returns:
            Dictionary containing accuracy and detailed results
        """
        print("\n" + "="*60)
        print("CLOSED-SET EVALUATION")
        print("="*60)
        
        total_tests = 0
        correct_predictions = 0
        results = []
        
        for vowel in self.vowels:
            vowel_dir = os.path.join(self.data_dir, vowel)
            test_files = sorted(glob.glob(os.path.join(vowel_dir, 'uji_p*.wav')))
            
            if not test_files:
                print(f"No test files found for vowel '{vowel}'")
                continue
            
            for test_file in test_files:
                try:
                    test_features = self.extract_mfcc_features(test_file)
                    predicted, distance, all_distances = self.classify(test_features)
                    
                    is_correct = (predicted == vowel)
                    total_tests += 1
                    if is_correct:
                        correct_predictions += 1
                    
                    result = {
                        'file': os.path.basename(test_file),
                        'true_label': vowel,
                        'predicted_label': predicted,
                        'distance': distance,
                        'correct': is_correct
                    }
                    results.append(result)
                    
                    status = "✓" if is_correct else "✗"
                    print(f"{status} {os.path.basename(test_file):20s} | True: {vowel} | Predicted: {predicted} | Distance: {distance:.2f}")
                    
                except Exception as e:
                    print(f"Error processing {test_file}: {e}")
        
        accuracy = (correct_predictions / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\nClosed-set Accuracy: {correct_predictions}/{total_tests} = {accuracy:.2f}%")
        
        return {
            'accuracy': accuracy,
            'correct': correct_predictions,
            'total': total_tests,
            'results': results
        }
    
    def evaluate_open_set(self, unknown_threshold=None):
        """
        Evaluate open-set recognition with threshold-based rejection
        
        Args:
            unknown_threshold: Distance threshold for rejecting unknown samples
                             If None, uses 1.5 * median of template distances
        
        Returns:
            Dictionary containing accuracy and detailed results
        """
        print("\n" + "="*60)
        print("OPEN-SET EVALUATION")
        print("="*60)
        
        # Calculate threshold if not provided
        if unknown_threshold is None:
            all_template_distances = []
            for vowel in self.vowels:
                for template in self.templates.get(vowel, []):
                    # Compare each template with other templates of same vowel
                    for other_template in self.templates.get(vowel, []):
                        if template != other_template:
                            alignment = dtw(template['features'], 
                                          other_template['features'],
                                          dist_method='euclidean')
                            all_template_distances.append(alignment.distance)
            
            if all_template_distances:
                unknown_threshold = 1.5 * np.median(all_template_distances)
            else:
                unknown_threshold = 1000.0  # Default large threshold
        
        print(f"Unknown threshold: {unknown_threshold:.2f}\n")
        
        total_tests = 0
        correct_predictions = 0
        results = []
        
        for vowel in self.vowels:
            vowel_dir = os.path.join(self.data_dir, vowel)
            test_files = sorted(glob.glob(os.path.join(vowel_dir, 'uji_p*.wav')))
            
            if not test_files:
                continue
            
            for test_file in test_files:
                try:
                    test_features = self.extract_mfcc_features(test_file)
                    predicted, distance, all_distances = self.classify(test_features)
                    
                    # Apply threshold for open-set
                    if distance > unknown_threshold:
                        predicted = 'unknown'
                    
                    is_correct = (predicted == vowel)
                    total_tests += 1
                    if is_correct:
                        correct_predictions += 1
                    
                    result = {
                        'file': os.path.basename(test_file),
                        'true_label': vowel,
                        'predicted_label': predicted,
                        'distance': distance,
                        'correct': is_correct,
                        'threshold': unknown_threshold
                    }
                    results.append(result)
                    
                    status = "✓" if is_correct else "✗"
                    print(f"{status} {os.path.basename(test_file):20s} | True: {vowel} | Predicted: {predicted} | Distance: {distance:.2f}")
                    
                except Exception as e:
                    print(f"Error processing {test_file}: {e}")
        
        accuracy = (correct_predictions / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\nOpen-set Accuracy: {correct_predictions}/{total_tests} = {accuracy:.2f}%")
        
        return {
            'accuracy': accuracy,
            'correct': correct_predictions,
            'total': total_tests,
            'threshold': unknown_threshold,
            'results': results
        }
    
    def evaluate(self):
        """
        Run complete evaluation: closed-set, open-set, and compute average
        """
        print("\n" + "="*60)
        print("DTW SPEECH RECOGNITION SYSTEM - EVALUATION")
        print("="*60)
        
        # Load templates
        self.load_templates()
        
        if not self.templates:
            print("\nError: No templates loaded. Please add template audio files.")
            return
        
        # Evaluate closed-set
        closed_set_results = self.evaluate_closed_set()
        
        # Evaluate open-set
        open_set_results = self.evaluate_open_set()
        
        # Compute average accuracy
        avg_accuracy = (closed_set_results['accuracy'] + open_set_results['accuracy']) / 2
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Closed-set Accuracy: {closed_set_results['accuracy']:.2f}%")
        print(f"Open-set Accuracy:   {open_set_results['accuracy']:.2f}%")
        print(f"Average Accuracy:    {avg_accuracy:.2f}%")
        print("="*60)
        
        return {
            'closed_set': closed_set_results,
            'open_set': open_set_results,
            'average_accuracy': avg_accuracy
        }


def main():
    """
    Main function to run the DTW speech recognition system
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='DTW-based Speech Recognition System for Vowel Recognition'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory containing vowel subdirectories (default: data)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Distance threshold for open-set evaluation (default: auto-calculated)'
    )
    
    args = parser.parse_args()
    
    # Initialize recognizer
    recognizer = DTWRecognizer(data_dir=args.data_dir)
    
    # Run evaluation
    recognizer.evaluate()


if __name__ == '__main__':
    main()
