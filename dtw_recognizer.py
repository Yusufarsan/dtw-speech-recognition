import os
import glob
import numpy as np
from scipy.io import wavfile
from scipy.stats import multivariate_normal
import librosa
import soundfile as sf
from python_speech_features import mfcc, delta
from dtw import dtw
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


class DTWRecognizer:
    
    def __init__(self, data_dir='data'):
        """
        Initialize the DTW recognizer
        
        Args:
            data_dir: Directory containing vowel subdirectories
        """
        self.data_dir = data_dir
        self.vowels = ['a', 'i', 'u', 'e', 'o']
        self.templates = {}
        self.supported_formats = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    
    def get_supported_formats(self):
        """
        Get list of supported audio formats
        
        Returns:
            List of supported file extensions
        """
        return self.supported_formats
    
    def find_audio_files(self, directory, pattern_prefix):
        """
        Find audio files with given pattern prefix in multiple formats
        
        Args:
            directory: Directory to search in
            pattern_prefix: Pattern prefix (e.g., 'template_p', 'uji_p')
            
        Returns:
            List of found audio files
        """
        audio_files = []
        for ext in self.supported_formats:
            pattern = os.path.join(directory, f'{pattern_prefix}*{ext}')
            files = glob.glob(pattern)
            audio_files.extend(files)
        return sorted(audio_files)
    
    def validate_audio_file(self, audio_file):
        """
        Validate if audio file can be processed
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            True if file is valid, False otherwise
        """
        try:
            _, sr = librosa.load(audio_file, sr=None, duration=0.1)
            return True
        except Exception:
            return False
        
    def calculate_mean_template(self, features_list):
        """
        Calculate mean template from a list of feature matrices
        
        Args:
            features_list: List of feature matrices, each of shape (n_frames, n_features)
        
        Returns:
            Mean template (average feature vector across all frames and samples)
        """
        if not features_list:
            return None
        
        # Concatenate all feature vectors from all samples
        all_features = []
        for features in features_list:
            # features shape: (n_frames, 39)
            all_features.append(features)
        
        # Stack all features into one large matrix
        all_features_concat = np.vstack(all_features)  # (total_frames, 39)
        
        # Compute mean across all frames from all samples
        mean_template = np.mean(all_features_concat, axis=0)  # (39,)
        
        return mean_template
    
    def calculate_covariance_matrix(self, features_list, mean_template=None):
        """
        Calculate covariance matrix from a list of feature matrices
        
        Args:
            features_list: List of feature matrices, each of shape (n_frames, n_features)
            mean_template: Pre-calculated mean template (optional)
        
        Returns:
            Covariance matrix of shape (n_features, n_features)
        """
        # Concatenate all features
        all_features = np.vstack(features_list)  # (total_frames, 39)
        
        # Calculate mean if not provided
        if mean_template is None:
            mean_template = np.mean(all_features, axis=0)
        
        # Calculate covariance
        cov_matrix = np.cov(all_features, rowvar=False)  # (39, 39)
        
        # Add regularization for numerical stability
        regularization = 1e-6
        cov_matrix += regularization * np.eye(cov_matrix.shape[0])
        
        return cov_matrix
    
    def extract_mfcc_features(self, audio_file):
        """
        Extract 39D MFCC features (13 MFCC + Δ + ΔΔ) from audio file
        Supports multiple audio formats: WAV, MP3, FLAC, M4A, OGG
        
        Args:
            audio_file: Path to audio file (various formats supported)
            
        Returns:
            39D MFCC features array
        """
        try:
            # Auto-resample to 16kHz for consistency
            signal, sample_rate = librosa.load(audio_file, sr=16000, mono=True)
            
            # Convert to int16 for compatibility with python-speech-features
            signal = (signal * 32767).astype(np.int16)
            
        except Exception as e:
            # Fallback to scipy.io.wavfile for .wav files
            try:
                if audio_file.lower().endswith('.wav'):
                    sample_rate, signal = wavfile.read(audio_file)
                    # Convert to mono if stereo
                    if len(signal.shape) > 1:
                        signal = np.mean(signal, axis=1)
                    if signal.dtype != np.int16:
                        signal = (signal / np.max(np.abs(signal)) * 32767).astype(np.int16)
                else:
                    raise Exception(f"Unsupported format: {audio_file}")
            except Exception as e2:
                raise Exception(f"Failed to load audio file {audio_file}: {e2}")
        
        # Extract 13 MFCCs
        mfcc_features = mfcc(signal, sample_rate, numcep=13, nfilt=26, 
                            nfft=512, winfunc=np.hamming)
        
        # first derivative
        delta_features = delta(mfcc_features, 2)
        
        # second derivative
        delta_delta_features = delta(delta_features, 2)
        
        # Concatenate all features to get 39D feature vector
        features_39d = np.hstack([mfcc_features, delta_features, delta_delta_features])
        
        return features_39d
    
    def load_templates(self):
        """
        Load training data and create generalized templates using mean and covariance
        Training data is organized in train/ directory with subdirectories for each vowel
        """
        print("Loading training data and creating generalized templates...")
        print(f"Supported formats: {', '.join(self.supported_formats)}")
        
        train_dir = os.path.join(self.data_dir, 'train')
        
        if not os.path.exists(train_dir):
            print(f"Warning: Training directory not found: {train_dir}")
            return
        
        for vowel in self.vowels:
            vowel_train_dir = os.path.join(train_dir, vowel)
            
            if not os.path.exists(vowel_train_dir):
                print(f"Warning: Training directory not found for vowel '{vowel}': {vowel_train_dir}")
                continue
            
            # Find all audio files in the vowel training directory
            audio_files = []
            for ext in self.supported_formats:
                pattern = os.path.join(vowel_train_dir, f'*{ext}')
                files = glob.glob(pattern)
                audio_files.extend(files)
            
            if not audio_files:
                print(f"Warning: No training files found for vowel '{vowel}' in {vowel_train_dir}")
                continue
            
            # Extract features from all training files
            all_features = []
            print(f"  Loading training files for vowel '{vowel}':")
            for audio_file in sorted(audio_files):
                try:
                    # Validate file before processing
                    if not self.validate_audio_file(audio_file):
                        print(f"    Warning: Invalid audio file {os.path.basename(audio_file)}")
                        continue
                    
                    features = self.extract_mfcc_features(audio_file)
                    all_features.append(features)
                    print(f"    Loaded {os.path.basename(audio_file)}")
                except Exception as e:
                    print(f"    Error loading {audio_file}: {e}")
            
            if not all_features:
                print(f"  Warning: No valid features extracted for vowel '{vowel}'")
                continue
            
            # Calculate generalized template from all training features
            mean_template = self.calculate_mean_template(all_features)
            covariance_matrix = self.calculate_covariance_matrix(all_features)

            # For now, store all features for future processing
            self.templates[vowel] = {
                'raw_features': all_features,
                'num_samples': len(all_features),
                'mean': mean_template,
                'covariance': covariance_matrix
            }
            print(f"  Created template for vowel '{vowel}' from {len(all_features)} samples")
        
        print(f"\nTotal vowel templates created: {len(self.templates)}")
        total_samples = sum(template['num_samples'] for template in self.templates.values())
        print(f"Total training samples processed: {total_samples}")
    
    def calculate_mahalanobis_distance(self, test_features, mean_template, covariance_matrix):
        """
        Calculate Mahalanobis distance between test features and template
        
        Args:
            test_features: Test feature matrix of shape (n_frames, n_features)
            mean_template: Mean template
            covariance_matrix: Covariance matrix of shape (n_features, n_features)
        
        Returns:
            Mahalanobis distance (scalar)
        """
        # Flatten test features by taking mean across all frames
        if len(test_features.shape) > 1:
            test_vector = np.mean(test_features, axis=0)  # (39,)
        else:
            test_vector = test_features
        
        # Calculate difference
        diff = test_vector - mean_template
        
        # Formula: sqrt((x - mu)^T * Sigma^-1 * (x - mu))
        try:
            inv_cov = np.linalg.inv(covariance_matrix)
            mahalanobis_dist = np.sqrt(np.dot(np.dot(diff, inv_cov), diff))
            
            # Handle potential NaN or inf values
            if not np.isfinite(mahalanobis_dist):
                mahalanobis_dist = float('inf')
                
        except np.linalg.LinAlgError:
            # Fallback if matrix is singular
            mahalanobis_dist = float('inf')
        
        return mahalanobis_dist
    
    def calculate_gaussian_likelihood(self, test_features, mean_template, covariance_matrix):
        """
        Calculate Gaussian likelihood for test features given template parameters
        
        Args:
            test_features: Test feature matrix of shape (n_frames, n_features)
            mean_template: Mean template
            covariance_matrix: Covariance matrix of shape (n_features, n_features)
        
        Returns:
            Log-likelihood (scalar, higher is better)
        """
        # Flatten test features by taking mean across all frames
        if len(test_features.shape) > 1:
            test_vector = np.mean(test_features, axis=0)  # (39,)
        else:
            test_vector = test_features
        
        # Calculate log-likelihood using scipy's multivariate_normal
        try:
            log_likelihood = multivariate_normal.logpdf(
                test_vector, 
                mean=mean_template, 
                cov=covariance_matrix,
                allow_singular=True
            )
            
            # Handle potential NaN or inf values
            if not np.isfinite(log_likelihood):
                log_likelihood = -1e10
                
        except Exception as e:
            # Fallback to large negative value if computation fails
            print(f"Warning: Gaussian likelihood calculation failed: {e}")
            log_likelihood = -1e10
        
        return log_likelihood
    
    def classify(self, test_features, distance_metric='euclidean'):
        """
        Classify a test sample using specified distance metric
        
        Args:
            test_features: MFCC features of test sample
            distance_metric: Distance/score function to use. Options:
                           - 'euclidean': Standard DTW with Euclidean distance
                           - 'mahalanobis': Mahalanobis distance
                           - 'gaussian': Gaussian log-likelihood (converted to distance)
            
        Returns:
            Tuple of (predicted_vowel, min_distance, distances_dict)
        """
        distances = {}
        
        for vowel, template_data in self.templates.items():
            if distance_metric == 'euclidean':
                # Standard DTW with Euclidean distance
                vowel_distances = []
                for features in template_data['raw_features']:
                    alignment = dtw(test_features, features, dist_method='euclidean')
                    vowel_distances.append(alignment.distance)
                distances[vowel] = min(vowel_distances) if vowel_distances else float('inf')
            
            elif distance_metric == 'mahalanobis':
                distance = self.calculate_mahalanobis_distance(test_features, template_data['mean'], template_data['covariance'])
                distances[vowel] = distance
            
            elif distance_metric == 'gaussian':
                log_likelihood = self.calculate_gaussian_likelihood(test_features, template_data['mean'], template_data['covariance'])
                distances[vowel] = -log_likelihood  # Convert to distance (lower is better)
            
            else:
                raise ValueError(f"Unknown distance metric: {distance_metric}. Choose from: euclidean, mahalanobis, gaussian")
        
        # Find vowel with minimum distance
        if distances:
            predicted_vowel = min(distances, key=distances.get)
            min_distance = distances[predicted_vowel]
            return predicted_vowel, min_distance, distances
        
        return None, float('inf'), {}
    
    def evaluate_closed_set(self, distance_metric='euclidean'):
        """
        Evaluate closed-set recognition (test files from closed_test directory)
        Test files are organized in closed_test/ with subdirectories for each vowel
        
        Args:
            distance_metric: Distance metric to use for classification
        
        Returns:
            Dictionary containing accuracy and detailed results
        """
        print("\n" + "="*60)
        print("CLOSED-SET EVALUATION")
        print("="*60)
        print(f"Using distance metric: {distance_metric}")
        
        closed_test_dir = os.path.join(self.data_dir, 'closed_test')
        
        if not os.path.exists(closed_test_dir):
            print(f"Warning: Closed test directory not found: {closed_test_dir}")
            return {
                'accuracy': 0.0,
                'correct': 0,
                'total': 0,
                'results': []
            }
        
        total_tests = 0
        correct_predictions = 0
        results = []
        
        for vowel in self.vowels:
            vowel_test_dir = os.path.join(closed_test_dir, vowel)
            
            if not os.path.exists(vowel_test_dir):
                print(f"No test directory found for vowel '{vowel}': {vowel_test_dir}")
                continue
            
            # Find all audio files in the vowel test directory
            test_files = []
            for ext in self.supported_formats:
                pattern = os.path.join(vowel_test_dir, f'*{ext}')
                files = glob.glob(pattern)
                test_files.extend(files)
            
            if not test_files:
                print(f"No test files found for vowel '{vowel}' in {vowel_test_dir}")
                continue
            
            for test_file in sorted(test_files):
                try:
                    test_features = self.extract_mfcc_features(test_file)
                    predicted, distance, all_distances = self.classify(test_features, distance_metric=distance_metric)
                    
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
                    
                    status = "[CORRECT]" if is_correct else "[WRONG]"
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
    
    def evaluate_open_set(self, unknown_threshold=None, distance_metric='euclidean'):
        """
        Evaluate open-set recognition with threshold-based rejection
        Test files are organized in open_test/ directory
        
        Args:
            unknown_threshold: Distance threshold for rejecting unknown samples
                             If None, uses 1.5 * median of intra-class distances
            distance_metric: Distance metric to use for classification
        
        Returns:
            Dictionary containing accuracy and detailed results
        """
        print("\n" + "="*60)
        print("OPEN-SET EVALUATION")
        print("="*60)
        print(f"Using distance metric: {distance_metric}")
        
        # Calculate threshold if not provided
        if unknown_threshold is None:
            all_template_distances = []
            for vowel, template_data in self.templates.items():
                raw_features = template_data['raw_features']
                # Compare each template with other templates of same vowel
                for i, feat1 in enumerate(raw_features):
                    for j, feat2 in enumerate(raw_features):
                        if i < j:  # Avoid duplicate comparisons
                            if distance_metric == 'euclidean':
                                alignment = dtw(feat1, feat2, dist_method='euclidean')
                                all_template_distances.append(alignment.distance)
                            elif distance_metric == 'mahalanobis':
                                distance = self.calculate_mahalanobis_distance(feat1, template_data['mean'], template_data['covariance'])
                                all_template_distances.append(distance)
                            elif distance_metric == 'gaussian':
                                log_likelihood = self.calculate_gaussian_likelihood(feat1, template_data['mean'], template_data['covariance'])
                                all_template_distances.append(-log_likelihood)
            
            if all_template_distances:
                unknown_threshold = 1.5 * np.median(all_template_distances)
            else:
                unknown_threshold = 1000.0  # Default large threshold
        
        print(f"Unknown threshold: {unknown_threshold:.2f}\n")
        
        open_test_dir = os.path.join(self.data_dir, 'open_test')
        
        if not os.path.exists(open_test_dir):
            print(f"Warning: Open test directory not found: {open_test_dir}")
            return {
                'accuracy': 0.0,
                'correct': 0,
                'total': 0,
                'threshold': unknown_threshold,
                'results': []
            }
        
        total_tests = 0
        correct_predictions = 0
        results = []
        
        # Scan open_test directory for subdirectories (each represents a vowel or unknown)
        for item in sorted(os.listdir(open_test_dir)):
            item_path = os.path.join(open_test_dir, item)
            
            if not os.path.isdir(item_path):
                continue
            
            true_label = item  # Directory name is the true label
            
            # Find all audio files in this directory
            test_files = []
            for ext in self.supported_formats:
                pattern = os.path.join(item_path, f'*{ext}')
                files = glob.glob(pattern)
                test_files.extend(files)
            
            if not test_files:
                continue
            
            for test_file in sorted(test_files):
                try:
                    test_features = self.extract_mfcc_features(test_file)
                    predicted, distance, all_distances = self.classify(test_features, distance_metric=distance_metric)
                    
                    # Apply threshold for open-set
                    if distance > unknown_threshold:
                        predicted = 'unknown'
                    
                    is_correct = (predicted == true_label)
                    total_tests += 1
                    if is_correct:
                        correct_predictions += 1
                    
                    result = {
                        'file': os.path.basename(test_file),
                        'true_label': true_label,
                        'predicted_label': predicted,
                        'distance': distance,
                        'correct': is_correct,
                        'threshold': unknown_threshold
                    }
                    results.append(result)
                    
                    status = "[CORRECT]" if is_correct else "[WRONG]"
                    print(f"{status} {os.path.basename(test_file):20s} | True: {true_label} | Predicted: {predicted} | Distance: {distance:.2f}")
                    
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
    
    def evaluate(self, unknown_threshold=None, distance_metric='euclidean'):
        """
        Run complete evaluation: closed-set, open-set, and compute average
        
        Args:
            unknown_threshold: Optional threshold for open-set evaluation
            distance_metric: Distance metric to use ('euclidean', 'mahalanobis', 'gaussian')
        """
        print("\n" + "="*60)
        print("DTW SPEECH RECOGNITION SYSTEM - EVALUATION")
        print("="*60)
        
        # Load templates
        self.load_templates()
        
        if not self.templates:
            print("\nError: No templates loaded. Please add training audio files in data/train/ directory.")
            return
        
        # Evaluate closed-set
        closed_set_results = self.evaluate_closed_set(distance_metric=distance_metric)
        
        # Evaluate open-set
        open_set_results = self.evaluate_open_set(unknown_threshold=unknown_threshold, distance_metric=distance_metric)
        
        # Compute average accuracy
        avg_accuracy = (closed_set_results['accuracy'] + open_set_results['accuracy']) / 2
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Distance Metric:     {distance_metric}")
        print(f"Closed-set Accuracy: {closed_set_results['accuracy']:.2f}%")
        print(f"Open-set Accuracy:   {open_set_results['accuracy']:.2f}%")
        print(f"Average Accuracy:    {avg_accuracy:.2f}%")
        print("="*60)
        
        return {
            'closed_set': closed_set_results,
            'open_set': open_set_results,
            'average_accuracy': avg_accuracy
        }
    
    def visualize_confusion_matrix(self, results, title="Confusion Matrix", save_path=None):
        """
        Visualize confusion matrix from evaluation results
        
        Args:
            results: Dictionary containing evaluation results with 'results' key
            title: Title for the plot
            save_path: Optional path to save the figure
        """
        # Extract true and predicted labels
        true_labels = [r['true_label'] for r in results['results']]
        pred_labels = [r['predicted_label'] for r in results['results']]
        
        # Get unique labels
        labels = sorted(set(true_labels + pred_labels))
        
        # Compute confusion matrix
        cm = confusion_matrix(true_labels, pred_labels, labels=labels)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={'label': 'Count'})
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def visualize_dtw_alignment(self, test_features, template_features, save_path=None):
        """
        Visualize DTW alignment path between test and template features
        
        Args:
            test_features: Test feature matrix
            template_features: Template feature matrix
            save_path: Optional path to save the figure
        """
        # Compute DTW alignment
        alignment = dtw(test_features, template_features, dist_method='euclidean')
        
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot 1: Test features (first MFCC)
        ax1.plot(test_features[:, 0], 'b-', linewidth=2, label='Test Audio')
        ax1.set_title('Test Audio Features (1st MFCC)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('MFCC Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Template features (first MFCC)
        ax2.plot(template_features[:, 0], 'r-', linewidth=2, label='Template Audio')
        ax2.set_title('Template Audio Features (1st MFCC)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('MFCC Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: DTW alignment path
        ax3.plot(alignment.index1, alignment.index2, 'g-', linewidth=2, alpha=0.7)
        ax3.fill_between(alignment.index1, alignment.index2, alpha=0.3, color='green')
        ax3.set_title(f'DTW Alignment Path (Distance: {alignment.distance:.2f})', 
                     fontsize=12, fontweight='bold')
        ax3.set_xlabel('Test Audio Frame')
        ax3.set_ylabel('Template Audio Frame')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def visualize_mfcc_features(self, audio_file, save_path=None):
        """
        Visualize MFCC features from an audio file
        
        Args:
            audio_file: Path to audio file
            save_path: Optional path to save the figure
        """
        # Load audio
        signal, sr = librosa.load(audio_file, sr=16000)
        
        # Extract MFCC
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Waveform
        time = np.arange(len(signal)) / sr
        ax1.plot(time, signal, 'b-', linewidth=0.5)
        ax1.set_title(f'Audio Waveform: {os.path.basename(audio_file)}', 
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: MFCC Spectrogram
        im = ax2.imshow(mfccs, aspect='auto', origin='lower', cmap='viridis')
        ax2.set_title('MFCC Features (13 coefficients)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('MFCC Coefficient')
        plt.colorbar(im, ax=ax2, label='Magnitude')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def visualize_distance_distribution(self, results, save_path=None):
        """
        Visualize distance distribution for correct vs incorrect predictions
        
        Args:
            results: Dictionary containing evaluation results with 'results' key
            save_path: Optional path to save the figure
        """
        # Extract distances and correctness
        correct_distances = [r['distance'] for r in results['results'] if r['correct']]
        incorrect_distances = [r['distance'] for r in results['results'] if not r['correct']]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        plt.hist(correct_distances, bins=30, alpha=0.7, label='Correct', color='green', edgecolor='black')
        plt.hist(incorrect_distances, bins=30, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
        
        plt.xlabel('DTW Distance', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distance Distribution: Correct vs Incorrect Predictions', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def visualize_template_features(self, save_path=None):
        """
        Visualize mean template features for each vowel
        
        Args:
            save_path: Optional path to save the figure
        """
        if not self.templates:
            print("No templates loaded. Run load_templates() first.")
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, vowel in enumerate(self.vowels):
            if vowel not in self.templates:
                continue
            
            mean_features = self.templates[vowel]['mean']
            
            # Plot MFCC coefficients
            axes[idx].bar(range(len(mean_features)), mean_features, color='steelblue', edgecolor='black')
            axes[idx].set_title(f'Vowel "{vowel}" - Mean Template', fontsize=11, fontweight='bold')
            axes[idx].set_xlabel('Feature Index')
            axes[idx].set_ylabel('Mean Value')
            axes[idx].grid(True, alpha=0.3)
        
        # Remove extra subplot
        axes[-1].axis('off')
        
        plt.suptitle('Mean Template Features for Each Vowel', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def visualize_accuracy_comparison(self, closed_results, open_results, save_path=None):
        """
        Visualize accuracy comparison between closed-set and open-set
        
        Args:
            closed_results: Closed-set evaluation results
            open_results: Open-set evaluation results
            save_path: Optional path to save the figure
        """
        categories = ['Closed-Set', 'Open-Set', 'Average']
        accuracies = [
            closed_results['accuracy'],
            open_results['accuracy'],
            (closed_results['accuracy'] + open_results['accuracy']) / 2
        ]
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(categories, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'], 
                       edgecolor='black', linewidth=2)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('DTW Speech Recognition Performance', fontsize=14, fontweight='bold')
        plt.ylim(0, 110)
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def visualize_open_set_threshold_analysis(self, results, save_path=None):
        """
        Visualize threshold analysis for open-set recognition
        
        Args:
            results: Open-set evaluation results
            save_path: Optional path to save the figure
        """
        threshold = results['threshold']
        
        # Extract distances by true label
        distances_by_label = {}
        for r in results['results']:
            true_label = r['true_label']
            distance = r['distance']
            if true_label not in distances_by_label:
                distances_by_label[true_label] = []
            distances_by_label[true_label].append(distance)
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(distances_by_label)))
        
        for idx, (label, distances) in enumerate(distances_by_label.items()):
            plt.hist(distances, bins=20, alpha=0.6, label=label, 
                    color=colors[idx], edgecolor='black')
        
        # Draw threshold line
        plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold: {threshold:.2f}')
        
        plt.xlabel('DTW Distance', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Open-Set Recognition: Distance Distribution by Label', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
