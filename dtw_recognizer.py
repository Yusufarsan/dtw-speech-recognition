import os
import glob
import numpy as np
from scipy.io import wavfile
from scipy.spatial.distance import euclidean
import librosa
import soundfile as sf
from python_speech_features import mfcc, delta
from dtw import dtw
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


class DTWRecognizer:
    
    def __init__(self, data_dir='data', k_segments=1, sampling_rate=16000):
        """
        Initialize the DTW recognizer with generalized segmented templates.
        
        Args:
            data_dir: Directory containing vowel subdirectories
            k_segments: The number of segments to create for each 
                        generalized template.
        """
        self.data_dir = data_dir
        self.vowels = ['a', 'i', 'u', 'e', 'o']
        self.k_segments = k_segments
        self.templates = {}
        self.supported_formats = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        self.sampling_rate = sampling_rate
    
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
        
    def extract_mfcc_features(self, audio_file):
        """
        Extract 39D MFCC features (13 MFCC + Δ + ΔΔ) from audio file
        Supports multiple audio formats: WAV, MP3, FLAC, M4A, OGG
        Trims silence from the start and end of the audio before feature extraction
        
        Args:
            audio_file: Path to audio file (various formats supported)
            
        Returns:
            39D MFCC features array
        """
        try:
            # Auto-resample to 16kHz for consistency
            signal, sample_rate = librosa.load(audio_file, sr=self.sampling_rate, mono=True)
            
            # Trim silence from the start and end of the audio
            # top_db parameter controls the threshold (higher = more aggressive trimming)
            signal, _ = librosa.effects.trim(signal, top_db=3)
            
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
        Load training data and create generalized, segmented templates.
        Implements "Generalization DTW Template"  by aligning
        all samples to a reference, averaging, and then segmenting
        the result into k_segments.
        """
        print("Loading training data and creating generalized segmented templates...")
        print(f"Supported formats: {', '.join(self.supported_formats)}")
        print(f"Number of segments (k): {self.k_segments}")
        
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

            # Generalized Template
            # 1. reference template (median)
            all_features.sort(key=len)
            reference_template = all_features[len(all_features) // 2].copy()
            ref_len = reference_template.shape[0]
            
            # 2. Initialize accumulators for averaging
            sum_template = np.zeros_like(reference_template, dtype=float)
            count_template = np.zeros((ref_len, 1), dtype=float)

            # 3. Align templates
            for features in all_features:
                alignment = dtw(features, reference_template, 
                                dist_method='euclidean', 
                                keep_internals=True)
                
                for i, j in zip(alignment.index1, alignment.index2):
                    sum_template[j] += features[i]
                    count_template[j] += 1
            
            # 4. Calculate the average generalized template
            count_template[count_template == 0] = 1.0
            generalized_template = sum_template / count_template
            
            # 5. Segment generalized_template into k segments 
            segmented_frames_list = np.array_split(generalized_template, self.k_segments)
            
            template_model = {'segments': [], 'segmented_means': []}
            
            # 6. Calculate mean and covariance for each segment
            for segment_frames in segmented_frames_list:
                if segment_frames.shape[0] == 0:
                    continue
                
                # Calculate mean for the segment 
                segment_mean = np.mean(segment_frames, axis=0)
                
                # Calculate covariance for the segment 
                if segment_frames.shape[0] > 1:
                    segment_cov = np.cov(segment_frames, rowvar=False)
                else:
                    # Handle single-frame segment (identity matrix)
                    segment_cov = np.eye(generalized_template.shape[1])
                
                # Add regularization for numerical stability
                segment_cov += 1e-6 * np.eye(segment_cov.shape[0])
                
                template_model['segments'].append({
                    'mean': segment_mean,
                    'covariance': segment_cov
                })
                template_model['segmented_means'].append(segment_mean)
            
            # Store final model
            template_model['segmented_means'] = np.array(template_model['segmented_means'])
            self.templates[vowel] = template_model
            
            print(f"  Created segmented template for vowel '{vowel}' with {len(template_model['segments'])} segments")
        
        print(f"\nTotal vowel templates created: {len(self.templates)}")
    
    
    def classify(self, test_features, distance_metric='euclidean'):
        """
        Classify a test sample using generalized segmented templates.
        
        Args:
            test_features: MFCC features of test sample (n_frames, 39)
            distance_metric: Distance/score function to use. Options:
                           - 'euclidean': DTW against k-segment means 
                           - 'mahalanobis': DTW on cost matrix using 
                                            Mahalanobis distance 
                           - 'gaussian': DTW on cost matrix using 
                                         Gaussian log-likelihood 
            
        Returns:
            Tuple of (predicted_vowel, min_distance, distances_dict)
        """
        distances = {}
        D = test_features.shape[1] # Dimensionality (39)
        n_frames = test_features.shape[0]
        
        for vowel, template_model in self.templates.items():
            k_segments = len(template_model['segments'])
            if k_segments == 0:
                distances[vowel] = float('inf')
                continue
            
            if distance_metric == 'euclidean':
                segmented_means = template_model['segmented_means'] # Shape (k, 39)
                
                try:
                    alignment = dtw(test_features, segmented_means, dist_method='euclidean')
                    distances[vowel] = alignment.distance
                except Exception as e:
                    print(f"Warning: DTW failed for {vowel} (euclidean): {e}")
                    distances[vowel] = float('inf')

            elif distance_metric == 'mahalanobis' or distance_metric == 'gaussian':
                cost_matrix = np.zeros((n_frames, k_segments))
                template_segments = template_model['segments']
                
                for i in range(n_frames):
                    for j in range(k_segments):
                        segment = template_segments[j]
                        mean = segment['mean']
                        cov = segment['covariance']
                        
                        diff = test_features[i] - mean
                        
                        try:
                            # Calculate (x-m)^T * C^-1 * (x-m) 
                            inv_cov = np.linalg.inv(cov)
                            mahalanobis_sq = diff.dot(inv_cov).dot(diff)
                            
                            if not np.isfinite(mahalanobis_sq):
                                mahalanobis_sq = 1e10 # Large cost
                        
                        except np.linalg.LinAlgError:
                            # Fallback to pseudo-inverse if singular
                            inv_cov = np.linalg.pinv(cov)
                            mahalanobis_sq = diff.dot(inv_cov).dot(diff)
                            if not np.isfinite(mahalanobis_sq):
                                mahalanobis_sq = 1e10

                        if distance_metric == 'mahalanobis':
                            cost_matrix[i, j] = mahalanobis_sq
                        
                        # Gaussian (negative log likelihood)
                        else:
                            # Calculate -log(Gaussian) 
                            # Term 1: 0.5 * log((2pi)^D * |C_j|)
                            sign, logdet = np.linalg.slogdet(cov)
                            if sign <= 0:
                                log_const = 1e10
                            else:
                                log_const = 0.5 * (D * np.log(2 * np.pi) + logdet)
                            
                            if not np.isfinite(log_const): 
                                log_const = 1e10

                            # Full cost: 0.5*log_const + 0.5*mahalanobis_sq
                            cost_matrix[i, j] = log_const + 0.5 * mahalanobis_sq
                
                # Run DTW on the pre-computed cost matrix
                try:
                    alignment = dtw(cost_matrix)
                    distances[vowel] = alignment.distance
                except Exception as e:
                    print(f"Warning: DTW failed for {vowel} ({distance_metric}): {e}")
                    distances[vowel] = float('inf')
            
            else:
                raise ValueError(f"Unknown distance metric: {distance_metric}. "
                                 "Choose from: euclidean, mahalanobis, gaussian")
        
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
                             If None, a default large threshold is used.
            distance_metric: Distance metric to use for classification
        
        Returns:
            Dictionary containing accuracy and detailed results
        """
        print("\n" + "="*60)
        print("OPEN-SET EVALUATION")
        print("="*60)
        print(f"Using distance metric: {distance_metric}")
        
        if unknown_threshold is None:
            unknown_threshold = 1e10
        
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
        # self.load_templates()
        
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
        print(f"K-Segments:          {self.k_segments}")
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
        
        # plt.show()
    
    def visualize_dtw_alignment(self, test_audio_file, vowel, save_path=None):
        """
        Visualize DTW alignment path between a test audio file and a 
        segmented template model (Euclidean distance only).
        
        Args:
            test_audio_file: Path to test audio file
            vowel: The vowel template ('a', 'i', 'u', 'e', 'o') to align against
            save_path: Optional path to save the figure
        """
        if vowel not in self.templates:
            print(f"Error: No template found for vowel '{vowel}'")
            return
            
        try:
            test_features = self.extract_mfcc_features(test_audio_file)
        except Exception as e:
            print(f"Error loading test file: {e}")
            return
            
        template_features = self.templates[vowel]['segmented_means']
        
        # Compute DTW alignment
        alignment = dtw(test_features, template_features, dist_method='euclidean')
        
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot 1: Test features (first MFCC)
        ax1.plot(test_features[:, 0], 'b-', linewidth=2, label='Test Audio')
        ax1.set_title(f'Test Audio Features (1st MFCC): {os.path.basename(test_audio_file)}', 
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('MFCC Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Template features (first MFCC)
        ax2.plot(template_features[:, 0], 'r-o', linewidth=2, 
                 label=f'Template "{vowel}" ({self.k_segments} segments)')
        ax2.set_title(f'Segmented Template Features (1st MFCC)', 
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel('Segment Index')
        ax2.set_ylabel('Mean MFCC Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: DTW alignment path
        ax3.plot(alignment.index1, alignment.index2, 'g-', linewidth=2, alpha=0.7)
        ax3.fill_between(alignment.index1, alignment.index2, alpha=0.3, color='green')
        ax3.set_title(f'DTW Alignment Path (Distance: {alignment.distance:.2f})', 
                     fontsize=12, fontweight='bold')
        ax3.set_xlabel('Test Audio Frame')
        ax3.set_ylabel(f'Template Segment Index (0-{self.k_segments-1})')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # plt.show()
    
    def visualize_mfcc_features(self, audio_file, save_path=None):
        """
        Visualize MFCC features from an audio file
        
        Args:
            audio_file: Path to audio file
            save_path: Optional path to save the figure
        """
        # Load audio
        signal, sr = librosa.load(audio_file, sr=self.sampling_rate)
        
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
        
        # plt.show()
    
    def visualize_distance_distribution(self, results, title="Distance Distribution", save_path=None):
        """
        Visualize distance distribution for correct vs incorrect predictions
        
        Args:
            results: Dictionary containing evaluation results with 'results' key
            title: Title for the plot
            save_path: Optional path to save the figure
        """
        # Extract distances and correctness
        correct_distances = [r['distance'] for r in results['results'] if r['correct']]
        incorrect_distances = [r['distance'] for r in results['results'] if not r['correct']]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        if correct_distances:
            sns.histplot(correct_distances, bins=30, kde=True, 
                         label='Correct', color='green')
        if incorrect_distances:
            sns.histplot(incorrect_distances, bins=30, kde=True, 
                         label='Incorrect', color='red')
        
        plt.xlabel('DTW Distance / Cost', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # plt.show()
    
    def visualize_template_features(self, save_path=None):
        """
        Visualize the segmented mean template features (first 13 MFCCs)
        for each vowel.
        
        Args:
            save_path: Optional path to save the figure
        """
        if not self.templates:
            print("No templates loaded. Run load_templates() first.")
            return
        
        # Create figure
        fig, axes = plt.subplots(len(self.vowels), 1, 
                                 figsize=(12, 3 * len(self.vowels)),
                                 sharex=True)
        
        if len(self.vowels) == 1:
            axes = [axes]

        for idx, vowel in enumerate(self.vowels):
            if vowel not in self.templates:
                axes[idx].set_title(f'Vowel "{vowel}" - No Template')
                axes[idx].axis('off')
                continue
            
            # Get the (k, 39) matrix of mean vectors
            mean_vectors = self.templates[vowel]['segmented_means']
            
            # Plot only the first 13 MFCC coefficients for clarity
            mfcc_data = mean_vectors[:, :13]
            
            # Transpose for plotting (segments on x-axis)
            im = axes[idx].imshow(mfcc_data.T, aspect='auto', 
                                  origin='lower', cmap='viridis')
            axes[idx].set_title(f'Vowel "{vowel}" - Segmented Mean Template (13 MFCCs)', 
                               fontsize=11, fontweight='bold')
            axes[idx].set_ylabel('MFCC Coefficient')
            axes[idx].set_yticks(range(13))
            axes[idx].set_yticklabels(range(13))

        axes[-1].set_xlabel('Segment Index')
        axes[-1].set_xticks(range(self.k_segments))
        axes[-1].set_xticklabels(range(self.k_segments))

        fig.colorbar(im, ax=axes, label='Mean Value', orientation='vertical', 
                     fraction=0.05, pad=0.02)
        
        plt.suptitle('Generalized Segmented Template Features', 
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # plt.show()
    
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
        
        # plt.show()
    
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
            if distances:
                sns.histplot(distances, bins=20, kde=True, 
                             label=label, color=colors[idx])
        
        if threshold < 1e9:
            plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                       label=f'Threshold: {threshold:.2f}')
        
        plt.xlabel('DTW Distance / Cost', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Open-Set Recognition: Distance Distribution by Label', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # plt.show()