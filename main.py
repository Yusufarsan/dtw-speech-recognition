from dtw_recognizer import DTWRecognizer
import os

def run_basic_usage():
    recognizer = DTWRecognizer(data_dir='data')
    recognizer.load_templates()

    if not recognizer.templates:
        print("Error: No templates found. Run generate_sample_data.py first.")
        return
    
    # Show loaded templates
    print("\nLoaded templates by vowel:")
    for vowel, templates in recognizer.templates.items():
        print(f"  {vowel}: {len(templates)} templates")
        
def run_single_classification():
    print("\n" + "="*60)
    print("Example: Single File Classification")
    print("="*60)
    
    recognizer = DTWRecognizer(data_dir='data')
    recognizer.load_templates()
    
    if not recognizer.templates:
        print("Error: No templates found.")
        return
    
    # Classify a single test file from closed_test directory
    import os
    test_files = []
    
    # Try to find test files in closed_test directory
    closed_test_dir = 'data/closed_test/a'
    if os.path.exists(closed_test_dir):
        for ext in recognizer.get_supported_formats():
            pattern = os.path.join(closed_test_dir, f'*{ext}')
            import glob
            files = glob.glob(pattern)
            test_files.extend(files)
    
    if not test_files:
        print("No test files found in data/closed_test/a directory")
        return
    
    test_file = test_files[0]
    
    try:
        print(f"\nClassifying: {test_file}")
        
        test_features = recognizer.extract_mfcc_features(test_file)
        
        # Try different distance metrics
        for metric in ['euclidean', 'mahalanobis', 'gaussian', 'negative_gaussian']:
            print(f"\nUsing {metric} distance:")
            try:
                predicted, distance, all_distances = recognizer.classify(test_features, distance_metric=metric)
                
                print(f"  Predicted vowel: {predicted}")
                print(f"  Distance: {distance:.2f}")
                print("  Distances to all vowels:")
                for vowel, dist in sorted(all_distances.items(), key=lambda x: x[1]):
                    print(f"    {vowel}: {dist:.2f}")
            except NotImplementedError as e:
                print(f"  {e}")
    except Exception as e:
        print(f"  Error: {e}")

def run_custom_evaluation():
    print("\n" + "="*60)
    print("Example: Custom Evaluation")
    print("="*60)
    
    recognizer = DTWRecognizer(data_dir='data')
    
    # Run evaluation with custom threshold and distance metric
    try:
        results = recognizer.evaluate(unknown_threshold=1500.0, distance_metric='mahalanobis')
        
        if results:
            print("\nDetailed Results:")
            print(f"  Closed-set: {results['closed_set']['correct']}/{results['closed_set']['total']}")
            print(f"  Open-set: {results['open_set']['correct']}/{results['open_set']['total']}")
            print(f"  Average: {results['average_accuracy']:.2f}%")
    except Exception as e:
        print(f"Error during evaluation: {e}")

def run_visualization():
    recognizer = DTWRecognizer(data_dir='data')
    recognizer.load_templates()
    
    if not recognizer.templates:
        print("No templates loaded. Please ensure training data exists in data/train/")
        return
    
    os.makedirs('visualizations', exist_ok=True)
    
    metrics = ['euclidean', 'mahalanobis', 'gaussian', 'negative_gaussian']
    
    for metric in metrics:
        print(f"\nGenerating visualizations for {metric} metric...")
        metric_dir = f'visualizations/{metric}'
        os.makedirs(metric_dir, exist_ok=True)
        
        results = recognizer.evaluate(distance_metric=metric)
        
        if results:
            recognizer.visualize_confusion_matrix(
                results['closed_set'], 
                title=f"Closed-Set Confusion Matrix ({metric})",
                save_path=f'{metric_dir}/confusion_matrix_closed.png'
            )
            
            recognizer.visualize_confusion_matrix(
                results['open_set'], 
                title=f"Open-Set Confusion Matrix ({metric})",
                save_path=f'{metric_dir}/confusion_matrix_open.png'
            )
            
            recognizer.visualize_distance_distribution(
                results['closed_set'],
                save_path=f'{metric_dir}/distance_distribution_closed.png'
            )
            
            recognizer.visualize_open_set_threshold_analysis(
                results['open_set'],
                save_path=f'{metric_dir}/threshold_analysis.png'
            )
            
            recognizer.visualize_accuracy_comparison(
                results['closed_set'],
                results['open_set'],
                save_path=f'{metric_dir}/accuracy_comparison.png'
            )
            
        else:
            print(f"  Error: Could not generate results for {metric} metric")
    
    recognizer.visualize_template_features(save_path='visualizations/template_features.png')
    
    sample_file = os.path.join('data', 'train', 'a', os.listdir(os.path.join('data', 'train', 'a'))[0])
    if os.path.exists(sample_file):
        recognizer.visualize_mfcc_features(
            sample_file,
            save_path='visualizations/mfcc_example.png'
        )
        
        template_a = recognizer.templates['a']['raw_features'][0]
        test_file = os.path.join('data', 'closed_test', 'a', os.listdir(os.path.join('data', 'closed_test', 'a'))[0])
        if os.path.exists(test_file):
            test_features = recognizer.extract_mfcc_features(test_file)
            recognizer.visualize_dtw_alignment(
                test_features,
                template_a,
                save_path='visualizations/dtw_alignment_example.png'
            )
    

def main():
    run_basic_usage()
    run_single_classification()
    run_custom_evaluation()
    run_visualization()

if __name__ == '__main__':
    main()
