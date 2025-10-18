import os
import numpy as np
from scipy.io import wavfile

vowels = {
    'a': 700,
    'i': 300,
    'u': 400,
    'e': 500,
    'o': 600,
}

sample_rate = 16000  # 16 kHz
duration = 0.5       # 0.5 seconds
amplitude = 0.3

def generate_vowel_sound(frequency, duration, sample_rate, amplitude=0.3):
    """Generate a simple sine wave to simulate vowel sound"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Add some harmonics for more realistic sound
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    signal += 0.2 * amplitude * np.sin(2 * np.pi * frequency * 2 * t)
    signal += 0.1 * amplitude * np.sin(2 * np.pi * frequency * 3 * t)
    
    # Add slight random noise
    signal += 0.01 * np.random.randn(len(t))
    
    # Convert to int16
    signal = np.int16(signal * 32767)
    
    return signal

def create_sample_dataset(base_dir='data'):
    """Create sample dataset with template and test files"""
    
    for vowel, frequency in vowels.items():
        vowel_dir = os.path.join(base_dir, vowel)
        os.makedirs(vowel_dir, exist_ok=True)
        
        print(f"Generating samples for vowel '{vowel}'...")
        
        # Create 5 template files (P1-P5)
        for i in range(1, 6):
            filename = os.path.join(vowel_dir, f'template_p{i}.wav')
            # Add slight variation to each template
            freq_variation = frequency + np.random.uniform(-10, 10)
            signal = generate_vowel_sound(freq_variation, duration, sample_rate, amplitude)
            wavfile.write(filename, sample_rate, signal)
            print(f"  Created {os.path.basename(filename)}")
        
        # Create 3 test files (uji_p1, uji_p2, uji_p3)
        for i in range(1, 4):
            filename = os.path.join(vowel_dir, f'uji_p{i}.wav')
            # Add more variation to test files
            freq_variation = frequency + np.random.uniform(-15, 15)
            signal = generate_vowel_sound(freq_variation, duration, sample_rate, amplitude)
            wavfile.write(filename, sample_rate, signal)
            print(f"  Created {os.path.basename(filename)}")
    
    print("\nSample dataset created successfully!")
    print("Note: These are synthetic samples for testing. Replace with real recordings for actual use.")

if __name__ == '__main__':
    create_sample_dataset()
