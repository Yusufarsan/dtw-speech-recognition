# Implementation Guide for Mean and Covariance-Based Distance Metrics

This guide provides implementation suggestions for completing the TODO items in the DTW speech recognition system.

## 1. Mean Template Calculation

### Challenge
MFCC features are variable-length sequences (different number of frames per audio file). You need to compute a representative mean template.

### Approach Options

#### Option A: Global Feature Averaging (Simplest)
Concatenate all feature vectors and compute the mean across all frames.

```python
def calculate_mean_template(self, features_list):
    """Calculate mean template by averaging all feature vectors"""
    all_features = []
    for features in features_list:
        # features shape: (n_frames, 39)
        all_features.append(features)
    
    # Concatenate all features
    all_features_concat = np.vstack(all_features)  # (total_frames, 39)
    
    # Compute mean across all frames
    mean_template = np.mean(all_features_concat, axis=0)  # (39,)
    
    return mean_template
```

**Pros**: Simple, fast
**Cons**: Ignores temporal structure, treats all frames equally

#### Option B: DTW-Based Alignment and Averaging
Align all sequences using DTW, then average aligned frames.

```python
def calculate_mean_template(self, features_list):
    """Calculate mean template using DTW alignment"""
    if len(features_list) == 0:
        return None
    
    # Use first sample as initial reference
    reference = features_list[0]
    
    # Align all other samples to the reference
    aligned_features = [reference]
    
    for features in features_list[1:]:
        alignment = dtw(reference, features, dist_method='euclidean')
        # Warp features according to alignment path
        warped = self._warp_sequence(features, alignment.index2)
        aligned_features.append(warped)
    
    # Average aligned sequences
    mean_template = np.mean(aligned_features, axis=0)
    
    return mean_template

def _warp_sequence(self, sequence, index_mapping):
    """Warp sequence according to DTW index mapping"""
    # Implementation depends on DTW library output
    pass
```

**Pros**: Preserves temporal structure
**Cons**: More complex, computationally expensive

#### Option C: Fixed-Length Representation
Interpolate all sequences to fixed length, then average.

```python
def calculate_mean_template(self, features_list, target_length=100):
    """Calculate mean template with fixed-length interpolation"""
    from scipy.interpolate import interp1d
    
    resampled_features = []
    
    for features in features_list:
        n_frames, n_features = features.shape
        
        # Create interpolation function for each feature dimension
        x_old = np.linspace(0, 1, n_frames)
        x_new = np.linspace(0, 1, target_length)
        
        resampled = np.zeros((target_length, n_features))
        for i in range(n_features):
            f = interp1d(x_old, features[:, i], kind='linear')
            resampled[:, i] = f(x_new)
        
        resampled_features.append(resampled)
    
    # Average across all samples
    mean_template = np.mean(resampled_features, axis=0)  # (target_length, 39)
    
    return mean_template
```

**Pros**: Easy to work with, preserves temporal structure
**Cons**: Fixed length may not suit all applications

### Recommended: Option A (Start Simple)
For initial implementation, use Option A. You can improve later with Options B or C.

---

## 2. Covariance Matrix Calculation

### Challenge
Calculate covariance matrix that captures feature correlations.

### Approach Options

#### Option A: Global Covariance (Simplest)
Compute covariance across all feature vectors (ignoring temporal structure).

```python
def calculate_covariance_matrix(self, features_list, mean_template=None):
    """Calculate covariance matrix from all feature vectors"""
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
```

**Pros**: Simple, standard approach
**Cons**: May be singular or ill-conditioned

#### Option B: Diagonal Covariance
Assume features are independent (diagonal covariance matrix).

```python
def calculate_covariance_matrix(self, features_list, mean_template=None):
    """Calculate diagonal covariance matrix"""
    all_features = np.vstack(features_list)
    
    if mean_template is None:
        mean_template = np.mean(all_features, axis=0)
    
    # Calculate variance for each feature dimension
    variances = np.var(all_features, axis=0)  # (39,)
    
    # Create diagonal covariance matrix
    cov_matrix = np.diag(variances)
    
    # Add regularization
    regularization = 1e-6
    cov_matrix += regularization * np.eye(cov_matrix.shape[0])
    
    return cov_matrix
```

**Pros**: Computationally efficient, always invertible
**Cons**: Ignores feature correlations

### Recommended: Option A with Regularization
Use full covariance with strong regularization to ensure numerical stability.

---

## 3. Mahalanobis Distance

### Implementation

```python
def calculate_mahalanobis_distance(self, test_features, mean_template, covariance_matrix):
    """Calculate Mahalanobis distance"""
    from scipy.spatial.distance import mahalanobis
    from scipy.linalg import inv
    
    # Flatten test features (or take mean)
    if len(test_features.shape) > 1:
        # Option 1: Use mean of all frames
        test_vector = np.mean(test_features, axis=0)  # (39,)
    else:
        test_vector = test_features
    
    # Invert covariance matrix
    try:
        cov_inv = inv(covariance_matrix)
    except np.linalg.LinAlgError:
        # If singular, use pseudo-inverse
        cov_inv = np.linalg.pinv(covariance_matrix)
    
    # Calculate Mahalanobis distance
    diff = test_vector - mean_template
    distance = np.sqrt(diff.T @ cov_inv @ diff)
    
    return distance
```

**Note**: If test_features is a sequence, you need to aggregate it (e.g., take mean, or compute frame-wise distances).

---

## 4. Gaussian Likelihood

### Implementation

```python
def calculate_gaussian_likelihood(self, test_features, mean_template, covariance_matrix):
    """Calculate Gaussian log-likelihood"""
    from scipy.stats import multivariate_normal
    
    # Flatten test features (or take mean)
    if len(test_features.shape) > 1:
        test_vector = np.mean(test_features, axis=0)
    else:
        test_vector = test_features
    
    # Calculate log-likelihood
    try:
        log_likelihood = multivariate_normal.logpdf(
            test_vector, 
            mean=mean_template, 
            cov=covariance_matrix,
            allow_singular=True
        )
    except Exception as e:
        # Fallback to large negative value if computation fails
        log_likelihood = -1e10
    
    return log_likelihood
```

**Alternative: Frame-wise likelihood**
```python
def calculate_gaussian_likelihood(self, test_features, mean_template, covariance_matrix):
    """Calculate frame-wise Gaussian log-likelihood"""
    from scipy.stats import multivariate_normal
    
    # Calculate likelihood for each frame
    log_likelihoods = []
    for frame in test_features:
        ll = multivariate_normal.logpdf(
            frame,
            mean=mean_template,
            cov=covariance_matrix,
            allow_singular=True
        )
        log_likelihoods.append(ll)
    
    # Return average or sum
    return np.mean(log_likelihoods)
```

---

## 5. Integration Steps

### Step 1: Implement mean and covariance functions
Add the implementations to `dtw_recognizer.py`.

### Step 2: Update `load_templates()` method
```python
# Inside load_templates(), after extracting all features:
if all_features:
    # Calculate mean and covariance
    mean_template = self.calculate_mean_template(all_features)
    covariance_matrix = self.calculate_covariance_matrix(all_features, mean_template)
    
    self.templates[vowel] = {
        'raw_features': all_features,
        'num_samples': len(all_features),
        'mean': mean_template,
        'covariance': covariance_matrix
    }
```

### Step 3: Update distance metric implementations in `classify()`
```python
elif distance_metric == 'mahalanobis':
    distance = self.calculate_mahalanobis_distance(
        test_features, 
        template_data['mean'], 
        template_data['covariance']
    )
    distances[vowel] = distance
```

### Step 4: Test with each metric
```bash
python dtw_recognizer.py --distance-metric euclidean
python dtw_recognizer.py --distance-metric mahalanobis
python dtw_recognizer.py --distance-metric gaussian
python dtw_recognizer.py --distance-metric negative_gaussian
```

---

## Tips and Best Practices

1. **Start Simple**: Implement Option A for both mean and covariance first
2. **Test Incrementally**: Test each function independently before integration
3. **Handle Edge Cases**: Check for empty lists, singular matrices, NaN values
4. **Add Regularization**: Always add small constant to covariance diagonal
5. **Normalize Features**: Consider feature normalization for better numerical stability
6. **Debug Output**: Add print statements to verify template shapes and values
7. **Compare Metrics**: Run all metrics on same data to compare performance

---

## Expected Performance

- **Euclidean (DTW)**: Baseline performance, works well for time-series
- **Mahalanobis**: May improve if features have strong correlations
- **Gaussian**: Good for probabilistic interpretation, similar to Mahalanobis
- **Negative Gaussian**: Same as Gaussian, just expressed as distance

The best metric depends on your data characteristics and application requirements.
