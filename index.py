import numpy as np

def generate_train_test_indices(num_samples, train_ratio=0.8):
    """
    Generate train and test indices.

    Parameters:
    - num_samples: int, total number of samples in the dataset.
    - train_ratio: float, proportion of data to be used for training; the rest is for testing.

    Returns:
    - train_idx: numpy.ndarray, array of training set indices.
    - test_idx: numpy.ndarray, array of test set indices.
    """

    # Create an array of indices representing each sample
    indices = np.arange(num_samples)

    # Shuffle the indices
    np.random.seed(42)  # Set random seed for reproducibility
    np.random.shuffle(indices)

    # Calculate the size of the training set
    train_size = int(num_samples * train_ratio)

    # Split the indices into training and testing sets
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    return train_idx, test_idx

# Assume there are 196 samples
num_samples = 196

# Generate indices
train_idx, test_idx = generate_train_test_indices(num_samples)
