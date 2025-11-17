"""
Utility functions for PA5: Text Representation Learning
These helper functions are provided to students for dataset loading and basic preprocessing.
"""

import numpy as np
from tensorflow import keras


# Random seed for reproducibility (consistent with PA4)
RANDOM_SEED = 42


def load_imdb_raw(num_words=10000):
    """
    Load the raw IMDB dataset from Keras.

    Parameters:
    -----------
    num_words : int
        Keep only the top num_words most frequent words

    Returns:
    --------
    (X_train, y_train), (X_test, y_test) : tuple of arrays
        Training and test data where X contains sequences of word indices
        and y contains binary labels (0=negative, 1=positive)
    """
    np.random.seed(RANDOM_SEED)
    (X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(
        num_words=num_words,
        seed=RANDOM_SEED
    )
    return (X_train, y_train), (X_test, y_test)


def get_word_index():
    """
    Get the IMDB word index dictionary.

    Returns:
    --------
    word_index : dict
        Dictionary mapping words to integer indices
    """
    return keras.datasets.imdb.get_word_index()


def decode_review(sequence, word_index=None, reverse=True):
    """
    Decode a sequence of word indices back to text.
    Useful for visualization and debugging.

    Parameters:
    -----------
    sequence : list of int
        Sequence of word indices
    word_index : dict, optional
        Word to index mapping. If None, loads from Keras.
    reverse : bool
        If True, reverse the word_index to get index_to_word mapping

    Returns:
    --------
    text : str
        Decoded review text
    """
    if word_index is None:
        word_index = get_word_index()

    if reverse:
        # Reverse the word index to get index to word mapping
        # Note: indices are offset by 3 because 0, 1, 2 are reserved
        reverse_word_index = {v + 3: k for k, v in word_index.items()}
        reverse_word_index[0] = '<PAD>'
        reverse_word_index[1] = '<START>'
        reverse_word_index[2] = '<UNK>'
        return ' '.join([reverse_word_index.get(i, '?') for i in sequence])
    else:
        return ' '.join([word_index.get(i, '?') for i in sequence])


def create_stratified_subset(X, y, subset_size, random_seed=RANDOM_SEED):
    """
    Create a stratified subset of data maintaining class balance.

    Parameters:
    -----------
    X : array-like
        Input data
    y : array-like
        Labels
    subset_size : int
        Desired size of subset
    random_seed : int
        Random seed for reproducibility

    Returns:
    --------
    X_subset, y_subset : arrays
        Subset of data maintaining class balance
    """
    np.random.seed(random_seed)

    # Get indices for each class
    class_0_indices = np.where(y == 0)[0]
    class_1_indices = np.where(y == 1)[0]

    # Sample half from each class
    n_per_class = subset_size // 2

    selected_0 = np.random.choice(class_0_indices, size=n_per_class, replace=False)
    selected_1 = np.random.choice(class_1_indices, size=n_per_class, replace=False)

    # Combine and shuffle
    selected_indices = np.concatenate([selected_0, selected_1])
    np.random.shuffle(selected_indices)

    return X[selected_indices], y[selected_indices]


def pad_sequences_custom(sequences, maxlen, padding='post', truncating='post', value=0):
    """
    Simple wrapper around Keras pad_sequences for consistency.

    Parameters:
    -----------
    sequences : list of lists
        List of sequences (lists of integers)
    maxlen : int
        Maximum length of sequences
    padding : str
        'pre' or 'post', pad before or after each sequence
    truncating : str
        'pre' or 'post', remove values from beginning or end of sequences
    value : int
        Padding value

    Returns:
    --------
    padded : np.ndarray
        Padded sequences array of shape (len(sequences), maxlen)
    """
    return keras.preprocessing.sequence.pad_sequences(
        sequences,
        maxlen=maxlen,
        padding=padding,
        truncating=truncating,
        value=value
    )


def sequences_to_bow(sequences, vocab_size):
    """
    Convert sequences of word indices to bag-of-words representation.
    Useful for autoencoder input.

    Parameters:
    -----------
    sequences : array-like
        Sequences of word indices, shape (n_samples, sequence_length)
    vocab_size : int
        Size of vocabulary

    Returns:
    --------
    bow : np.ndarray
        Bag-of-words matrix, shape (n_samples, vocab_size)
        Each row contains counts of words in that sequence
    """
    bow = np.zeros((len(sequences), vocab_size), dtype=np.float32)

    for i, seq in enumerate(sequences):
        for word_idx in seq:
            if word_idx < vocab_size:
                bow[i, word_idx] += 1

    return bow


def print_data_statistics(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Print useful statistics about the dataset splits.

    Parameters:
    -----------
    X_train, y_train : arrays
        Training data and labels
    X_val, y_val : arrays
        Validation data and labels
    X_test, y_test : arrays
        Test data and labels
    """
    print("Dataset Statistics:")
    print(f"  Training samples: {len(X_train)} (positive: {np.sum(y_train)}, negative: {len(y_train) - np.sum(y_train)})")
    print(f"  Validation samples: {len(X_val)} (positive: {np.sum(y_val)}, negative: {len(y_val) - np.sum(y_val)})")
    print(f"  Test samples: {len(X_test)} (positive: {np.sum(y_test)}, negative: {len(y_test) - np.sum(y_test)})")

    if len(X_train) > 0:
        train_lengths = [len([w for w in seq if w != 0]) for seq in X_train[:100]]  # Sample for efficiency
        print(f"  Average review length (sample): {np.mean(train_lengths):.1f} words")
