"""
PA5: Text Representation Learning - A Comparative Study
Student Implementation File

In this assignment, you will implement and compare different approaches to learning
text representations: random embeddings, autoencoders, Word2Vec, and attention mechanisms.
All implementations should use TensorFlow/Keras.

Total: 70 points
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import utils


# ============================================================================
# PART 1: DATA PREPROCESSING (10 points)
# ============================================================================

def load_and_preprocess_imdb(vocab_size=10000, max_length=200,
                              train_size=10000, val_size=2500, test_size=5000):
    """
    Load and preprocess the IMDB dataset for sentiment classification.

    This function should:
    1. Load the raw IMDB data using utils.load_imdb_raw()
    2. Create stratified subsets of the specified sizes
    3. Pad/truncate sequences to max_length
    4. Return properly formatted train/val/test splits

    Parameters:
    -----------
    vocab_size : int
        Number of most frequent words to keep
    max_length : int
        Maximum length of sequences (pad or truncate to this length)
    train_size : int
        Number of training samples (should be stratified)
    val_size : int
        Number of validation samples (from original train set)
    test_size : int
        Number of test samples (stratified)

    Returns:
    --------
    (X_train, y_train), (X_val, y_val), (X_test, y_test) : tuple of tuples
        Each X is a numpy array of shape (n_samples, max_length) containing word indices
        Each y is a numpy array of shape (n_samples,) containing binary labels

    Notes:
    ------
    - Use utils.create_stratified_subset() to maintain class balance
    - Use utils.pad_sequences_custom() for sequence padding
    - Validation set should come from the original training data
    - Ensure reproducibility by using the random seed from utils

    [5 points]
    """
    # TODO: Implement this function
    pass


def create_vocabulary_mappings(vocab_size=10000):
    """
    Create mappings between words and indices for the IMDB vocabulary.

    Parameters:
    -----------
    vocab_size : int
        Number of most frequent words to include

    Returns:
    --------
    word_to_idx : dict
        Dictionary mapping words (str) to indices (int)
    idx_to_word : dict
        Dictionary mapping indices (int) to words (str)

    Notes:
    ------
    - Use utils.get_word_index() to get the base word index
    - Remember that Keras reserves indices 0 (pad), 1 (start), 2 (unknown)
    - Only include words with indices < vocab_size
    - Add special tokens to idx_to_word: {0: '<PAD>', 1: '<START>', 2: '<UNK>'}

    [5 points]
    """
    # TODO: Implement this function
    pass


# ============================================================================
# PART 2: BASELINE EMBEDDINGS (8 points)
# ============================================================================

def build_random_embedding_model(vocab_size, embedding_dim, max_length):
    """
    Build a simple sentiment classification model with random embeddings.

    Architecture:
    - Embedding layer (randomly initialized, trainable)
    - Global average pooling (to get fixed-size representation)
    - Dense layer(s) for classification
    - Output layer with sigmoid activation (binary classification)

    Parameters:
    -----------
    vocab_size : int
        Size of vocabulary
    embedding_dim : int
        Dimension of embedding vectors
    max_length : int
        Maximum sequence length

    Returns:
    --------
    model : keras.Model
        Compiled model ready for training

    Notes:
    ------
    - Use 'binary_crossentropy' loss and 'accuracy' metric
    - Choose an appropriate optimizer (Adam is a good default)
    - The embedding layer should be trainable

    [4 points]
    """
    # TODO: Implement this function
    pass


def build_pretrained_embedding_model(vocab_size, embedding_dim, max_length,
                                     pretrained_embeddings):
    """
    Build a sentiment classification model with pretrained embeddings.

    Similar to random embedding model, but uses pretrained embedding weights.

    Parameters:
    -----------
    vocab_size : int
        Size of vocabulary
    embedding_dim : int
        Dimension of embedding vectors
    max_length : int
        Maximum sequence length
    pretrained_embeddings : np.ndarray
        Pretrained embedding matrix of shape (vocab_size, embedding_dim)

    Returns:
    --------
    model : keras.Model
        Compiled model ready for training

    Notes:
    ------
    - Initialize the embedding layer with pretrained_embeddings
    - You can choose whether to make embeddings trainable or frozen
    - Architecture should be similar to random embedding model

    [4 points]
    """
    # TODO: Implement this function
    pass


# ============================================================================
# PART 3: AUTOENCODER EMBEDDINGS (15 points)
# ============================================================================

def build_autoencoder_encoder(vocab_size, embedding_dim, max_length):
    """
    Build the encoder part of an autoencoder for learning text embeddings.

    The encoder takes a bag-of-words representation and compresses it
    to a lower-dimensional embedding (the "bottleneck").

    Architecture suggestion:
    - Input: bag-of-words vector (vocab_size,)
    - Dense layer(s) to compress information
    - Bottleneck layer (embedding_dim,) - this is your embedding

    Parameters:
    -----------
    vocab_size : int
        Size of vocabulary (input dimension)
    embedding_dim : int
        Dimension of learned embedding (bottleneck size)
    max_length : int
        Not used for autoencoder, but kept for consistency

    Returns:
    --------
    encoder : keras.Model
        Encoder model that outputs embeddings

    Notes:
    ------
    - Use bag-of-words as input (not sequences)
    - The bottleneck layer produces the embedding
    - Consider using ReLU or tanh activations in hidden layers

    [7 points]
    """
    # TODO: Implement this function
    pass


def build_autoencoder_decoder(vocab_size, embedding_dim):
    """
    Build the decoder part of an autoencoder for learning text embeddings.

    The decoder takes the embedding and tries to reconstruct the
    original bag-of-words representation.

    Architecture suggestion:
    - Input: embedding vector (embedding_dim,)
    - Dense layer(s) to expand information
    - Output layer (vocab_size,) to reconstruct bag-of-words

    Parameters:
    -----------
    vocab_size : int
        Size of vocabulary (output dimension)
    embedding_dim : int
        Dimension of embedding (input dimension)

    Returns:
    --------
    decoder : keras.Model
        Decoder model that reconstructs from embeddings

    Notes:
    ------
    - Output should match the input bag-of-words representation
    - Consider using sigmoid or softmax activation for output
    - The decoder is symmetric to the encoder

    [6 points]
    """
    # TODO: Implement this function
    pass


def train_autoencoder(encoder, decoder, X_train, epochs=20, batch_size=128):
    """
    Train the autoencoder (encoder + decoder) on bag-of-words data.

    Parameters:
    -----------
    encoder : keras.Model
        Encoder model
    decoder : keras.Model
        Decoder model
    X_train : np.ndarray
        Training sequences of shape (n_samples, max_length)
        Will be converted to bag-of-words internally
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training

    Returns:
    --------
    history : keras.History
        Training history

    Notes:
    ------
    - Convert X_train to bag-of-words using utils.sequences_to_bow()
    - Chain encoder and decoder: input -> encoder -> decoder -> reconstruction
    - Use MSE or binary crossentropy loss
    - The autoencoder should reconstruct its input

    [2 points]
    """
    # TODO: Implement this function
    pass


# ============================================================================
# PART 4: WORD2VEC-STYLE EMBEDDINGS (15 points)
# ============================================================================

def generate_skipgram_pairs(sequences, window_size=2, vocab_size=10000):
    """
    Generate (target, context) pairs for skip-gram training.

    For each word in each sequence, create pairs with words within the window.
    Example: "the cat sat on mat" with window_size=2
    - "sat" -> "the", "cat", "on", "mat"

    Parameters:
    -----------
    sequences : list or np.ndarray
        List of sequences, each sequence is a list of word indices
    window_size : int
        Number of words to consider on each side of target word
    vocab_size : int
        Maximum vocabulary size (filter out indices >= vocab_size)

    Returns:
    --------
    pairs : np.ndarray
        Array of shape (n_pairs, 2) where each row is [target, context]

    Notes:
    ------
    - Skip padding tokens (index 0)
    - Only include words with index < vocab_size
    - Each target word pairs with all words in its window
    - Window is symmetric (words on both sides)

    [6 points]
    """
    # TODO: Implement this function
    pass


def build_word2vec_model(vocab_size, embedding_dim):
    """
    Build a Word2Vec-style model using skip-gram architecture.

    Architecture:
    - Target embedding layer (vocab_size, embedding_dim)
    - Context embedding layer (vocab_size, embedding_dim)
    - Dot product between target and context embeddings
    - Sigmoid activation for binary classification

    Parameters:
    -----------
    vocab_size : int
        Size of vocabulary
    embedding_dim : int
        Dimension of embedding vectors

    Returns:
    --------
    model : keras.Model
        Compiled Word2Vec model

    Notes:
    ------
    - Use two separate embedding layers (target and context)
    - The target embedding is what you'll extract later
    - Use binary crossentropy loss (predicting if context is real or negative sample)
    - This is a simplified version; you can use negative sampling implicitly

    [5 points]
    """
    # TODO: Implement this function
    pass


def train_word2vec(model, pairs, epochs=10, batch_size=128, validation_split=0.1):
    """
    Train the Word2Vec model on (target, context) pairs.

    Parameters:
    -----------
    model : keras.Model
        Word2Vec model from build_word2vec_model()
    pairs : np.ndarray
        Array of shape (n_pairs, 2) with [target, context] pairs
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    validation_split : float
        Fraction of data to use for validation

    Returns:
    --------
    history : keras.History
        Training history

    Notes:
    ------
    - Labels should be all 1s (real context pairs)
    - In a full implementation, you'd add negative samples
    - For this assignment, training on positive pairs is sufficient

    [4 points]
    """
    # TODO: Implement this function
    pass


# ============================================================================
# PART 5: SIMPLE ATTENTION MECHANISM (12 points)
# ============================================================================

class SimpleAttention(layers.Layer):
    """
    A simple self-attention layer for creating sentence embeddings.

    This layer computes attention weights for each word in a sequence
    and returns a weighted sum of the embeddings (a single vector representing
    the entire sequence).

    Architecture:
    1. Compute attention scores for each word
    2. Apply softmax to get attention weights
    3. Return weighted sum of input embeddings

    This is a simplified version - single attention head, no multi-head complexity.

    [12 points total]
    """

    def __init__(self, **kwargs):
        """
        Initialize the attention layer.

        You may want to create trainable weight matrices here.
        """
        super(SimpleAttention, self).__init__(**kwargs)
        # TODO: Initialize any trainable parameters you need

    def build(self, input_shape):
        """
        Create trainable weights.

        Parameters:
        -----------
        input_shape : tuple
            Shape of input: (batch_size, sequence_length, embedding_dim)

        Notes:
        ------
        - Create a weight matrix to compute attention scores
        - You might want a matrix of shape (embedding_dim, 1) to compute scores
        - Don't forget to call super().build(input_shape)
        """
        # TODO: Implement this method
        super(SimpleAttention, self).build(input_shape)

    def call(self, inputs, mask=None):
        """
        Apply attention to inputs.

        Parameters:
        -----------
        inputs : tensor
            Input embeddings of shape (batch_size, sequence_length, embedding_dim)
        mask : tensor, optional
            Mask for padding tokens

        Returns:
        --------
        output : tensor
            Attended representation of shape (batch_size, embedding_dim)
            This is the weighted sum of input embeddings

        Notes:
        ------
        - Compute attention scores for each position
        - Apply softmax to get weights
        - Handle masking if provided (set padding positions to large negative)
        - Return weighted sum of inputs
        """
        # TODO: Implement this method
        pass

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape.

        Parameters:
        -----------
        input_shape : tuple
            (batch_size, sequence_length, embedding_dim)

        Returns:
        --------
        output_shape : tuple
            (batch_size, embedding_dim)
        """
        return (input_shape[0], input_shape[-1])


# ============================================================================
# PART 6: SHARED CLASSIFIER & EVALUATION (10 points)
# ============================================================================

def build_sentiment_classifier(embedding_model, max_length, model_type='standard'):
    """
    Build a sentiment classifier using a given embedding approach.

    This function takes an embedding model and adds a classification head.

    Parameters:
    -----------
    embedding_model : keras.Model or keras.layers.Layer
        The embedding model (can be embedding layer, encoder, etc.)
    max_length : int
        Maximum sequence length
    model_type : str
        Type of embedding: 'standard', 'autoencoder', 'word2vec', 'attention'
        This determines how to integrate the embedding

    Returns:
    --------
    model : keras.Model
        Complete sentiment classification model

    Notes:
    ------
    - Different embedding types may require different integration strategies
    - For 'autoencoder': convert sequences to BoW first
    - For 'attention': apply attention layer to embeddings
    - Add appropriate classification layers on top
    - Use sigmoid activation for binary classification

    [4 points]
    """
    # TODO: Implement this function
    pass


def evaluate_all_embeddings(models_dict, X_test, y_test):
    """
    Evaluate all embedding approaches on the test set.

    Parameters:
    -----------
    models_dict : dict
        Dictionary mapping model names to trained models
        Example: {'random': model1, 'word2vec': model2, ...}
    X_test : np.ndarray
        Test sequences
    y_test : np.ndarray
        Test labels

    Returns:
    --------
    results : dict
        Dictionary mapping model names to evaluation metrics
        Each entry should contain at least 'accuracy' and 'loss'

    Notes:
    ------
    - Evaluate each model on the test set
    - Return metrics in a structured format for comparison
    - Consider including additional metrics (precision, recall, etc.)

    [3 points]
    """
    # TODO: Implement this function
    pass


def extract_embeddings(model, words, word_to_idx, model_type='standard'):
    """
    Extract embedding vectors for specific words.

    This is useful for visualization and analysis.

    Parameters:
    -----------
    model : keras.Model
        Trained model containing embeddings
    words : list of str
        List of words to extract embeddings for
    word_to_idx : dict
        Dictionary mapping words to indices
    model_type : str
        Type of embedding: 'standard', 'word2vec', 'autoencoder', 'attention'

    Returns:
    --------
    embeddings : np.ndarray
        Array of shape (len(words), embedding_dim) containing embedding vectors

    Notes:
    ------
    - Handle words not in vocabulary (return zero vector or skip)
    - Different model types store embeddings differently
    - For standard models: extract from embedding layer
    - For Word2Vec: extract from target embedding layer
    - For autoencoder: may need to encode one-hot vectors

    [3 points]
    """
    # TODO: Implement this function
    pass


# ============================================================================
# HELPER FUNCTIONS (Optional - not graded)
# ============================================================================

def create_simple_pretrained_embeddings(vocab_size, embedding_dim, seed=42):
    """
    Create simple "pretrained" embeddings for the baseline.

    In a real scenario, you'd load GloVe or Word2Vec embeddings.
    For this assignment, we'll create random embeddings as a placeholder.

    Parameters:
    -----------
    vocab_size : int
        Size of vocabulary
    embedding_dim : int
        Dimension of embeddings
    seed : int
        Random seed

    Returns:
    --------
    embeddings : np.ndarray
        Embedding matrix of shape (vocab_size, embedding_dim)
    """
    np.random.seed(seed)
    # Initialize with small random values
    embeddings = np.random.uniform(-0.1, 0.1, (vocab_size, embedding_dim))
    # Set padding embedding to zeros
    embeddings[0] = 0
    return embeddings.astype(np.float32)


if __name__ == '__main__':
    """
    You can use this section to test your implementations.
    This code will not be graded.
    """
    print("PA5: Text Representation Learning")
    print("=" * 50)

    # Test data loading
    print("\n1. Testing data loading...")
    try:
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_preprocess_imdb(
            vocab_size=10000,
            max_length=200,
            train_size=1000,  # Small for testing
            val_size=250,
            test_size=500
        )
        print(f"✓ Data loaded: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")

    # Test vocabulary mappings
    print("\n2. Testing vocabulary mappings...")
    try:
        word_to_idx, idx_to_word = create_vocabulary_mappings(vocab_size=10000)
        print(f"✓ Vocabulary created: {len(word_to_idx)} words")
    except Exception as e:
        print(f"✗ Error creating vocabulary: {e}")

    # Add more tests as you implement functions
    print("\n" + "=" * 50)
    print("Run tests.py for comprehensive testing")
