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
from typing import Tuple, Dict, List
import utils


# ============================================================================
# PART 1: DATA PREPROCESSING (10 points)
# ============================================================================

def load_and_preprocess_imdb(vocab_size: int = 10000, max_length: int = 200,
                              train_size: int = 10000, val_size: int = 2500,
                              test_size: int = 5000) -> Tuple[Tuple[np.ndarray, np.ndarray],
                                                               Tuple[np.ndarray, np.ndarray],
                                                               Tuple[np.ndarray, np.ndarray]]:
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


def create_vocabulary_mappings(vocab_size: int = 10000) -> Tuple[Dict[str, int], Dict[int, str]]:
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

def build_random_embedding_model(vocab_size: int, embedding_dim: int,
                                 max_length: int) -> keras.Model:
    """
    Build a COMPLETE sentiment classification model with random embeddings.

    This builds an end-to-end model specifically for random embeddings.
    (See build_sentiment_classifier() for a more flexible approach.)

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
    - This builds a COMPLETE model (embeddings + classifier)
    - The embedding layer should be trainable
    - Compile the model with appropriate loss, optimizer, and metrics for binary classification
    - Make sure to compile before returning

    [4 points]
    """
    # TODO: Implement this function
    pass


def build_pretrained_embedding_model(vocab_size: int, embedding_dim: int,
                                     max_length: int,
                                     pretrained_embeddings: np.ndarray) -> keras.Model:
    """
    Build a COMPLETE sentiment classification model with pretrained embeddings.

    This builds an end-to-end model specifically for pretrained embeddings.
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
    - This builds a COMPLETE model (embeddings + classifier)
    - Initialize the embedding layer with pretrained_embeddings
    - You can choose whether to make embeddings trainable or frozen
    - Architecture should be similar to random embedding model
    - Look up how to pass pretrained weights to a Keras Embedding layer

    [4 points]
    """
    # TODO: Implement this function
    pass


# ============================================================================
# PART 3: AUTOENCODER EMBEDDINGS (15 points)
# ============================================================================

def build_autoencoder_encoder(vocab_size: int, embedding_dim: int,
                               max_length: int) -> keras.Model:
    """
    Build the encoder part of an autoencoder for learning text embeddings.

    The encoder takes a bag-of-words representation and compresses it
    to a lower-dimensional embedding (the "bottleneck").

    Architecture suggestion:
    - Input: bag-of-words vector of shape (vocab_size,)
    - Hidden layer(s): Dense layers to compress information
    - Bottleneck layer: Dense with embedding_dim units - this is your learned embedding
    - Choose appropriate activation functions

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
    - Choose appropriate activation functions for hidden and bottleneck layers
    - No need to compile this model - it will be used as part of the full autoencoder

    [7 points]
    """
    # TODO: Implement this function
    pass


def build_autoencoder_decoder(vocab_size: int, embedding_dim: int) -> keras.Model:
    """
    Build the decoder part of an autoencoder for learning text embeddings.

    The decoder takes the embedding and tries to reconstruct the
    original bag-of-words representation.

    Architecture suggestion:
    - Input: embedding vector of shape (embedding_dim,)
    - Hidden layer(s): Dense layers to expand information
    - Output layer: Dense with vocab_size units to reconstruct bag-of-words
    - Choose appropriate activation functions (output should be values between 0 and 1)

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
    - The decoder should be roughly symmetric to the encoder
    - No need to compile this model - it will be used as part of the full autoencoder

    [6 points]
    """
    # TODO: Implement this function
    pass


def train_autoencoder(encoder: keras.Model, decoder: keras.Model,
                      X_train: np.ndarray, epochs: int = 20,
                      batch_size: int = 128) -> keras.callbacks.History:
    """
    Train the autoencoder (encoder + decoder) on bag-of-words data.

    Parameters:
    -----------
    encoder : keras.Model
        Encoder model from build_autoencoder_encoder()
    decoder : keras.Model
        Decoder model from build_autoencoder_decoder()
    X_train : np.ndarray
        Training sequences of shape (n_samples, max_length)
        Will be converted to bag-of-words internally
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training

    Returns:
    --------
    history : keras.callbacks.History
        Training history object

    Notes:
    ------
    Step-by-step approach:
    1. Convert X_train to bag-of-words using utils.sequences_to_bow()
    2. Create a combined model:
       - Chain encoder and decoder: input → encoder → decoder → reconstruction
       - Use keras.Model with Input layer and chained outputs
    3. Compile the combined model with appropriate loss and optimizer
       - Consider what loss function makes sense for reconstructing bag-of-words
    4. Train the model:
       - Input and target are both the bag-of-words representation
       - The model learns to reconstruct its input through the bottleneck

    The autoencoder should compress and reconstruct: BoW → encoder → embedding → decoder → reconstructed BoW

    [2 points]
    """
    # TODO: Implement this function
    pass


# ============================================================================
# PART 4: WORD2VEC-STYLE EMBEDDINGS (15 points)
# ============================================================================

def generate_skipgram_pairs(sequences: np.ndarray, window_size: int = 2,
                             vocab_size: int = 10000) -> np.ndarray:
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


def build_word2vec_model(vocab_size: int, embedding_dim: int) -> keras.Model:
    """
    Build a Word2Vec-style model using skip-gram architecture.

    This model has TWO separate embedding layers (one for targets, one for contexts)
    and learns to predict whether a target-context pair is real.

    Architecture:
    - Two inputs: target word index and context word index
    - Target embedding layer (vocab_size, embedding_dim) - this is what you'll use later
    - Context embedding layer (vocab_size, embedding_dim) - used during training
    - Dot product between target and context embeddings
    - Sigmoid activation for binary classification (is this a real pair?)

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
    Step-by-step approach:
    1. Create two Input layers (one for target, one for context)
    2. Create two separate Embedding layers (target_embedding and context_embedding)
    3. Apply embeddings to the inputs
    4. Compute dot product between target and context embeddings
       - Result should be a single score per pair
       - Look up how to compute dot product in Keras/TensorFlow
    5. Apply sigmoid activation to get probability
    6. Create Model with both inputs and the probability output
    7. Compile with appropriate loss and optimizer for binary classification

    The target_embedding layer is what contains your learned word embeddings.

    [5 points]
    """
    # TODO: Implement this function
    pass


def train_word2vec(model: keras.Model, pairs: np.ndarray, epochs: int = 10,
                   batch_size: int = 128, validation_split: float = 0.1) -> keras.callbacks.History:
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
    history : keras.callbacks.History
        Training history object

    Notes:
    ------
    - Split pairs into target words (pairs[:, 0]) and context words (pairs[:, 1])
    - Labels should be all 1s (these are all real context pairs)
    - Call model.fit() with [targets, contexts] as input and labels as output
    - In a full implementation, you'd add negative samples (random non-context words)
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

        Note: Trainable weights should be created in build(), not here.
        Just call the parent class initializer.
        """
        super(SimpleAttention, self).__init__(**kwargs)
        # Trainable weights will be created in build()

    def build(self, input_shape):
        """
        Create trainable weights for computing attention scores.

        Parameters:
        -----------
        input_shape : tuple
            Shape of input: (batch_size, sequence_length, embedding_dim)

        Notes:
        ------
        You need to create a weight matrix to compute attention scores.

        Step-by-step approach:
        1. Get the embedding dimension from input_shape (last dimension)
        2. Create a trainable weight matrix of shape (embedding_dim, 1)
           - Use self.add_weight() to create trainable parameters
           - Initialize with appropriate random initialization
           - Give it a descriptive name
        3. Call super().build(input_shape) at the end

        This weight matrix will be used to compute a score for each word in the sequence.
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
            Boolean mask for padding tokens, shape (batch_size, sequence_length)

        Returns:
        --------
        output : tensor
            Attended representation of shape (batch_size, embedding_dim)
            This is the weighted sum of input embeddings

        Notes:
        ------
        Step-by-step approach:

        1. Compute attention scores:
           - Matrix multiply inputs with your weight matrix
           - Result should have shape (batch_size, sequence_length, 1)
           - Squeeze to get (batch_size, sequence_length)

        2. Apply mask (if provided):
           - Padding positions should not receive attention
           - Set masked positions to large negative values
           - This ensures they get ~0 weight after softmax

        3. Apply softmax:
           - Convert scores to normalized attention weights
           - Weights should sum to 1.0 across each sequence

        4. Apply attention weights:
           - Expand weights for broadcasting if needed
           - Multiply inputs by attention weights element-wise
           - Sum across the sequence dimension
           - Result shape: (batch_size, embedding_dim)

        Useful TensorFlow functions:
        - tf.matmul() - matrix multiplication
        - tf.squeeze() / tf.expand_dims() - manipulate dimensions
        - tf.where() - conditional selection
        - tf.nn.softmax() - normalize to probabilities
        - tf.reduce_sum() - sum along a dimension
        """
        # TODO: Implement this method
        pass

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape.

        THIS METHOD IS ALREADY COMPLETE - you don't need to modify it.
        It tells Keras what shape to expect from this layer.

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

def build_sentiment_classifier(embedding_model, max_length: int,
                                model_type: str = 'standard') -> keras.Model:
    """
    Build a sentiment classifier using a given embedding approach.

    This is a FLEXIBLE/GENERIC function that can work with any embedding method.
    It's different from the specific builder functions (build_random_embedding_model, etc.)
    which create complete models for a single approach. This function lets you plug in
    different embeddings to compare them.

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
    Different embedding types require different integration strategies.
    Here's what to do for each model_type:

    'standard': embedding_model is a keras.layers.Embedding layer
        - Create Input layer for sequences with appropriate shape
        - Apply embedding_model to the input
        - Add GlobalAveragePooling1D to get fixed-size vector
        - Add Dense layer(s) for classification
        - Add output layer with appropriate activation for binary classification

    'autoencoder': embedding_model is your trained encoder
        - Create Input layer for sequences
        - Convert sequences to bag-of-words (use Lambda layer with utils.sequences_to_bow)
        - Apply embedding_model (encoder) to bag-of-words
        - Add Dense layer(s) for classification
        - Add output layer for binary classification

    'word2vec': embedding_model is your trained Word2Vec model
        - Extract the target embedding weights from Word2Vec model
        - Create new Embedding layer with these weights
        - Follow same structure as 'standard'

    'attention': embedding_model is an Embedding layer
        - Create Input layer for sequences
        - Apply embedding_model to get embeddings
        - Apply SimpleAttention() layer to get attended representation
        - Add Dense layer(s) for classification
        - Add output layer for binary classification

    All models should be compiled with appropriate loss, optimizer, and metrics for binary classification

    [4 points]
    """
    # TODO: Implement this function
    pass


def evaluate_all_embeddings(models_dict: Dict[str, keras.Model],
                             X_test: np.ndarray,
                             y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Evaluate multiple trained models on the test set with the provided models.

    Parameters:
    -----------
    models_dict : dict[str, keras.Model]
        Dictionary mapping descriptive model names to trained classifier models.
        All models should already be trained and ready for evaluation.
        Example: {
            'Random Embeddings': random_model,
            'Word2Vec': word2vec_model,
            'Autoencoder': autoencoder_model,
            'Attention': attention_model
        }
    X_test : np.ndarray
        Test sequences of shape (n_samples, max_length)
    y_test : np.ndarray
        Test labels of shape (n_samples,)

    Returns:
    --------
    results : dict[str, dict[str, float]]
        Dictionary mapping model names to their evaluation metrics.
        Each entry should contain at least 'accuracy' and 'loss'.
        Example: {
            'Random Embeddings': {'accuracy': 0.75, 'loss': 0.52},
            'Word2Vec': {'accuracy': 0.82, 'loss': 0.43},
            ...
        }

    Notes:
    ------
    Step-by-step approach:
    1. Create an empty results dictionary
    2. For each model name and model in models_dict:
       - Evaluate the model on test set to get metrics
       - Store in results dictionary with structure: {'accuracy': acc, 'loss': loss}
    3. Return the results dictionary

    Look up how to evaluate a trained Keras model and extract the metrics.

    [3 points]
    """
    # TODO: Implement this function
    pass


def extract_embeddings(model: keras.Model, words: List[str],
                       word_to_idx: Dict[str, int],
                       model_type: str = 'standard') -> np.ndarray:
    """
    Extract embedding vectors for specific words from a trained model.

    This is useful for visualization and analysis in Part 2.

    Parameters:
    -----------
    model : keras.Model
        Trained model containing embeddings
    words : list of str
        List of words to extract embeddings for
    word_to_idx : dict[str, int]
        Dictionary mapping words to indices
    model_type : str
        Type of embedding: 'standard', 'word2vec', 'autoencoder', 'attention'

    Returns:
    --------
    embeddings : np.ndarray
        Array of shape (len(words), embedding_dim) containing embedding vectors

    Notes:
    ------
    Different model types store embeddings in different places:

    'standard' or 'attention':
        - Find the Embedding layer in the model
        - Extract its weights: embedding_layer.get_weights()[0]
        - Look up word indices and return corresponding rows

    'word2vec':
        - Similar to 'standard', but find the target embedding layer
        - May be named or be the first Embedding layer

    'autoencoder':
        - Create one-hot or bag-of-words vectors for the words
        - Pass through the encoder to get embeddings

    Handle words not in vocabulary:
    - Return zero vector for unknown words
    - Or skip them entirely (adjust output shape accordingly)

    [3 points]
    """
    # TODO: Implement this function
    pass


# ============================================================================
# HELPER FUNCTIONS (Optional - not graded)
# ============================================================================

def create_simple_pretrained_embeddings(vocab_size: int, embedding_dim: int,
                                       seed: int = 42) -> np.ndarray:
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
