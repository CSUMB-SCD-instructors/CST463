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
    """Load and preprocess the IMDB dataset for sentiment classification."""
    # Load raw IMDB data
    (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = utils.load_imdb_raw(num_words=vocab_size)

    # Create train and validation subsets from original training data
    combined_size = train_size + val_size
    X_train_combined, y_train_combined = utils.create_stratified_subset(
        X_train_raw, y_train_raw, combined_size
    )

    # Split into train and validation
    X_train, y_train = utils.create_stratified_subset(
        X_train_combined, y_train_combined, train_size
    )

    # Create validation set from remaining data
    val_indices = []
    for label in [0, 1]:
        label_indices = np.where(y_train_combined == label)[0]
        # Get indices not in train set
        train_label_indices = np.where(y_train == label)[0]
        remaining_indices = [i for i in label_indices if i not in train_label_indices]
        val_indices.extend(remaining_indices[:val_size // 2])

    # Simpler approach: just split the combined set
    X_val = X_train_combined[train_size:train_size + val_size]
    y_val = y_train_combined[train_size:train_size + val_size]

    # Create test subset
    X_test, y_test = utils.create_stratified_subset(X_test_raw, y_test_raw, test_size)

    # Pad sequences
    X_train = utils.pad_sequences_custom(X_train, maxlen=max_length)
    X_val = utils.pad_sequences_custom(X_val, maxlen=max_length)
    X_test = utils.pad_sequences_custom(X_test, maxlen=max_length)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_vocabulary_mappings(vocab_size: int = 10000) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create mappings between words and indices for the IMDB vocabulary."""
    # Get base word index from Keras
    word_index = utils.get_word_index()

    # Create word_to_idx (only words with index < vocab_size)
    word_to_idx = {}
    for word, idx in word_index.items():
        # Keras reserves 0, 1, 2, so actual indices are offset by 3
        adjusted_idx = idx + 3
        if adjusted_idx < vocab_size:
            word_to_idx[word] = adjusted_idx

    # Create idx_to_word
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    # Add special tokens
    idx_to_word[0] = '<PAD>'
    idx_to_word[1] = '<START>'
    idx_to_word[2] = '<UNK>'

    return word_to_idx, idx_to_word


# ============================================================================
# PART 2: BASELINE EMBEDDINGS (8 points)
# ============================================================================

def build_random_embedding_model(vocab_size: int, embedding_dim: int,
                                 max_length: int) -> keras.Model:
    """Build a sentiment classification model with random embeddings."""
    model = keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_pretrained_embedding_model(vocab_size: int, embedding_dim: int,
                                     max_length: int,
                                     pretrained_embeddings: np.ndarray) -> keras.Model:
    """Build a sentiment classification model with pretrained embeddings."""
    model = keras.Sequential([
        layers.Embedding(
            vocab_size,
            embedding_dim,
            input_length=max_length,
            weights=[pretrained_embeddings],
            trainable=True  # Can experiment with False
        ),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# ============================================================================
# PART 3: AUTOENCODER EMBEDDINGS (15 points)
# ============================================================================

def build_autoencoder_encoder(vocab_size: int, embedding_dim: int,
                               max_length: int) -> keras.Model:
    """Build the encoder part of an autoencoder for learning text embeddings."""
    inputs = layers.Input(shape=(vocab_size,))
    x = layers.Dense(512, activation='relu')(inputs)
    embeddings = layers.Dense(embedding_dim, activation='relu')(x)

    encoder = keras.Model(inputs, embeddings, name='encoder')
    return encoder


def build_autoencoder_decoder(vocab_size: int, embedding_dim: int) -> keras.Model:
    """Build the decoder part of an autoencoder for learning text embeddings."""
    inputs = layers.Input(shape=(embedding_dim,))
    x = layers.Dense(512, activation='relu')(inputs)
    outputs = layers.Dense(vocab_size, activation='sigmoid')(x)

    decoder = keras.Model(inputs, outputs, name='decoder')
    return decoder


def train_autoencoder(encoder: keras.Model, decoder: keras.Model,
                      X_train: np.ndarray, epochs: int = 20,
                      batch_size: int = 128) -> keras.callbacks.History:
    """Train the autoencoder (encoder + decoder) on bag-of-words data."""
    # Convert sequences to bag-of-words
    vocab_size = decoder.output_shape[-1]
    X_bow = utils.sequences_to_bow(X_train, vocab_size)

    # Create combined autoencoder model
    ae_input = layers.Input(shape=(vocab_size,))
    encoded = encoder(ae_input)
    decoded = decoder(encoded)
    autoencoder = keras.Model(ae_input, decoded, name='autoencoder')

    # Compile
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    # Train (input and target are the same)
    history = autoencoder.fit(
        X_bow, X_bow,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=0
    )

    return history


# ============================================================================
# PART 4: WORD2VEC-STYLE EMBEDDINGS (15 points)
# ============================================================================

def generate_skipgram_pairs(sequences: np.ndarray, window_size: int = 2,
                             vocab_size: int = 10000) -> np.ndarray:
    """Generate (target, context) pairs for skip-gram training."""
    pairs = []

    for sequence in sequences:
        # Filter out padding and out-of-vocab words
        valid_words = [w for w in sequence if 0 < w < vocab_size]

        for i, target in enumerate(valid_words):
            # Get context words within window
            start = max(0, i - window_size)
            end = min(len(valid_words), i + window_size + 1)

            for j in range(start, end):
                if j != i:  # Don't pair word with itself
                    context = valid_words[j]
                    pairs.append([target, context])

    return np.array(pairs)


def build_word2vec_model(vocab_size: int, embedding_dim: int) -> keras.Model:
    """Build a Word2Vec-style model using skip-gram architecture."""
    # Two inputs: target and context word indices
    target_input = layers.Input(shape=(1,), name='target')
    context_input = layers.Input(shape=(1,), name='context')

    # Shared embedding layer for both (we'll use target_embedding later)
    target_embedding = layers.Embedding(vocab_size, embedding_dim, input_length=1, name='target_embedding')
    context_embedding = layers.Embedding(vocab_size, embedding_dim, input_length=1, name='context_embedding')

    # Get embeddings
    target_vector = target_embedding(target_input)
    context_vector = context_embedding(context_input)

    # Flatten to (batch_size, embedding_dim)
    target_vector = layers.Flatten()(target_vector)
    context_vector = layers.Flatten()(context_vector)

    # Dot product
    dot_product = layers.Dot(axes=1)([target_vector, context_vector])

    # Sigmoid activation
    output = layers.Activation('sigmoid')(dot_product)

    # Create model
    model = keras.Model(inputs=[target_input, context_input], outputs=output, name='word2vec')

    # Compile
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def train_word2vec(model: keras.Model, pairs: np.ndarray, epochs: int = 10,
                   batch_size: int = 128, validation_split: float = 0.1) -> keras.callbacks.History:
    """Train the Word2Vec model on (target, context) pairs."""
    # Split pairs
    targets = pairs[:, 0]
    contexts = pairs[:, 1]

    # All labels are 1 (real pairs)
    labels = np.ones(len(pairs))

    # Train
    history = model.fit(
        [targets, contexts],
        labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=0
    )

    return history


# ============================================================================
# PART 5: SIMPLE ATTENTION MECHANISM (12 points)
# ============================================================================

class SimpleAttention(layers.Layer):
    """A simple self-attention layer for creating sentence embeddings."""

    def __init__(self, **kwargs):
        super(SimpleAttention, self).__init__(**kwargs)
        self.W = None

    def build(self, input_shape):
        """Create trainable weights for computing attention scores."""
        embedding_dim = input_shape[-1]

        self.W = self.add_weight(
            name='attention_weight',
            shape=(embedding_dim, 1),
            initializer='glorot_uniform',
            trainable=True
        )

        super(SimpleAttention, self).build(input_shape)

    def call(self, inputs, mask=None):
        """Apply attention to inputs."""
        # Compute attention scores
        # inputs: (batch_size, seq_length, embedding_dim)
        # W: (embedding_dim, 1)
        # scores: (batch_size, seq_length, 1)
        scores = tf.matmul(inputs, self.W)
        scores = tf.squeeze(scores, axis=-1)  # (batch_size, seq_length)

        # Apply mask if provided
        if mask is not None:
            # Convert mask to float and apply
            mask = tf.cast(mask, tf.float32)
            scores = scores * mask + (1 - mask) * (-1e9)

        # Apply softmax
        attention_weights = tf.nn.softmax(scores, axis=1)  # (batch_size, seq_length)

        # Apply attention weights to inputs
        attention_weights = tf.expand_dims(attention_weights, axis=-1)  # (batch_size, seq_length, 1)
        weighted_input = inputs * attention_weights  # (batch_size, seq_length, embedding_dim)

        # Sum across sequence
        output = tf.reduce_sum(weighted_input, axis=1)  # (batch_size, embedding_dim)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


# ============================================================================
# PART 6: SHARED CLASSIFIER & EVALUATION (10 points)
# ============================================================================

def build_sentiment_classifier(embedding_model, max_length: int,
                                model_type: str = 'standard') -> keras.Model:
    """Build a sentiment classifier using a given embedding approach."""

    if model_type == 'standard':
        # embedding_model is an Embedding layer
        inputs = layers.Input(shape=(max_length,))
        x = embedding_model(inputs)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(16, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        model = keras.Model(inputs, outputs)

    elif model_type == 'autoencoder':
        # embedding_model is the trained encoder
        vocab_size = embedding_model.input_shape[-1]
        inputs = layers.Input(shape=(max_length,))
        # Convert to BoW
        bow = layers.Lambda(lambda x: utils.sequences_to_bow(x, vocab_size))(inputs)
        x = embedding_model(bow)
        x = layers.Dense(16, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        model = keras.Model(inputs, outputs)

    elif model_type == 'word2vec':
        # Extract embeddings from Word2Vec model
        target_embedding_layer = None
        for layer in embedding_model.layers:
            if 'target_embedding' in layer.name:
                target_embedding_layer = layer
                break

        weights = target_embedding_layer.get_weights()
        vocab_size, embedding_dim = weights[0].shape

        inputs = layers.Input(shape=(max_length,))
        x = layers.Embedding(vocab_size, embedding_dim, weights=weights, input_length=max_length)(inputs)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(16, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        model = keras.Model(inputs, outputs)

    elif model_type == 'attention':
        # embedding_model is an Embedding layer
        inputs = layers.Input(shape=(max_length,))
        x = embedding_model(inputs)
        x = SimpleAttention()(x)
        x = layers.Dense(16, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        model = keras.Model(inputs, outputs)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def evaluate_all_embeddings(models_dict: Dict[str, keras.Model],
                             X_test: np.ndarray,
                             y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Evaluate multiple trained models on the test set."""
    results = {}

    for name, model in models_dict.items():
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        results[name] = {
            'loss': float(loss),
            'accuracy': float(accuracy)
        }

    return results


def extract_embeddings(model: keras.Model, words: List[str],
                       word_to_idx: Dict[str, int],
                       model_type: str = 'standard') -> np.ndarray:
    """Extract embedding vectors for specific words from a trained model."""
    embeddings = []

    if model_type in ['standard', 'attention']:
        # Find embedding layer
        embedding_layer = None
        for layer in model.layers:
            if isinstance(layer, layers.Embedding):
                embedding_layer = layer
                break

        if embedding_layer is None:
            raise ValueError("No Embedding layer found in model")

        embedding_weights = embedding_layer.get_weights()[0]

        for word in words:
            if word in word_to_idx:
                idx = word_to_idx[word]
                embeddings.append(embedding_weights[idx])
            else:
                # Unknown word - return zero vector
                embeddings.append(np.zeros(embedding_weights.shape[1]))

    elif model_type == 'word2vec':
        # Find target embedding layer
        target_embedding_layer = None
        for layer in model.layers:
            if 'target_embedding' in layer.name:
                target_embedding_layer = layer
                break

        if target_embedding_layer is None:
            raise ValueError("No target_embedding layer found in Word2Vec model")

        embedding_weights = target_embedding_layer.get_weights()[0]

        for word in words:
            if word in word_to_idx:
                idx = word_to_idx[word]
                embeddings.append(embedding_weights[idx])
            else:
                embeddings.append(np.zeros(embedding_weights.shape[1]))

    elif model_type == 'autoencoder':
        # Need to encode one-hot or BoW vectors
        # This is more complex - create BoW for single words
        vocab_size = model.input_shape[-1] if hasattr(model, 'input_shape') else 10000

        for word in words:
            if word in word_to_idx:
                idx = word_to_idx[word]
                # Create one-hot style BoW
                bow = np.zeros(vocab_size)
                bow[idx] = 1.0
                embedding = model.predict(bow.reshape(1, -1), verbose=0)[0]
                embeddings.append(embedding)
            else:
                # Unknown word
                embeddings.append(np.zeros(model.output_shape[-1]))

    return np.array(embeddings)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_simple_pretrained_embeddings(vocab_size: int, embedding_dim: int,
                                       seed: int = 42) -> np.ndarray:
    """Create simple pretrained embeddings for the baseline."""
    np.random.seed(seed)
    embeddings = np.random.uniform(-0.1, 0.1, (vocab_size, embedding_dim))
    embeddings[0] = 0  # Padding
    return embeddings.astype(np.float32)


if __name__ == '__main__':
    """Test the solution implementations."""
    print("PA5 Solution Code - Testing Implementations")
    print("=" * 60)

    # Test data loading
    print("\n1. Testing data loading...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_preprocess_imdb(
        vocab_size=1000,
        max_length=50,
        train_size=100,
        val_size=20,
        test_size=30
    )
    print(f"✓ Data loaded: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # Test vocabulary
    print("\n2. Testing vocabulary...")
    word_to_idx, idx_to_word = create_vocabulary_mappings(vocab_size=1000)
    print(f"✓ Vocabulary: {len(word_to_idx)} words")

    # Test random embedding model
    print("\n3. Testing random embedding model...")
    model = build_random_embedding_model(vocab_size=1000, embedding_dim=32, max_length=50)
    print(f"✓ Model built with {model.count_params()} parameters")

    # Test autoencoder
    print("\n4. Testing autoencoder...")
    encoder = build_autoencoder_encoder(vocab_size=1000, embedding_dim=32, max_length=50)
    decoder = build_autoencoder_decoder(vocab_size=1000, embedding_dim=32)
    history = train_autoencoder(encoder, decoder, X_train, epochs=2, batch_size=32)
    print(f"✓ Autoencoder trained, final loss: {history.history['loss'][-1]:.4f}")

    # Test skip-gram
    print("\n5. Testing Word2Vec...")
    pairs = generate_skipgram_pairs(X_train[:10], window_size=2, vocab_size=1000)
    print(f"✓ Generated {len(pairs)} skip-gram pairs")
    w2v_model = build_word2vec_model(vocab_size=1000, embedding_dim=32)
    print(f"✓ Word2Vec model built")

    # Test attention
    print("\n6. Testing SimpleAttention...")
    attention = SimpleAttention()
    test_input = tf.random.normal((2, 10, 32))
    attention.build(input_shape=(None, 10, 32))
    output = attention(test_input)
    print(f"✓ Attention output shape: {output.shape}")

    print("\n" + "=" * 60)
    print("All solution code tests passed!")
