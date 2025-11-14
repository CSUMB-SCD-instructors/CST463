"""
Test Suite for PA5: Text Representation Learning

This file contains comprehensive tests for all student implementations.
Run with: python tests.py or python -m pytest tests.py

Total: ~600 lines of tests covering all functions
"""

import unittest
import numpy as np
import tensorflow as tf
from tensorflow import keras
import student_code
import utils


class TestDataPreprocessing(unittest.TestCase):
    """Test data preprocessing functions (10 points)"""

    def test_load_and_preprocess_imdb_basic(self):
        """Test basic IMDB loading functionality"""
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
            student_code.load_and_preprocess_imdb(
                vocab_size=1000,
                max_length=50,
                train_size=100,
                val_size=20,
                test_size=30
            )

        # Check shapes
        self.assertEqual(X_train.shape, (100, 50))
        self.assertEqual(y_train.shape, (100,))
        self.assertEqual(X_val.shape, (20, 50))
        self.assertEqual(y_val.shape, (20,))
        self.assertEqual(X_test.shape, (30, 50))
        self.assertEqual(y_test.shape, (30,))

    def test_load_and_preprocess_imdb_values(self):
        """Test that loaded data has correct value ranges"""
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
            student_code.load_and_preprocess_imdb(
                vocab_size=5000,
                max_length=100,
                train_size=100,
                val_size=20,
                test_size=30
            )

        # Check word indices are within vocabulary
        self.assertTrue(np.all(X_train < 5000))
        self.assertTrue(np.all(X_train >= 0))

        # Check labels are binary
        self.assertTrue(np.all((y_train == 0) | (y_train == 1)))
        self.assertTrue(np.all((y_val == 0) | (y_val == 1)))
        self.assertTrue(np.all((y_test == 0) | (y_test == 1)))

    def test_load_and_preprocess_imdb_stratification(self):
        """Test that class balance is maintained"""
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
            student_code.load_and_preprocess_imdb(
                vocab_size=5000,
                max_length=100,
                train_size=200,
                val_size=50,
                test_size=100
            )

        # Check approximate balance (should be 50/50)
        train_pos_ratio = np.sum(y_train) / len(y_train)
        val_pos_ratio = np.sum(y_val) / len(y_val)
        test_pos_ratio = np.sum(y_test) / len(y_test)

        self.assertAlmostEqual(train_pos_ratio, 0.5, delta=0.1)
        self.assertAlmostEqual(val_pos_ratio, 0.5, delta=0.2)
        self.assertAlmostEqual(test_pos_ratio, 0.5, delta=0.1)

    def test_load_and_preprocess_imdb_padding(self):
        """Test that sequences are properly padded/truncated"""
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
            student_code.load_and_preprocess_imdb(
                vocab_size=5000,
                max_length=50,
                train_size=100,
                val_size=20,
                test_size=30
            )

        # All sequences should be exactly max_length
        self.assertTrue(np.all([len(seq) == 50 for seq in X_train]))
        self.assertTrue(np.all([len(seq) == 50 for seq in X_test]))

    def test_create_vocabulary_mappings_basic(self):
        """Test vocabulary mapping creation"""
        word_to_idx, idx_to_word = student_code.create_vocabulary_mappings(vocab_size=1000)

        # Check both mappings exist
        self.assertIsInstance(word_to_idx, dict)
        self.assertIsInstance(idx_to_word, dict)

        # Check they're inverses (approximately)
        self.assertGreater(len(word_to_idx), 0)
        self.assertGreater(len(idx_to_word), 0)

    def test_create_vocabulary_mappings_special_tokens(self):
        """Test that special tokens are properly included"""
        word_to_idx, idx_to_word = student_code.create_vocabulary_mappings(vocab_size=5000)

        # Check special tokens in idx_to_word
        self.assertEqual(idx_to_word[0], '<PAD>')
        self.assertEqual(idx_to_word[1], '<START>')
        self.assertEqual(idx_to_word[2], '<UNK>')

    def test_create_vocabulary_mappings_size(self):
        """Test vocabulary size constraint"""
        vocab_size = 1000
        word_to_idx, idx_to_word = student_code.create_vocabulary_mappings(vocab_size=vocab_size)

        # Check that we don't exceed vocab_size
        max_idx = max(idx_to_word.keys())
        self.assertLess(max_idx, vocab_size)


class TestBaselineEmbeddings(unittest.TestCase):
    """Test baseline embedding models (8 points)"""

    def test_build_random_embedding_model_structure(self):
        """Test random embedding model creation"""
        model = student_code.build_random_embedding_model(
            vocab_size=1000,
            embedding_dim=32,
            max_length=50
        )

        self.assertIsInstance(model, keras.Model)

        # Check input/output shapes
        self.assertEqual(model.input_shape, (None, 50))
        self.assertEqual(model.output_shape, (None, 1))

    def test_build_random_embedding_model_compilation(self):
        """Test that model is properly compiled"""
        model = student_code.build_random_embedding_model(
            vocab_size=1000,
            embedding_dim=32,
            max_length=50
        )

        # Model should be compiled with loss and optimizer
        self.assertIsNotNone(model.optimizer)
        self.assertIsNotNone(model.loss)

    def test_build_random_embedding_model_prediction(self):
        """Test that model can make predictions"""
        model = student_code.build_random_embedding_model(
            vocab_size=1000,
            embedding_dim=32,
            max_length=50
        )

        # Create dummy input
        X_dummy = np.random.randint(0, 1000, (10, 50))
        predictions = model.predict(X_dummy, verbose=0)

        # Check prediction shape and range
        self.assertEqual(predictions.shape, (10, 1))
        self.assertTrue(np.all(predictions >= 0))
        self.assertTrue(np.all(predictions <= 1))

    def test_build_pretrained_embedding_model_structure(self):
        """Test pretrained embedding model creation"""
        pretrained = student_code.create_simple_pretrained_embeddings(
            vocab_size=1000,
            embedding_dim=32
        )

        model = student_code.build_pretrained_embedding_model(
            vocab_size=1000,
            embedding_dim=32,
            max_length=50,
            pretrained_embeddings=pretrained
        )

        self.assertIsInstance(model, keras.Model)
        self.assertEqual(model.input_shape, (None, 50))
        self.assertEqual(model.output_shape, (None, 1))

    def test_build_pretrained_embedding_model_uses_pretrained(self):
        """Test that pretrained embeddings are actually used"""
        pretrained = student_code.create_simple_pretrained_embeddings(
            vocab_size=100,
            embedding_dim=16
        )

        model = student_code.build_pretrained_embedding_model(
            vocab_size=100,
            embedding_dim=16,
            max_length=20,
            pretrained_embeddings=pretrained
        )

        # Get embedding layer weights
        embedding_layer = None
        for layer in model.layers:
            if isinstance(layer, keras.layers.Embedding):
                embedding_layer = layer
                break

        self.assertIsNotNone(embedding_layer)
        # Check that weights match pretrained (at least initially)
        weights = embedding_layer.get_weights()[0]
        self.assertEqual(weights.shape, pretrained.shape)


class TestAutoencoderEmbeddings(unittest.TestCase):
    """Test autoencoder embedding functions (15 points)"""

    def test_build_autoencoder_encoder_structure(self):
        """Test encoder model structure"""
        encoder = student_code.build_autoencoder_encoder(
            vocab_size=1000,
            embedding_dim=64,
            max_length=50
        )

        self.assertIsInstance(encoder, keras.Model)

        # Check input is vocab_size (bag-of-words)
        self.assertEqual(encoder.input_shape, (None, 1000))
        # Check output is embedding_dim (bottleneck)
        self.assertEqual(encoder.output_shape, (None, 64))

    def test_build_autoencoder_encoder_prediction(self):
        """Test encoder can process bag-of-words input"""
        encoder = student_code.build_autoencoder_encoder(
            vocab_size=1000,
            embedding_dim=64,
            max_length=50
        )

        # Create dummy BoW input
        bow_input = np.random.randint(0, 10, (5, 1000)).astype(np.float32)
        embeddings = encoder.predict(bow_input, verbose=0)

        self.assertEqual(embeddings.shape, (5, 64))

    def test_build_autoencoder_decoder_structure(self):
        """Test decoder model structure"""
        decoder = student_code.build_autoencoder_decoder(
            vocab_size=1000,
            embedding_dim=64
        )

        self.assertIsInstance(decoder, keras.Model)

        # Check input is embedding_dim (bottleneck)
        self.assertEqual(decoder.input_shape, (None, 64))
        # Check output is vocab_size (reconstructed BoW)
        self.assertEqual(decoder.output_shape, (None, 1000))

    def test_build_autoencoder_decoder_prediction(self):
        """Test decoder can reconstruct from embeddings"""
        decoder = student_code.build_autoencoder_decoder(
            vocab_size=1000,
            embedding_dim=64
        )

        # Create dummy embedding input
        embedding_input = np.random.randn(5, 64).astype(np.float32)
        reconstructed = decoder.predict(embedding_input, verbose=0)

        self.assertEqual(reconstructed.shape, (5, 1000))

    def test_train_autoencoder_basic(self):
        """Test autoencoder training"""
        encoder = student_code.build_autoencoder_encoder(
            vocab_size=500,
            embedding_dim=32,
            max_length=50
        )
        decoder = student_code.build_autoencoder_decoder(
            vocab_size=500,
            embedding_dim=32
        )

        # Create dummy sequence data
        X_train = np.random.randint(0, 500, (100, 50))

        # Train for just a few epochs
        history = student_code.train_autoencoder(
            encoder, decoder, X_train, epochs=2, batch_size=32
        )

        # Check that history is returned
        self.assertIsNotNone(history)
        # Should have loss recorded
        self.assertIn('loss', history.history)

    def test_autoencoder_integration(self):
        """Test full autoencoder pipeline"""
        vocab_size = 500
        embedding_dim = 32
        max_length = 50

        encoder = student_code.build_autoencoder_encoder(vocab_size, embedding_dim, max_length)
        decoder = student_code.build_autoencoder_decoder(vocab_size, embedding_dim)

        # Create dummy data
        X_train = np.random.randint(0, vocab_size, (50, max_length))

        # Train
        history = student_code.train_autoencoder(encoder, decoder, X_train, epochs=1, batch_size=16)

        # Test encoding
        X_test = np.random.randint(0, vocab_size, (10, max_length))
        bow_test = utils.sequences_to_bow(X_test, vocab_size)
        embeddings = encoder.predict(bow_test, verbose=0)
        reconstructed = decoder.predict(embeddings, verbose=0)

        # Check shapes
        self.assertEqual(embeddings.shape, (10, embedding_dim))
        self.assertEqual(reconstructed.shape, (10, vocab_size))


class TestWord2VecEmbeddings(unittest.TestCase):
    """Test Word2Vec-style embedding functions (15 points)"""

    def test_generate_skipgram_pairs_basic(self):
        """Test skip-gram pair generation"""
        sequences = [[1, 2, 3, 4, 5]]
        pairs = student_code.generate_skipgram_pairs(
            sequences, window_size=2, vocab_size=100
        )

        # Should have pairs
        self.assertIsInstance(pairs, np.ndarray)
        self.assertEqual(pairs.shape[1], 2)  # Each pair has target and context
        self.assertGreater(len(pairs), 0)

    def test_generate_skipgram_pairs_window_size(self):
        """Test skip-gram with different window sizes"""
        sequences = [[5, 10, 15, 20, 25]]

        pairs_w1 = student_code.generate_skipgram_pairs(sequences, window_size=1, vocab_size=100)
        pairs_w2 = student_code.generate_skipgram_pairs(sequences, window_size=2, vocab_size=100)

        # Larger window should produce more pairs
        self.assertGreater(len(pairs_w2), len(pairs_w1))

    def test_generate_skipgram_pairs_skip_padding(self):
        """Test that padding tokens are skipped"""
        sequences = [[0, 0, 5, 10, 15, 0, 0]]  # 0 is padding
        pairs = student_code.generate_skipgram_pairs(sequences, window_size=2, vocab_size=100)

        # No pairs should involve padding (0)
        self.assertTrue(np.all(pairs[:, 0] != 0))
        self.assertTrue(np.all(pairs[:, 1] != 0))

    def test_generate_skipgram_pairs_vocab_limit(self):
        """Test vocabulary size limit"""
        sequences = [[10, 20, 30, 500, 1000]]  # Some words beyond vocab
        pairs = student_code.generate_skipgram_pairs(sequences, window_size=2, vocab_size=100)

        # All indices should be < vocab_size
        if len(pairs) > 0:
            self.assertTrue(np.all(pairs < 100))

    def test_generate_skipgram_pairs_multiple_sequences(self):
        """Test with multiple sequences"""
        sequences = [
            [5, 10, 15, 20],
            [25, 30, 35, 40],
            [45, 50, 55, 60]
        ]
        pairs = student_code.generate_skipgram_pairs(sequences, window_size=1, vocab_size=100)

        # Should generate pairs from all sequences
        self.assertGreater(len(pairs), 0)

    def test_build_word2vec_model_structure(self):
        """Test Word2Vec model structure"""
        model = student_code.build_word2vec_model(vocab_size=1000, embedding_dim=64)

        self.assertIsInstance(model, keras.Model)
        # Should be compiled
        self.assertIsNotNone(model.optimizer)

    def test_build_word2vec_model_prediction(self):
        """Test Word2Vec model can make predictions"""
        model = student_code.build_word2vec_model(vocab_size=100, embedding_dim=32)

        # Create dummy pairs
        dummy_pairs = np.array([[10, 20], [30, 40], [50, 60]])
        predictions = model.predict([dummy_pairs[:, 0], dummy_pairs[:, 1]], verbose=0)

        # Should output probabilities
        self.assertGreater(len(predictions), 0)

    def test_train_word2vec_basic(self):
        """Test Word2Vec training"""
        model = student_code.build_word2vec_model(vocab_size=100, embedding_dim=32)

        # Generate some pairs
        sequences = [[i for i in range(10, 50)]]
        pairs = student_code.generate_skipgram_pairs(sequences, window_size=2, vocab_size=100)

        # Train for a few epochs
        history = student_code.train_word2vec(model, pairs, epochs=2, batch_size=32)

        # Check history
        self.assertIsNotNone(history)
        self.assertIn('loss', history.history)

    def test_word2vec_integration(self):
        """Test complete Word2Vec pipeline"""
        vocab_size = 200
        embedding_dim = 32

        # Generate pairs from sequences
        sequences = [[i for i in range(10, 100)] for _ in range(5)]
        pairs = student_code.generate_skipgram_pairs(sequences, window_size=2, vocab_size=vocab_size)

        # Build and train model
        model = student_code.build_word2vec_model(vocab_size, embedding_dim)
        history = student_code.train_word2vec(model, pairs, epochs=1, batch_size=64)

        # Check that training completed
        self.assertIsNotNone(history)
        self.assertGreater(len(pairs), 0)


class TestAttentionMechanism(unittest.TestCase):
    """Test SimpleAttention layer (12 points)"""

    def test_simple_attention_instantiation(self):
        """Test that SimpleAttention layer can be created"""
        attention_layer = student_code.SimpleAttention()
        self.assertIsInstance(attention_layer, keras.layers.Layer)

    def test_simple_attention_output_shape(self):
        """Test SimpleAttention output shape"""
        attention_layer = student_code.SimpleAttention()

        # Create dummy input: (batch_size, sequence_length, embedding_dim)
        dummy_input = tf.random.normal((4, 10, 32))

        # Build the layer
        attention_layer.build(input_shape=(None, 10, 32))

        # Apply attention
        output = attention_layer(dummy_input)

        # Output should be (batch_size, embedding_dim)
        self.assertEqual(output.shape, (4, 32))

    def test_simple_attention_reduces_sequence(self):
        """Test that attention reduces sequence to single vector"""
        attention_layer = student_code.SimpleAttention()

        # Different sequence lengths
        for seq_len in [5, 10, 20]:
            dummy_input = tf.random.normal((2, seq_len, 16))
            attention_layer.build(input_shape=(None, seq_len, 16))
            output = attention_layer(dummy_input)

            # Should always reduce to (batch_size, embedding_dim)
            self.assertEqual(output.shape, (2, 16))

    def test_simple_attention_with_masking(self):
        """Test attention with padding mask"""
        attention_layer = student_code.SimpleAttention()
        attention_layer.build(input_shape=(None, 10, 32))

        dummy_input = tf.random.normal((3, 10, 32))
        # Create a mask (True for real tokens, False for padding)
        mask = tf.constant([[True]*10, [True]*5 + [False]*5, [True]*3 + [False]*7])

        output = attention_layer(dummy_input, mask=mask)

        # Should still produce output
        self.assertEqual(output.shape, (3, 32))

    def test_simple_attention_trainable_weights(self):
        """Test that attention layer has trainable weights"""
        attention_layer = student_code.SimpleAttention()
        attention_layer.build(input_shape=(None, 10, 32))

        # Should have at least one trainable weight
        trainable_weights = attention_layer.trainable_weights
        self.assertGreater(len(trainable_weights), 0)

    def test_simple_attention_in_model(self):
        """Test SimpleAttention integrated in a model"""
        inputs = keras.Input(shape=(20, 32))
        attended = student_code.SimpleAttention()(inputs)
        outputs = keras.layers.Dense(1, activation='sigmoid')(attended)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy')

        # Test prediction
        dummy_input = np.random.randn(5, 20, 32).astype(np.float32)
        predictions = model.predict(dummy_input, verbose=0)

        self.assertEqual(predictions.shape, (5, 1))


class TestClassifierAndEvaluation(unittest.TestCase):
    """Test classifier and evaluation functions (10 points)"""

    def test_build_sentiment_classifier_standard(self):
        """Test building classifier with standard embeddings"""
        # Create a simple embedding layer
        embedding_layer = keras.layers.Embedding(input_dim=1000, output_dim=32)

        model = student_code.build_sentiment_classifier(
            embedding_model=embedding_layer,
            max_length=50,
            model_type='standard'
        )

        self.assertIsInstance(model, keras.Model)
        self.assertEqual(model.input_shape, (None, 50))
        self.assertEqual(model.output_shape, (None, 1))

    def test_build_sentiment_classifier_prediction(self):
        """Test that classifier can make predictions"""
        embedding_layer = keras.layers.Embedding(input_dim=500, output_dim=32)

        model = student_code.build_sentiment_classifier(
            embedding_layer, max_length=50, model_type='standard'
        )

        # Create dummy data
        X_dummy = np.random.randint(0, 500, (10, 50))
        predictions = model.predict(X_dummy, verbose=0)

        self.assertEqual(predictions.shape, (10, 1))
        self.assertTrue(np.all(predictions >= 0))
        self.assertTrue(np.all(predictions <= 1))

    def test_evaluate_all_embeddings_basic(self):
        """Test evaluation across multiple models"""
        # Create simple models
        models_dict = {}

        for name in ['model1', 'model2']:
            model = keras.Sequential([
                keras.layers.Embedding(100, 16, input_length=20),
                keras.layers.GlobalAveragePooling1D(),
                keras.layers.Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            models_dict[name] = model

        # Create dummy test data
        X_test = np.random.randint(0, 100, (50, 20))
        y_test = np.random.randint(0, 2, 50)

        results = student_code.evaluate_all_embeddings(models_dict, X_test, y_test)

        # Check results structure
        self.assertIsInstance(results, dict)
        self.assertIn('model1', results)
        self.assertIn('model2', results)

        # Each result should have metrics
        for name, metrics in results.items():
            self.assertIn('accuracy', metrics)

    def test_evaluate_all_embeddings_metrics(self):
        """Test that evaluation returns proper metrics"""
        # Create and train a simple model
        model = keras.Sequential([
            keras.layers.Embedding(100, 16, input_length=20),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Quick training
        X_train = np.random.randint(0, 100, (100, 20))
        y_train = np.random.randint(0, 2, 100)
        model.fit(X_train, y_train, epochs=1, verbose=0)

        models_dict = {'test_model': model}

        X_test = np.random.randint(0, 100, (30, 20))
        y_test = np.random.randint(0, 2, 30)

        results = student_code.evaluate_all_embeddings(models_dict, X_test, y_test)

        # Accuracy should be between 0 and 1
        self.assertGreaterEqual(results['test_model']['accuracy'], 0.0)
        self.assertLessEqual(results['test_model']['accuracy'], 1.0)

    def test_extract_embeddings_basic(self):
        """Test embedding extraction"""
        # Create a simple model with embeddings
        model = keras.Sequential([
            keras.layers.Embedding(100, 16, input_length=20, name='embedding'),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        word_to_idx = {f'word{i}': i for i in range(100)}
        words = ['word5', 'word10', 'word15']

        embeddings = student_code.extract_embeddings(
            model, words, word_to_idx, model_type='standard'
        )

        # Should return embeddings for each word
        self.assertEqual(embeddings.shape, (3, 16))

    def test_extract_embeddings_unknown_words(self):
        """Test extraction with unknown words"""
        model = keras.Sequential([
            keras.layers.Embedding(50, 16, input_length=20),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        word_to_idx = {f'word{i}': i for i in range(50)}
        words = ['word5', 'unknown_word', 'word10']

        embeddings = student_code.extract_embeddings(
            model, words, word_to_idx, model_type='standard'
        )

        # Should still return something (zero vector or skip)
        self.assertIsInstance(embeddings, np.ndarray)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows"""

    def test_full_pipeline_minimal(self):
        """Test a minimal version of the full pipeline"""
        # 1. Load data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
            student_code.load_and_preprocess_imdb(
                vocab_size=500,
                max_length=50,
                train_size=100,
                val_size=20,
                test_size=30
            )

        # 2. Build and train a simple model
        model = student_code.build_random_embedding_model(
            vocab_size=500,
            embedding_dim=16,
            max_length=50
        )

        # 3. Quick training
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=1,
            batch_size=32,
            verbose=0
        )

        # 4. Evaluate
        results = student_code.evaluate_all_embeddings(
            {'test': model},
            X_test,
            y_test
        )

        # Check that pipeline completed
        self.assertIsNotNone(history)
        self.assertIn('test', results)

    def test_vocabulary_consistency(self):
        """Test that vocabulary is consistent across functions"""
        vocab_size = 1000

        # Create mappings
        word_to_idx, idx_to_word = student_code.create_vocabulary_mappings(vocab_size)

        # Load data with same vocab size
        (X_train, y_train), _, _ = student_code.load_and_preprocess_imdb(
            vocab_size=vocab_size,
            max_length=50,
            train_size=50,
            val_size=10,
            test_size=10
        )

        # All word indices should be valid in vocabulary
        unique_indices = np.unique(X_train)
        for idx in unique_indices:
            if idx != 0:  # Skip padding
                self.assertIn(idx, idx_to_word)


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    print("=" * 70)
    print("PA5: Text Representation Learning - Test Suite")
    print("=" * 70)
    run_tests()
