# PA5: Text Representation Learning - A Comparative Study

**Due Date:** [To be announced]
**Total Points:** 100 (70 for Part 1, 30 for Part 2)

## Overview

In this capstone assignment, you will implement and compare different approaches to learning text representations for sentiment classification. You'll build four different embedding methods from the ground up and analyze how each captures semantic meaning in different ways.

This assignment ties together key themes from throughout the course: **representation learning** is the fundamental challenge underlying many machine learning tasks, from PCA's dimensionality reduction (PA1) to CNN filters (PA4) to the text embeddings you'll explore here.

## Learning Objectives

By completing this assignment, you will:

1. **Understand representation learning**: See how different architectures learn to represent the same data (text) in different ways
2. **Implement core NLP techniques**: Build Word2Vec, autoencoders, and attention mechanisms from scratch
3. **Compare and contrast approaches**: Develop intuition for when to use each embedding method
4. **Connect course concepts**: Recognize parallels between text embeddings, CNN filters, PCA, and autoencoder latent spaces
5. **Apply to real tasks**: Use learned representations for sentiment classification on IMDB reviews

## Assignment Structure

This is a **two-part assignment** following our standard structure:

### Part 1: Implementation (70 points) - Week 1
Implement four text embedding approaches using TensorFlow/Keras:
- Baseline embeddings (random and pretrained)
- Autoencoder-based embeddings
- Word2Vec-style embeddings
- Attention-based embeddings

### Part 2: Visualization & Analysis (30 points) - Week 2
Analyze and compare your implementations:
- Visualize embedding spaces using PCA
- Investigate semantic relationships
- Analyze attention patterns
- Compare classification performance
- Write executive summary connecting to course themes

## Dataset

You will work with the **IMDB Movie Review Dataset**:
- **Task**: Binary sentiment classification (positive/negative)
- **Size**: 10,000 training, 2,500 validation, 5,000 test samples (CPU-friendly subset)
- **Vocabulary**: 10,000 most frequent words
- **Sequence length**: 200 tokens (padded/truncated)

This is a new dataset for our course, representing the natural progression from images (MNIST in PA4) → time series (RNN demos) → text sequences.

## Files Provided

- `student_code.py` - Function stubs for your implementations (**you will edit this**)
- `utils.py` - Helper functions for data loading (provided, do not edit)
- `tests.py` - Comprehensive test suite to validate your work
- `visualization_analysis.ipynb` - Structured notebook for Part 2 analysis
- `requirements.txt` - Required dependencies
- `README.md` - This file
- `peer_review_rubric.md` - Rubric for peer review (30 points)

## Part 1: Implementation Details (70 points)

### 1. Data Preprocessing (10 points)

**`load_and_preprocess_imdb()`** - 5 points
- Load IMDB data, create stratified subsets, pad sequences
- Maintain class balance across train/val/test splits

**`create_vocabulary_mappings()`** - 5 points
- Build word↔index dictionaries
- Handle special tokens (PAD, START, UNK)

### 2. Baseline Embeddings (8 points)

**`build_random_embedding_model()`** - 4 points
- Simple model with randomly initialized trainable embeddings
- Provides baseline for comparison

**`build_pretrained_embedding_model()`** - 4 points
- Uses pretrained embedding weights
- Explore trainable vs. frozen embeddings

### 3. Autoencoder Embeddings (15 points)

**`build_autoencoder_encoder()`** - 7 points
- Encoder: bag-of-words → embedding (bottleneck)
- Learns compressed representation

**`build_autoencoder_decoder()`** - 6 points
- Decoder: embedding → reconstructed bag-of-words
- Symmetric to encoder

**`train_autoencoder()`** - 2 points
- Train full autoencoder with reconstruction loss
- Similar concept to PCA from PA1!

### 4. Word2Vec-Style Embeddings (15 points)

**`generate_skipgram_pairs()`** - 6 points
- Create (target, context) word pairs
- Window-based context extraction

**`build_word2vec_model()`** - 5 points
- Skip-gram architecture with target/context embeddings
- Learns from word co-occurrence

**`train_word2vec()`** - 4 points
- Train on skip-gram pairs
- Simplified version (positive pairs only)

### 5. Simple Attention Mechanism (12 points)

**`SimpleAttention` custom Keras layer** - 12 points
- Compute attention weights for each word
- Return weighted sum (sentence embedding)
- Single attention head (simplified, but challenging!)

**Note**: This is the most complex component but worth fewer points. We expect it to be challenging - do your best and ask questions!

### 6. Shared Classifier & Evaluation (10 points)

**`build_sentiment_classifier()`** - 4 points
- Takes any embedding approach and adds classification head
- Flexible design to handle different embedding types

**`evaluate_all_embeddings()`** - 3 points
- Compare performance across all methods
- Return structured metrics

**`extract_embeddings()`** - 3 points
- Extract embedding vectors for specific words
- Enables visualization and analysis

## Part 2: Analysis Details (30 points)

Work in `visualization_analysis.ipynb` to complete the following analyses:

### 1. Embedding Space Visualization (8 points)
- Extract embeddings for semantically meaningful words
- Use PCA to project to 2D (connects to PA1!)
- Compare embedding spaces across methods
- Identify which captures semantic structure best

### 2. Semantic Relationships (7 points)
- Find nearest neighbors in embedding space
- Analyze similarity matrices
- Evaluate semantic quality of each method

### 3. Attention Pattern Investigation (6 points)
- Visualize attention weights for sample reviews
- Determine if attention focuses on sentiment words
- Compare to simple averaging baseline

### 4. Classification Performance Analysis (5 points)
- Evaluate all models on test set
- Compare training dynamics
- Perform error analysis

### 5. Cross-Course Connections - Executive Summary (4 points)
- Synthesize findings for peer audience
- Connect to PCA (PA1), CNN filters (PA4), other course concepts
- Discuss practical implications

## Getting Started

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Or with conda:
conda install tensorflow numpy matplotlib scikit-learn
```

### 2. Verify Setup

```bash
# Run the student_code.py main block
python student_code.py

# Should see test output (many will fail initially)
```

### 3. Development Workflow

**Week 1 (Part 1):**
1. Implement functions in `student_code.py` one at a time
2. Run tests frequently: `python tests.py`
3. Start with data preprocessing, then baselines, then advanced methods
4. Use the test suite to validate correctness
5. Train small models first (small vocab, few epochs) to verify functionality

**Week 2 (Part 2):**
1. Open `visualization_analysis.ipynb` in Jupyter
2. Train all your models on the full dataset
3. Complete each analysis section with code and written responses
4. Create clear visualizations (matplotlib only!)
5. Write your executive summary
6. Prepare questions for peer reviewers

## Testing Your Code

```bash
# Run all tests
python tests.py

# Run specific test class
python -m unittest tests.TestDataPreprocessing

# Run with verbose output
python tests.py -v
```

**Expected test results:**
- All tests should pass before submitting Part 1
- Tests verify correctness, not performance
- Focus on passing tests for basic functionality first

## Important Notes

### Technical Requirements

- **Use TensorFlow/Keras** (not PyTorch) for consistency with PA4
- **Use matplotlib only** for visualizations (no seaborn)
- **Follow provided function signatures** - tests depend on them
- **CPU-friendly**: All training should complete in 60-90 minutes total
- **Reproducibility**: Use provided random seeds

### Grading Philosophy

**Part 1 (Implementation):**
- Graded on correctness and passing tests
- Point values emphasize foundational components
- SimpleAttention is challenging but lower weight

**Part 2 (Analysis):**
- Graded on **reasoning quality**, NOT specific numerical results
- We don't expect specific accuracy values or predetermined conclusions
- Focus on:
  - Clear, evidence-based analysis
  - Thoughtful interpretation of results
  - Connections to course concepts
  - Quality of visualizations and communication

### Common Pitfalls to Avoid

1. **Vocabulary mismatches**: Ensure consistent vocab_size across all functions
2. **Shape mismatches**: Pay attention to input/output shapes for each model type
3. **Padding tokens**: Remember to skip or mask padding (index 0)
4. **Memory issues**: Use the specified subset sizes, not full IMDB dataset
5. **Over-engineering**: Keep SimpleAttention simple - single head is enough!

## Conceptual Framework

This assignment explores a **key insight**: Most deep learning methods are fundamentally about learning good representations (embeddings) of data in an N-dimensional latent space.

Consider the parallels:
- **PCA (PA1)**: Learns linear projection to capture variance
- **Autoencoder**: Learns nonlinear compression through reconstruction
- **CNN filters (PA4)**: Learn spatial features that represent visual patterns
- **Word embeddings**: Learn semantic features that represent word meaning
- **Attention**: Learns which parts of input matter for the representation

All are solving the same problem: **How do we represent complex data in a way that captures its essential structure?**

Your analysis should explore these connections and develop intuition for when each approach works best.

## Submission Instructions

Submit the following files to Canvas:

**Part 1 (Week 1):**
- `student_code.py` (your implementations)
- Test output showing all tests pass

**Part 2 (Week 2):**
- `visualization_analysis.ipynb` (completed notebook)
- PDF export of notebook (for peer review)

**Peer Review:**
- Will be conducted using `peer_review_rubric.md`
- Worth 30 points (separate from the 100 points above)

## Getting Help

- **Office hours**: [Times TBD]
- **Discussion board**: For conceptual questions and clarifications
- **Test failures**: Include full error message when asking for help
- **Start early**: This is a substantial assignment - don't wait until the deadline!

## Resources

### Recommended Reading
- Course notes on embeddings and attention
- [TensorFlow Keras Layer Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer)
- [IMDB Dataset Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb)

### Debugging Tips
- Print shapes frequently: `print(tensor.shape)`
- Use small data for initial testing (vocab_size=100, etc.)
- Visualize embeddings early to catch issues
- Test one component at a time

## Academic Integrity

- You may discuss concepts with classmates
- You may not share code or copy implementations
- All code must be your own work
- Cite any external resources used (beyond course materials)

---

**Questions?** Post on the discussion board or come to office hours. This is a challenging but rewarding assignment - start early and ask for help when needed!

Good luck, and enjoy exploring the fascinating world of representation learning!
