# PA4: Convolutional Neural Networks & Training Dynamics

This assignment provides hands-on experience with CNNs, training callbacks, hyperparameter tuning, and overfitting analysis using TensorFlow/Keras on the MNIST dataset.

## Assignment Components

### 1. CNN Implementation (`student_code.py`)
- **Model Architectures**: Build sequential and functional CNNs with different design patterns
- **Custom Callbacks**: Implement early stopping and learning rate scheduling from scratch
- **Hyperparameter Search**: Conduct systematic exploration of optimizers, learning rates, and batch sizes
- **Training Pipeline**: Complete training functions with proper configuration management

### 2. Training Dynamics Analysis (`visualization_analysis.ipynb`)
- **Architecture Comparison**: Analyze sequential vs. functional model designs
- **Optimizer Evaluation**: Compare convergence behavior across different optimizers
- **Grid Search Analysis**: Systematically explore hyperparameter space
- **Overfitting Study**: Observe, induce, and mitigate overfitting through controlled experiments
- **Callback Effectiveness**: Evaluate early stopping and learning rate scheduling impact
- **Professional Communication**: Present findings with clear matplotlib visualizations

### 3. Peer Review Component
- Export notebook as PDF for peer evaluation
- Focus on experimental design and interpretation of training dynamics
- Provide actionable insights for CNN development

## Learning Objectives
- **CNN Architecture**: Understand convolutional layers, pooling, padding, and stride
- **Functional API**: Learn when and why to use Functional API vs. Sequential
- **Training Callbacks**: Implement mechanisms to control and monitor training
- **Hyperparameter Tuning**: Conduct systematic search and evaluation
- **Overfitting Diagnosis**: Identify, understand, and mitigate overfitting
- **Training Best Practices**: Develop intuition for effective CNN training

## Dataset
MNIST handwritten digit classification:
- Training: 10,000 samples (subset for CPU feasibility)
- Validation: 2,000 samples
- Test: 10,000 samples (full test set)
- Input: 28×28 grayscale images
- Output: 10 classes (digits 0-9)

## Grading Breakdown
- **Technical Implementation (70%)**: `student_code.py` - Model building, callbacks, training pipeline, test suite performance
- **Analysis & Communication (30%)**: `visualization_analysis.ipynb` - Experimental design, overfitting analysis, hyperparameter insights, peer review preparation

## Technical Requirements
- All visualizations must use **matplotlib.pyplot only** - NO seaborn
- Use provided RNG seeds for reproducibility
- CPU-feasible design (30-60 minute total runtime expected)
- Complete both code implementation and experimental analysis
- Test your implementation using the provided test suite

## Getting Started

### 1. Implementation Phase
```bash
# Test your implementations
python -m pytest tests.py -v

# Check specific function
python -m pytest tests.py::TestModelBuilding::test_sequential_model_structure -v
```

### 2. Analysis Phase
Open `visualization_analysis.ipynb` and work through the analysis sections. You'll use helper functions but make choices about:
- Which model architectures to compare and why
- What hyperparameter ranges to explore
- How to induce and mitigate overfitting
- Which callback parameters work best
- How to interpret and present your findings

### 3. Key Implementation Functions
- `build_sequential_cnn()` - Standard CNN with sequential architecture
- `build_functional_inception_cnn()` - CNN with mini-inception module (parallel convolution paths)
- `EarlyStoppingCallback` - Custom callback to stop training when validation performance plateaus
- `LearningRateSchedulerCallback` - Custom callback to adjust learning rate during training
- `train_model_with_config()` - Train with specific hyperparameter configuration
- `run_grid_search()` - Systematic hyperparameter exploration

## Tips for Success

### Implementation Strategy
1. **Start with sequential model** - Get basic CNN working first
2. **Understand functional API** - Build inception module step by step
3. **Test callbacks independently** - Verify early stopping and LR scheduling work correctly
4. **Use small datasets for debugging** - Test with 100 samples before full training
5. **Run tests frequently** - Catch errors early

### Analysis Strategy
1. **Start with baseline** - Get a working model before experimenting
2. **Change one variable at a time** - Isolate effects of different choices
3. **Use matplotlib effectively** - Create clear, well-labeled plots
4. **Connect theory to observations** - Explain why certain behaviors occur
5. **Focus on insights** - What would help someone building CNNs in practice?

## Common Challenges
- **Shape mismatches**: Pay attention to Conv2D output shapes with different padding/stride
- **Callback logic**: Ensure patience mechanism and weight restoration work correctly
- **Training time**: Start with small grid searches and short training runs
- **Overfitting analysis**: Need clear comparison between baseline, induced, and mitigated scenarios
- **Functional API**: Understanding how layers connect and concatenate

## Computational Considerations
This assignment is designed to run on CPU in reasonable time (~30-60 minutes total):
- Small model architectures (~50k-100k parameters)
- Subset of MNIST (10k training samples)
- Short training runs (10-30 epochs for grid search, 30-50 for final models)
- Limited grid search (6-12 configurations)

If you have GPU access (Colab, Kaggle, local), you can optionally:
- Use full MNIST dataset (60k training samples)
- Train for more epochs
- Explore larger grid searches
- Use more complex architectures

## Submission Requirements
1. Completed `student_code.py` with all functions implemented
2. Completed `visualization_analysis.ipynb` with analysis and matplotlib visualizations
3. All tests passing (`python -m pytest tests.py`)
4. PDF export of analysis notebook for peer review

This assignment builds practical skills for training and tuning CNNs - essential capabilities for modern deep learning development.
