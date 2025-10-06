# PA3: Backpropagation & Neural Networks

This assignment focuses on understanding the mechanics of neural network training through implementation of backpropagation and analysis of training dynamics on challenging nonlinear problems.

## Assignment Components

### 1. Neural Network Implementation (`student_code.py`)
- **Forward Pass Functions**: Implement linear transformations and activation functions (sigmoid, ReLU, softmax)
- **Loss Functions**: Implement MSE and cross-entropy loss with their derivatives
- **Backpropagation Core**: Implement gradient computation through linear layers using the chain rule
- **Training Functions**: Complete single-layer training with gradient descent and weight updates
- **Gradient Checking**: Implement simplified gradient verification for debugging

### 2. Neural Network Analysis (`visualization_analysis.ipynb`)
- **Chain Rule Analysis**: Trace gradient flow through networks with concrete numerical examples
- **Architecture Comparison**: Design experiments comparing single neurons vs. multi-layer networks
- **Training Dynamics**: Analyze learning rate effects, convergence patterns, and common training problems
- **Activation Function Study**: Compare how different activation functions affect gradient flow and training
- **Professional Skills**: Develop practical debugging and optimization techniques for neural networks

### 3. Peer Review Component
- Export notebook as PDF for peer evaluation
- Focus on experimental design choices and interpretation of results
- Provide actionable insights for neural network development

## Learning Objectives
- **Chain Rule Mastery**: Understand gradient flow through multi-layer networks
- **Backpropagation Implementation**: Build gradient computation from scratch
- **Training Dynamics**: Diagnose and fix common neural network training problems
- **Architecture Understanding**: Analyze how network complexity affects solution quality
- **Professional ML Skills**: Develop debugging techniques used in real-world neural network development

## Dataset
The assignment uses a "Swiss Roll" dataset - a synthetic 2D nonlinear classification problem that demonstrates the limitations of linear methods and the power of neural networks.

## Grading Breakdown
- **Technical Implementation (70%)**: `student_code.py` - Forward pass, backpropagation, gradient checking, test suite performance
- **Analysis & Communication (30%)**: `visualization_analysis.ipynb` - Experimental design, gradient flow analysis, training insights, peer review preparation

## Technical Requirements
- All gradient computations must use your custom backpropagation implementation
- Use provided helper functions but make your own analytical choices
- Complete both code implementation and experimental analysis
- Test your implementation using the provided test suite and gradient checking

## Getting Started

### 1. Implementation Phase
```bash
# Test your implementations
python -m pytest tests.py -v

# Check specific function
python -m pytest tests.py::TestBackpropagation::test_linear_backward_known_gradients -v
```

### 2. Analysis Phase
Open `visualization_analysis.ipynb` and work through the analysis sections. You'll use helper functions but make choices about:
- Which examples to analyze for chain rule demonstration
- What network architectures to compare
- Which learning rates and training parameters to test
- How to interpret and present your findings

### 3. Key Implementation Functions
- `linear_forward()` and `linear_backward()` - Core linear layer operations
- `sigmoid_forward()` and `sigmoid_derivative()` - Sigmoid activation and its derivative
- `single_layer_forward()` and `single_layer_backward()` - Complete single layer with activation
- `train_single_layer()` - End-to-end training loop
- `simple_gradient_check()` - Gradient verification tool

## Tips for Success

### Implementation Strategy
1. **Start with forward pass** - Get linear transformations and activations working first
2. **Implement derivatives carefully** - Pay attention to matrix dimensions and chain rule application
3. **Use gradient checking** - Verify your backpropagation implementation before training
4. **Test incrementally** - Run tests frequently to catch errors early

### Analysis Strategy
1. **Choose examples thoughtfully** - Pick cases that clearly illustrate key concepts
2. **Design systematic experiments** - Compare architectures and parameters methodically
3. **Connect theory to practice** - Explain why certain phenomena occur, not just what happens
4. **Focus on professional insights** - What would help someone building neural networks in practice?

## Common Challenges
- **Shape mismatches**: Pay careful attention to matrix dimensions in gradient computation
- **Chain rule application**: Remember that gradients flow backward, multiplying local derivatives
- **Numerical stability**: Handle edge cases in activation functions and loss computation
- **Training diagnosis**: Learn to recognize oscillation, divergence, and convergence patterns

## Submission Requirements
1. Completed `student_code.py` with all functions implemented
2. Completed `visualization_analysis.ipynb` with analysis and insights
3. All tests passing (`python -m pytest tests.py`)
4. PDF export of analysis notebook for peer review

This assignment builds critical skills for understanding and debugging neural networks - skills that are essential for real-world machine learning development.