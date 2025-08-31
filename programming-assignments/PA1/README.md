# PA1: Linear Algebra, PCA, and Data Visualization

This homework assignment focuses on material from the first week or two of class, specifically on linear algebra and dimensionality reduction with practical application to real data.

## Assignment Components

### 1. Linear Algebra Implementation (`student_code.py`)
- **Custom Matrix Multiplication**: Implement `matrix_multiply()` function using nested loops to internalize how matrix operations work
- **PCA Pipeline**: Complete implementation of Principal Component Analysis using your custom matrix multiplication
- **Core Functions**: Data centering, SVD decomposition, component extraction, variance calculation, and transformations

### 2. Data Visualization & Analysis (`visualization_analysis.ipynb`)
- **Naive Feature Exploration**: Select and visualize 2-3 features you think are most informative about colleges
- **PCA-Based Analysis**: Apply your custom PCA implementation to the college dataset  
- **Comparative Analysis**: Compare insights from naive feature selection vs. principled dimensionality reduction
- **Communication Skills**: Prepare findings for peer review with clear explanations and visualizations

### 3. Peer Review Component
- Export notebook as PDF for peer evaluation
- Focus on clear communication of insights and methodology
- Provide specific questions for peer reviewers

## Learning Objectives
- **Linear Algebra**: Understand matrix multiplication through hands-on implementation
- **Dimensionality Reduction**: Apply PCA to real data and interpret results
- **Data Visualization**: Create effective plots using matplotlib
- **Critical Thinking**: Compare different approaches to feature selection and analysis
- **Communication**: Explain technical concepts clearly for peer review

## Dataset
The assignment uses `College.csv`, containing data on 777 US colleges with 17 features including enrollment numbers, tuition costs, graduation rates, and institutional characteristics.

## Grading Breakdown
- **Technical Implementation (70%)**: `student_code.py` - Matrix multiplication, PCA functions, test suite performance
- **Communication & Analysis (30%)**: `visualization_analysis.ipynb` - Data exploration, PCA interpretation, comparative analysis, peer review preparation

## Technical Requirements
- All matrix operations must use your custom `matrix_multiply()` function
- Visualizations should use matplotlib (no seaborn)
- Complete both code implementation and written analysis
- Test your PCA implementation against the provided test suite