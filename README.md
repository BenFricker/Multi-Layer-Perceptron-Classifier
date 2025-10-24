# Pokemon Multi-Layer Perceptron Classifier 🎮🤖

A machine learning project that classifies Pokemon images using a Multi-Layer Perceptron (MLP) neural network with optimized color histogram features.

# 📋 Project Overview
This project implements a comprehensive machine learning pipeline to classify 10 different Pokemon species using computer vision techniques and neural networks. The system automatically determines the optimal MLP architecture and histogram configuration through systematic experimentation.
Key Features

Automated Architecture Selection: Tests 9 different MLP structures to find optimal configuration
Feature Optimization: Evaluates multiple color histogram sizes for best performance
Comprehensive Evaluation: Implements accuracy, precision, recall, and F1-score metrics
Visual Analysis: Generates confusion matrices to identify classification patterns

# 🎯 Pokemon Classes
The classifier distinguishes between 10 Pokemon:

Bulbasaur
Meowth
Mew
Pidgeot
Pikachu
Snorlax
Squirtle
Venusaur
Wartortle
Zubat

# 🛠️ Technologies Used

Python 3.x
scikit-learn: MLP implementation and evaluation metrics
OpenCV: Image processing and feature extraction
NumPy: Numerical computations
Matplotlib: Visualization
tqdm: Progress tracking

# 📊 Methodology
## 1. Data Pipeline

Preprocessing: Images resized to 150x150 pixels
Split Strategy:

Training: 70%
Validation: 10%
Testing: 20%



## 2. Feature Extraction
Utilizes color histograms as feature vectors:

Extracts RGB color distribution
Tests histogram bins: 4x4x4, 6x6x6, and 8x8x8
Normalizes features for consistent scaling

## 3. Architecture Optimization
### Evaluates 9 different MLP configurations:

Single hidden layer: 3 variations based on neuron count formulas
Two hidden layers: 3 configurations
Three hidden layers: 3 deep network structures

### Neuron count calculations:

Formula 1: (input_size + output_size) / 2
Formula 2: (2/3 × input_size) + output_size
Formula 3: 80% of input_size

## 4. Model Training

Activation Function: ReLU
Optimizer: Adam
Early Stopping: Enabled to prevent overfitting
Maximum Iterations: 1500

# 📈 Results
The system automatically identifies:

Optimal MLP architecture for the dataset
Best performing histogram configuration
Most frequently confused Pokemon pairs
Detailed performance metrics for each configuration

## Performance Metrics

Accuracy: Overall classification correctness
Precision: Positive prediction accuracy
Recall: True positive identification rate
F1-Score: Harmonic mean of precision and recall

# 🚀 Getting Started
Prerequisites
bashpip install numpy opencv-python scikit-learn matplotlib tqdm
Dataset Structure
Organize your Pokemon images as follows:

# Dataset/
└── PokemonData/
    ├── Bulbasaur/
    │   ├── image1.jpg
    │   └── ...
    ├── Meowth/
    │   └── ...
    └── [Other Pokemon folders]
Running the Classifier
bashpython Multi-Layer-Perceptron-Classifier.py
Expected Output

## Data split statistics
Neuron count calculations for each formula
Performance metrics for each MLP structure
Optimal structure identification
Results for different histogram sizes
Final confusion matrix visualization
Most confused Pokemon pair identification

# 📊 Sample Output

## DETERMINING OPTIMAL MLP STRUCTURE
The Optimal MLP Structure is: (154, 154)
With validation accuracy: 0.8567

## CLASSIFICATION METRICS FOR MLPCLASSIFIER
The Accuracy Score for this model is:   0.8734
The Precision Score for this model is:  0.8756
The Recall Score for this model is:     0.8734
The f1 Score for this model is:         0.8721

# 🔍 Key Insights
Feature Engineering
Color histograms prove effective for Pokemon classification, capturing distinctive color patterns of each species while remaining computationally efficient.
Architecture Selection
The systematic approach to architecture selection ensures optimal performance without manual hyperparameter tuning.
Confusion Analysis
The confusion matrix reveals which Pokemon share similar visual characteristics, providing insights for potential feature improvements.

## 🎓 Learning Outcomes
This project demonstrates:

1. Machine Learning Pipeline: End-to-end ML workflow implementation
2. Neural Network Design: Understanding of MLP architecture decisions
3. Computer Vision: Practical feature extraction techniques
4. Model Evaluation: Comprehensive performance assessment
5. Code Organization: Clean, documented, and modular programming

# 🔄 Future Improvements

 Implement CNN for improved accuracy
 Add data augmentation techniques
 Expand to more Pokemon species
 Create web interface for real-time classification
 Implement transfer learning approach
 Add cross-validation for more robust evaluation

# 📄 License
This project is for educational purposes. Pokemon characters are property of Nintendo/Game Freak.

# 👤 Author
Ben Fricker

Computer Science Student at University of Wollongong
Majoring in AI & Big Data and Cybersecurity
LinkedIn
GitHub

# 🤝 Acknowledgments

University of Wollongong for project guidance
scikit-learn documentation for MLP implementation details
Pokemon dataset contributors


This project is part of my journey in exploring machine learning and computer vision applications.
