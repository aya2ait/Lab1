Deep Learning Lab: Classification and Regression with PyTorch

Lab Overview
This lab focused on building and training Deep Neural Networks (DNNs) using the PyTorch library for two main tasks: regression and multi-class classification. The goal was to gain hands-on experience with data preprocessing, model architecture design, hyperparameter tuning, and evaluation techniques. The lab was divided into two parts, each using a different dataset and addressing specific machine learning challenges.

Part 1: Regression
Dataset: NYSE Dataset
The dataset contains financial data from the New York Stock Exchange (NYSE). The goal was to predict a continuous target variable using regression techniques.

Key Steps:
Exploratory Data Analysis (EDA):

Analyzed the dataset to understand its structure, identify missing values, and visualize trends.

Used plots like histograms, scatter plots, and correlation matrices to gain insights.

Model Building:

Designed a Deep Neural Network (DNN) using PyTorch for regression.

Implemented layers, activation functions, and loss functions suitable for regression tasks.

Hyperparameter Tuning:

Used GridSearch from sklearn to optimize hyperparameters such as learning rate, optimizer, and number of epochs.

Visualization:

Plotted Loss vs. Epochs and Accuracy vs. Epochs for training and test data.

Analyzed the learning curves to ensure the model was not overfitting or underfitting.

Regularization:

Applied techniques like dropout and L2 regularization to improve generalization.

Compared the performance of the regularized model with the initial model.

Part 2: Multi-Class Classification
Dataset: Machine Predictive Maintenance Classification Dataset
This dataset contains information about industrial machines, and the goal was to classify them into multiple categories based on their condition.

Key Steps:
Pre-processing:

Cleaned the dataset by handling missing values, encoding categorical variables, and normalizing/standardizing features.

Exploratory Data Analysis (EDA):

Visualized class distributions, feature correlations, and outliers to understand the dataset.

Data Augmentation:

Applied techniques like SMOTE or random oversampling to balance the dataset and address class imbalance.

Model Building:

Designed a Deep Neural Network (DNN) using PyTorch for multi-class classification.

Used appropriate loss functions (e.g., Cross-Entropy Loss) and activation functions (e.g., Softmax for the output layer).

Hyperparameter Tuning:

Used GridSearch to find the best combination of hyperparameters.

Visualization:

Plotted Loss vs. Epochs and Accuracy vs. Epochs to monitor training and test performance.

Evaluation Metrics:

Calculated metrics like accuracy, precision, recall, and F1 score to evaluate the model's performance.

Regularization:

Applied regularization techniques and compared the results with the initial model to ensure better generalization.

What I Learned
PyTorch Basics:

Gained a solid understanding of PyTorch's core components, such as tensors, autograd, and neural network modules.

Model Design:

Learned how to design and implement DNN architectures for both regression and classification tasks.

Hyperparameter Tuning:

Used GridSearch to systematically explore and optimize hyperparameters for better model performance.

Regularization:

Understood the importance of regularization techniques like dropout and L2 regularization to prevent overfitting.

Evaluation:

Learned how to evaluate models using appropriate metrics and visualize training progress using learning curves.

Data Preprocessing:

Gained experience in cleaning, normalizing, and augmenting datasets to improve model performance.

Conclusion
This lab provided a comprehensive introduction to deep learning using PyTorch. By working on both regression and classification tasks, I gained practical experience in data preprocessing, model building, hyperparameter tuning, and evaluation. The lab reinforced the importance of understanding the dataset, designing appropriate architectures, and using regularization techniques to build robust models. Overall, it was a valuable learning experience that deepened my understanding of deep learning concepts and their practical applications.
