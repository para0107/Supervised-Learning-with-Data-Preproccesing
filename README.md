📊 Predictive Analysis Using Regression Models

🔍 This project utilizes pre-implemented machine learning algorithms in Python to predict an important feature from a processed and cleaned dataset. The following regression models are applied:

✅ Linear Regression
✅ Decision Tree Regression
✅ Random Forest Regression

📈 Below, we visualize the accuracy of predictions compared to actual values from the database.

📌 Table of Contents

📊 Algorithms Used

⚙️ Data Processing

📏 Model Evaluation

📉 Visualizations



📊 Algorithms Used
The following regression algorithms are implemented and compared:

🔹 Linear Regression
A simple regression model that assumes a linear relationship between input and output variables.

🔹 Decision Tree Regression
A non-linear model that splits the data into decision-based branches for predictions.

🔹 Random Forest Regression
An ensemble learning method that builds multiple decision trees and averages their predictions for higher accuracy.

📌 Each model's performance is evaluated and compared visually and numerically.

⚙️ Data Processing
The dataset used in this project has undergone multiple preprocessing and cleaning steps, including:

🔹 Removing null or missing values 🗑️
🔹 Normalizing/standardizing numerical features 📏
🔹 Encoding categorical variables 🔡
🔹 Handling outliers and inconsistencies 📊

✅ These steps ensure data quality and improve model accuracy.

📏 Model Evaluation
The models are assessed using the following key performance metrics:

📌 Mean Absolute Error (MAE) – Measures the average magnitude of prediction errors.
📌 Mean Squared Error (MSE) – Penalizes larger errors by squaring them.
📌 Root Mean Squared Error (RMSE) – Square root of MSE, making it more interpretable.
📌 R-squared (R²) – Indicates how well the model explains variance in the data.

📊 These metrics help us understand how accurate and reliable each model is.

Although not part of the initial regression models, CNN.ipynb introduces a deep learning-based approach for image recognition:

It uses the MNIST dataset to recognize handwritten digits.

Two approaches are compared:

A basic neural network using Scikit-learn.

A more advanced architecture with TensorFlow/Keras using fully connected layers.

The notebook shows preprocessing steps like:

Normalizing image pixel values to [0, 1]

Flattening 2D images to 1D arrays

This model is evaluated using accuracy and loss curves, complementing the regression models with a classification task.


📘 H4.ipynb — Regression-Based Prediction & Evaluation
This notebook implements and evaluates multiple regression models to predict a target variable from a cleaned dataset. Key steps include:

🔹 Data Splitting: The dataset is divided into training and testing subsets using train_test_split to ensure a fair model evaluation.

🔹 Model Training: Three models are implemented:

LinearRegression

DecisionTreeRegressor

RandomForestRegressor

🔹 Performance Metrics: The following metrics are computed to assess model accuracy:

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

R-squared (R²)

🔹 Visualizations:

Prediction vs Actual plots show how well model outputs align with real values.

Error Distribution graphs help detect bias and error spread across data.

This notebook forms the core analysis of the project, showcasing how different regression models compare in predicting outcomes based on the same input data.

 CNN.ipynb — MNIST Digit Classification with Neural Networks
Although not part of the regression task, this notebook introduces a classification problem using the MNIST dataset of handwritten digits. Key highlights include:

🔹 Data Preprocessing:

Normalization of pixel values to [0, 1] for better convergence.

Flattening 28×28 pixel images into 1D arrays of 784 values for input into neural networks.

🔹 Model Architectures:

A simple neural network is built using Sequential from TensorFlow/Keras.

The architecture includes fully connected (dense) layers with activation functions.

🔹 Training & Evaluation:

Models are trained using cross-entropy loss and evaluated by accuracy.

Loss and accuracy curves are plotted to observe training dynamics.

This notebook complements the regression analysis with a deep learning example, demonstrating model creation and training for image-based classification tasks.



📉 Visualizations
📌 Prediction vs Actual Values
📈 This plot demonstrates how well the model's predictions align with actual values.

📈 This plot visually compares the values predicted by each regression model (Linear, Decision Tree, Random Forest) with the actual target values from the dataset.
It helps assess whether the model is systematically underperforming, overfitting, or matching the trend accurately. A near 45-degree line suggests high alignment between prediction and reality.


📌 Error Distribution
🔍 This plot shows the distribution of errors across the dataset for each model.


