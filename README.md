# Diabetes Prediction using Machine Learning Models

This project focuses on building machine learning models to predict the likelihood of diabetes based on patient data. The dataset consists of various medical attributes, including glucose levels, insulin, BMI, age, and others, which are processed to predict whether a patient has diabetes or not.

The dataset used in this project is the **Pima Indians Diabetes Database** from Kaggle:
[Kaggle Dataset - Pima Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data)

## Approach Followed

### 1. **Data Preprocessing:**
   - **Scaling:** The features were scaled using `StandardScaler` to ensure that they all have a similar range, which is essential for certain models (e.g., SVM).
   - **Handling Class Imbalance:** The dataset was imbalanced, so `RandomOverSampler` was used to balance the classes by oversampling the minority class (Diabetes).
   - **Train-Test Split:** The dataset was split into training and testing sets using `train_test_split` with a 70-30 split, ensuring that the distribution of classes was stratified.

### 2. **Exploratory Data Analysis (EDA):**
   - Histograms were plotted for each feature, comparing the distributions of patients with and without diabetes.



### 3. **User Interface (Gradio):**
   - A user-friendly interface was created using Gradio, where users can input various patient attributes (e.g., glucose levels, age, BMI) to predict the likelihood of diabetes.
   - The interface uses a simple slider to accept input values and displays the prediction as text.

### 4. **Model Performance Evaluation:**
   - **Neural Network Model:** The NN model showed the highest performance with an average accuracy of 86.4% across 10 folds in cross-validation. The model was able to capture more complex relationships in the data compared to traditional models.
   - **Random Forest:** The Random Forest model also provided strong results with an accuracy of 83.6%.
   - **SVM:** The SVM model had a slightly lower performance but still showed reasonable accuracy (77.33%).

## Results

1. **SVM Model:**
   - Test Accuracy: 77.33%
   - Confusion Matrix:
     - TP: 116, TN: 34, FP: 34, FN: 116

2. **Random Forest Model:**
   - Test Accuracy: 83.60%
   - Confusion Matrix:
     - TP: 122, TN: 28, FP: 21, FN: 129

3. **Neural Network Model:**
   - Average Accuracy (10-Fold CV): 86.40%
   - Average Loss (10-Fold CV): 0.339
   - Confusion Matrix:
     - TP: 148, TN: 2, FP: 12, FN: 138

## Takeaways

- **Imbalanced Data:** The dataset had a class imbalance problem, but oversampling the minority class improved model performance significantly.
- **Model Performance:** The Neural Network (NN) outperformed traditional models like SVM and Random Forest in terms of both accuracy and handling the data complexity.
- **Model Evaluation:** The confusion matrices revealed that while the models were good at detecting diabetes (high TP), they struggled with the False Negatives (FN), which indicates the need for further model tuning or data enhancement.
- **Practical Application:** This model can be used in real-world applications where medical professionals can use input features (e.g., glucose levels, age, etc.) to predict the likelihood of diabetes, helping with early diagnosis and prevention.

## Future Work

- **Hyperparameter Tuning:** The performance of models can be further improved by tuning hyperparameters like the number of trees in Random Forest or the learning rate in the Neural Network.
- **Handling Class Imbalance:** Further techniques like SMOTE (Synthetic Minority Over-sampling Technique) could be explored to improve model performance.
- **Advanced Neural Networks:** More complex neural networks or deep learning models could be explored for even better performance.

## Technologies Used
- Python
- Scikit-learn
- TensorFlow (Keras)
- Gradio
- Matplotlib
- Pandas
- NumPy
