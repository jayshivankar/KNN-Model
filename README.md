# K-Nearest Neighbors (KNN) Classifier on Iris Dataset

This project implements a **K-Nearest Neighbors (KNN)** classifier on the classic **Iris** dataset to classify flowers into three species: *Setosa*, *Versicolor*, and *Virginica* based on their features.

## 📄 Dataset

The dataset used is [`Iris.csv`](./Iris.csv), which contains:

- **Features**:  
  - Sepal Length (cm)  
  - Sepal Width (cm)  
  - Petal Length (cm)  
  - Petal Width (cm)

- **Target**:  
  - Species (*Setosa*, *Versicolor*, *Virginica*)

## 📚 Libraries Used

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## 🚀 Project Workflow

1. **Load the data**  
   Read and explore the Iris dataset using `pandas`.

2. **Preprocessing**  
   - Handle missing values (if any).  
   - Encode the target labels if necessary.

3. **Data Visualization**  
   - Plot pairplots and correlation heatmaps for feature exploration.

4. **Train-Test Split**  
   Split the data into training and testing sets (e.g., 80/20 split).

5. **Model Training**  
   - Use `KNeighborsClassifier` from `sklearn.neighbors`.
   - Choose an appropriate value of **k** (optimized via cross-validation or manually tested).

6. **Evaluation**  
   - Accuracy Score
   - Confusion Matrix
   - Classification Report

7. **Result Visualization**  
   Plot the decision boundary (optional) and visualize the confusion matrix.

## 🛠 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/knn-iris-classifier.git
   cd knn-iris-classifier
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook or Python script:
   ```bash
   python knn_iris.py
   ```

## 📝 Sample Results

| Metric          | Score |
|-----------------|-------|
| Accuracy        | 96%   |
| Precision       | 96%   |
| Recall          | 96%   |
| F1 Score        | 96%   |

> 📈 The model achieves excellent classification performance with minimal hyperparameter tuning.


