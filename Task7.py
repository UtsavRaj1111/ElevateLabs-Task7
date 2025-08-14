# Task 7: Support Vector Machines (SVM)
# =====================================
# Objective: Linear & Non-linear classification using SVM
# Dataset: Breast Cancer dataset from seaborn / sklearn
# Tools: Scikit-learn, NumPy, Matplotlib, seaborn

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# 1. Load dataset
# -------------------------------------------------
# Using sklearn's breast cancer dataset
data = datasets.load_breast_cancer()
X = data.data
y = data.target

# For visualization, let's use only first two features
X_vis = X[:, :2]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_vis, y, test_size=0.3, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. Train SVM with Linear Kernel
# -------------------------------------------------
svm_linear = SVC(kernel='linear', C=1)
svm_linear.fit(X_train, y_train)

# 3. Train SVM with RBF Kernel (non-linear)
# -------------------------------------------------
svm_rbf = SVC(kernel='rbf', C=1, gamma='scale')
svm_rbf.fit(X_train, y_train)

# 4. Visualization function
# -------------------------------------------------
def plot_decision_boundary(model, X, y, title):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.show()

# Plot decision boundaries
plot_decision_boundary(svm_linear, X_train, y_train, "SVM Linear Kernel (Train Data)")
plot_decision_boundary(svm_rbf, X_train, y_train, "SVM RBF Kernel (Train Data)")

# 5. Hyperparameter tuning for RBF kernel
# -------------------------------------------------
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 'scale', 'auto']
}
grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Params:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)

# 6. Final evaluation
# -------------------------------------------------
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Test Accuracy:", accuracy_score(y_test, y_pred))
