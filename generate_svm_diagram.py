# generate_svm_diagram.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from matplotlib.lines import Line2D

# Load dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Reduce dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Train SVM with specified parameters
svm = SVC(kernel='rbf', C=0.1, gamma=0.01, probability=True)
svm.fit(X_pca, y)

# Create mesh grid for visualization
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))

# Predict on mesh grid
Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot configuration
plt.figure(figsize=(12, 8))

# Plot decision boundary and margins
plt.contourf(xx, yy, Z > 0, cmap=plt.cm.coolwarm, alpha=0.3)
contour = plt.contour(xx, yy, Z,
                     levels=[-1, 0, 1],
                     colors='k',
                     linestyles=['--', '-', '--'])

# Plot data points
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y,
                     cmap=plt.cm.coolwarm,
                     edgecolors='k',
                     s=50, label='Samples')

# Highlight support vectors
plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
            s=150, facecolors='none', edgecolors='k',
            linewidths=1.5, label='Support Vectors')

# Labels and annotations
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
plt.title('SVM Decision Boundary (C=1, γ=0.01) - Breast Cancer Diagnosis',
         fontsize=14, pad=20)

# Custom legend for boundary lines
legend_elements = [
    Line2D([0], [0], color='k', linestyle='-', label='Decision Boundary'),
    Line2D([0], [0], color='k', linestyle='--', label='Margin Boundaries'),
    Line2D([0], [0], marker='o', color='w', label='Support Vectors',
           markerfacecolor='none', markeredgecolor='k', markersize=10, linewidth=0),
    Line2D([0], [0], marker='o', color='w', label='Benign',
           markerfacecolor=plt.cm.coolwarm(0.), markeredgecolor='k', markersize=10, linewidth=0),
    Line2D([0], [0], marker='o', color='w', label='Malignant',
           markerfacecolor=plt.cm.coolwarm(1.), markeredgecolor='k', markersize=10, linewidth=0)
]
plt.legend(handles=legend_elements, loc='lower right', fontsize=11)

# Parameter annotation box
plt.text(0.05, 0.95,
        f"SVM Parameters:\n- C (Regularization): 0.1\n- γ (Gamma): 0.01\n- Kernel: RBF\n"
        f"Support Vectors: {len(svm.support_vectors_)}",
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.savefig('svm_diagram.png', dpi=300, bbox_inches='tight')
plt.show()
