import os
import numpy as np
from PIL import Image

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def load_dataset(root_dir):
    X = []
    y = []

    for label_name in sorted(os.listdir(root_dir)):
        label_path = os.path.join(root_dir, label_name)
        if not os.path.isdir(label_path):
            continue

        label = int(label_name)

        for file in os.listdir(label_path):
            img_path = os.path.join(label_path, file)
            img = Image.open(img_path).convert('L')  # 灰度
            pixels = np.array(img)
            pixels.shape  # (height, width)

            # 二值图已经是 0/255 或 0/1，这里统一拉平
            X.append(pixels.flatten())
            y.append(label)

    return np.array(X), np.array(y)

X_train, y_train = load_dataset('data/train')
X_test, y_test = load_dataset('data/test')

def evaluate_knn(k):
    knn = KNeighborsClassifier(
        n_neighbors=k,
        weights='distance',   # 距离加权
        metric='euclidean'    # 欧氏距离
    )
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc

K_values = range(1, 21)
accuracies = []

for k in K_values:
    acc = evaluate_knn(k)
    accuracies.append(acc)
    print(f"K={k}, Test Accuracy={acc:.4f}")


from sklearn.model_selection import StratifiedShuffleSplit

# Find the best K from the first evaluation
best_k = K_values[np.argmax(accuracies)]
print(f"\nBest K value: {best_k} with accuracy: {accuracies[np.argmax(accuracies)]:.4f}\n")

def evaluate_with_ratio(ratio, k, repeats=5):
    knn = KNeighborsClassifier(
        n_neighbors=k,
        weights='distance',   # 距离加权
        metric='euclidean'    # 欧氏距离
    )
    accs = []
    
    # If ratio is 1.0, use full training set directly
    if ratio >= 1.0:
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        return acc, 0.0  # No std for single evaluation
    
    for _ in range(repeats):
        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=ratio, random_state=None
        )
        for train_idx, _ in sss.split(X_train, y_train):
            X_sub = X_train[train_idx]
            y_sub = y_train[train_idx]

            knn.fit(X_sub, y_sub)
            y_pred = knn.predict(X_test)
            accs.append(accuracy_score(y_test, y_pred))
    return np.mean(accs), np.std(accs)

# Evaluate with different training set ratios
ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ratio_results = []

for ratio in ratios:
    mean_acc, std_acc = evaluate_with_ratio(ratio, best_k)
    ratio_results.append((ratio, mean_acc, std_acc))
    print(f"Ratio={ratio:.1f}, Test Accuracy={mean_acc:.4f} ± {std_acc:.4f}")
