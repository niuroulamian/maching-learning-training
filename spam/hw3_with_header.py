from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer # 提取文本特征向量的类
from sklearn.model_selection import train_test_split # 划分训练集和测试集
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB # 三种朴素贝叶斯算法，差别在于估计p(x|y)的方式
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import os
import numpy as np

# 同时返回 header + body
def extract_header_and_body(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    content = content.replace('\r\n', '\n')

    parts = content.split('\n\n', 1)

    if len(parts) == 2:
        header, body = parts
    else:
        header, body = content, ""

    return header.strip(), body.strip()

# 提取 Subject / From
def extract_header_fields(header):
    subject = ""
    sender = ""

    for line in header.split('\n'):
        if line.lower().startswith('subject:'):
            subject = line[len('subject:'):].strip()
        elif line.lower().startswith('from:'):
            sender = line[len('from:'):].strip()

    return subject, sender

# 把 header + body 拼在一起，但加标记防止混淆
def build_full_text(filepath):
    header, body = extract_header_and_body(filepath)
    subject, sender = extract_header_fields(header)
    # 加 SUBJECT / FROM / BODY，让模型区分字段语义
    full_text = (
        "SUBJECT " + subject + " "
        "FROM " + sender + " "
        "BODY " + body
    )

    return full_text

# 用这个文本重新构建数据集
def load_dataset_with_header(label_file, base_dir):
    texts = []
    labels = []

    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            label, path = line.strip().split()
            filepath = os.path.join(base_dir, path.replace('../', ''))

            text = build_full_text(filepath)

            texts.append(text)
            labels.append(1 if label == 'spam' else 0)

    return texts, labels

# 特征构造： 将词变成模型能用的数值向量
texts, labels = load_dataset_with_header(
    label_file='trec06c-utf8/label/index',
    base_dir='trec06c-utf8'
)

#min_df=2 去掉只出现一次的噪声词
#max_df=0.95 去掉“的、了、是”这类全邮件通用词
vectorizer = TfidfVectorizer(
    min_df=2,
    max_df=0.95,
    sublinear_tf=True  # 用 log(tf)
)

#TREC06 的特点是：
#垃圾邮件和正常邮件 比例不一定 1:1
#如果你随机切分：
#可能出现：
#训练集 spam 很多
#测试集 spam 很少（或反之）
#模型评估直接失真
#stratify=labels 的作用：
#在训练集和测试集中，保持 spam / ham 的比例一致
#这是文本分类的基本规范。

train_texts, test_texts, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,        # 20% 做测试集
    random_state=42,      # 固定随机种子，方便复现实验
    stratify=labels       # 保证 spam / ham 比例一致（非常重要）
)

X_train = vectorizer.fit_transform(train_texts)
X_test  = vectorizer.transform(test_texts)
# 检查划分结果
print(len(train_texts), len(test_texts))
print("Train spam ratio:", np.mean(y_train))
print("Test spam ratio:", np.mean(y_test))

# 训练朴素贝叶斯
# alpha=1.0 是为了防止：P(w∣y)=0，否则只要测试集中出现一个训练中没见过的词，整封邮件概率就变成0
clf = MultinomialNB(alpha=1.0)  # Laplace smoothing
clf.fit(X_train, y_train)

# 输出准确率 / 混淆矩阵
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Accuracy:", acc)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))


# 不同特征数量下的效果对比
from sklearn.metrics import f1_score

feature_sizes = [1000, 3000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
results = []

for max_feat in feature_sizes:
    vectorizer = TfidfVectorizer(
        max_features=max_feat,
        min_df=2,
        max_df=0.95
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_test  = vectorizer.transform(test_texts)

    clf = MultinomialNB(alpha=1.0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)

    results.append((max_feat, X_train.shape[1], acc, f1))

import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))

vocab_sizes = []
accuracies = []
f1_scores = []

for r in results:
    print(f"max_features={r[0]:6d}, "
          f"vocab_size={r[1]:6d}, "
          f"acc={r[2]:.4f}, "
          f"f1={r[3]:.4f}")
    vocab_sizes.append(r[1])
    accuracies.append(r[2])
    f1_scores.append(r[3])

# 画 Accuracy 和 F1-score
plt.plot(vocab_sizes, accuracies, marker='o', label='Accuracy')
plt.plot(vocab_sizes, f1_scores, marker='s', label='F1-score')

plt.xlabel('Vocabulary Size')
plt.ylabel('Score')
plt.title('Effect of Vocabulary Size on Spam Classification')
plt.legend()
plt.grid(True)
plt.show()
