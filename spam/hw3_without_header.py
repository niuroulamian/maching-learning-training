from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer # 提取文本特征向量的类
from sklearn.model_selection import train_test_split # 划分训练集和测试集
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB # 三种朴素贝叶斯算法，差别在于估计p(x|y)的方式
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import os
import numpy as np

def extract_body_from_email(filepath):
    """
    从一封邮件中提取正文部分
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # 统一换行符
    content = content.replace('\r\n', '\n')

    # 按空行分割
    parts = content.split('\n\n', 1)

    if len(parts) == 2:
        body = parts[1]
    else:
        # 如果没有明显分隔，直接返回全文（兜底）
        body = content

    return body.strip()

def load_all_bodies(data_dir):
    """
    读取目录下所有邮件正文
    返回：{filepath: body_text}
    """
    bodies = {}

    for root, _, files in os.walk(data_dir):
        for fname in files:
            filepath = os.path.join(root, fname)
            body = extract_body_from_email(filepath)
            bodies[filepath] = body

    return bodies

def load_dataset(label_file, base_dir):
    texts = []
    labels = []

    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            label, path = line.strip().split()
            filepath = os.path.join(base_dir, path.replace('../', ''))

            body = extract_body_from_email(filepath)

            texts.append(body)
            labels.append(1 if label == 'spam' else 0)

    return texts, labels

# 特征构造： 将词变成模型能用的数值向量
texts, labels = load_dataset(
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