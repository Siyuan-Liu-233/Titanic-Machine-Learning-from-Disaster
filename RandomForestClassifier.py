from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
from sklearn.model_selection import cross_val_score
import re
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import time
# 数据预处理

# 处理缺失值
titanic = pd.read_csv("train.csv")
# print(titanic.describe())  #发现age缺失值
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())  # 用中值填充年龄的缺失值
# print(titanic.describe())

# 字符串类别转换为数字
# print(titanic["Sex"].unique()) #查看类别种类
# print(titanic["Sex"]=="male")
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
# print(titanic["Embarked"].describe()) 看哪个最多 来填充缺失值
# print(titanic.Embarked.unique())
titanic["Embarked"] = titanic["Embarked"].fillna("S")
num = 0
for i in titanic.Embarked.unique():  # 仓型转化为数值
    titanic.loc[titanic.Embarked == i, "Embarked"] = num
    num = num + 1

predictoos = ["Pclass", "Sex", "Age", "SibSp", "Parch",
              "Fare", "Embarked", "titles", "FamiliySize", "NameLength"]
titanic["FamiliySize"] = titanic["SibSp"] + titanic["Parch"]
titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7,
                 "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 9, "Countess": 9, "Capt": 10, "Ms": 10, "Sir": 10, "Jonkheer": 7}


def get_title(name):  # 提取出称谓
    title_search = re.search('([A-Z,a-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    else:
        return ''
titles = titanic["Name"].apply(get_title)
for k, v in title_mapping.items():
    titles[titles == k] = v  # 称谓转化为数字形式

titanic["titles"] = titles
alg = ExtraTreesClassifier(
    random_state=1, n_estimators=50, min_samples_split=4, min_samples_leaf=2)
score = cross_val_score(alg, titanic[predictoos], titanic.Survived, cv=3)
print("随机森林准确率为:{accuracy}".format(accuracy=score.mean()))

# 用随机噪声代替特征 比较错误率 来得到重要的特征

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch",
              "Fare", "Embarked", "titles", "FamiliySize", "NameLength"]
selector = SelectKBest(f_classif, k=3)
selector.fit(titanic[predictors], titanic.Survived)

scores = selector.scores_
plt.figure(figsize=(6, 8))
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation="45")
plt.title("score of feature")


# 用简化后的特征重新预测

predictors = ["Pclass", "Sex", "Fare",
              "titles", "NameLength", "Embarked", "Parch"]
alg = ExtraTreesClassifier(
    random_state=1, n_estimators=50, min_samples_split=4, min_samples_leaf=2)
score = cross_val_score(alg, titanic[predictors], titanic.Survived, cv=3)
print("删除少量特征后的准确率:{accuracy}".format(accuracy=score.mean()))


# 用SVM和Logsitc回归进行预测
predictors = ["Pclass", "Sex", "Fare",
              "titles", "NameLength", "Embarked", "Parch"]
algorithms = [
    [GradientBoostingClassifier(
        random_state=1, n_estimators=25, max_depth=3), predictors],
    [LogisticRegression(random_state=1), predictors]
]
predictions = []

# 运用K折交叉验证测算准确率
Kf = KFold(random_state=1)
for train, test in Kf.split(range(titanic.shape[0])):
    train_target = titanic.Survived.iloc[train]
    full_test_predictions = []
    for alg, predi in algorithms:
        alg.fit(titanic[predi].iloc[train], train_target)
        test_predictions = alg.predict_proba(
            titanic[predi].iloc[test].astype(float))[:, 1]
        full_test_predictions.append(test_predictions)

    test_predictions = (full_test_predictions[
                        0] + full_test_predictions[1] / 2)

    test_predictions[test_predictions <= 0.5] = 0
    test_predictions[test_predictions > 0.5] = 1
    predictions = np.hstack((predictions, test_predictions))

accuracy = 1 - sum(abs(predictions - titanic.Survived)) / predictions.shape[0]
print("SVM与Logistic回归共同作用准确率:{accuracy}".format(accuracy=accuracy))
plt.savefig("score of feature")
plt.show()
