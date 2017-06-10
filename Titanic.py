import pandas as pd

# 数据预处理
# 处理缺失值
titanic = pd.read_csv("train.csv")
#print(titanic.describe())  #发现age缺失值
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())  # 用中值填充年龄的缺失值
# print(titanic.describe())

# 字符串类别转换为数字
print(titanic["Sex"].unique())  # 查看类别种类
# print(titanic["Sex"]=="male")
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
# print(titanic["Embarked"].describe()) 看哪个最多 来填充缺失值
# print(titanic.Embarked.unique())
titanic["Embarked"] = titanic["Embarked"].fillna("S")
num = 0
print(titanic.Embarked.unique())
for i in titanic.Embarked.unique():  # 仓型转化为数值
    titanic.loc[titanic.Embarked == i, "Embarked"] = num
    num = num + 1


##################################################
# 线性回归预测
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import numpy as np

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
alg = LinearRegression()
Kf = KFold(random_state=1, n_splits=3)
predictions = []
for train, test in Kf.split(range(titanic.shape[0])):
    train_predictors = (titanic[predictors].iloc[train])
    print(test)
    train_target = titanic["Survived"].iloc[train]
    alg.fit(train_predictors, train_target)
    test_predictions = alg.predict(titanic[predictors].iloc[test])
    predictions = np.hstack((predictions, test_predictions))

predictions[predictions > 0.5] = 1
predictions[predictions < 0.5] = 0
accuracy = 1 - sum(abs(predictions - titanic.Survived)) / predictions.shape[0]
print(accuracy)
