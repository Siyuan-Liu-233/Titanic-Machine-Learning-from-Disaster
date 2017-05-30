from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
from sklearn.model_selection import cross_val_score
import re

##数据预处理
##处理缺失值
titanic=pd.read_csv("train.csv")
#print(titanic.describe())  #发现age缺失值
titanic["Age"]=titanic["Age"].fillna(titanic["Age"].median()) #用中值填充年龄的缺失值
#print(titanic.describe()) 

##字符串类别转换为数字
print(titanic["Sex"].unique()) #查看类别种类
#print(titanic["Sex"]=="male")
titanic.loc[titanic["Sex"]=="male","Sex"]=0
titanic.loc[titanic["Sex"]=="female","Sex"]=1
#print(titanic["Embarked"].describe()) 看哪个最多 来填充缺失值
#print(titanic.Embarked.unique()) 
titanic["Embarked"]=titanic["Embarked"].fillna("S")
num=0
print(titanic.Embarked.unique())
for i in titanic.Embarked.unique():     #仓型转化为数值
    titanic.loc[titanic.Embarked==i,"Embarked"]=num
    num=num+1  

predictoos=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","titles","FamiliySize","NameLength"]
titanic["FamiliySize"]=titanic["SibSp"]+titanic["Parch"]
titanic["NameLength"]=titanic["Name"].apply(lambda x:len(x))
title_mapping={"Mr":1,"Miss":2,"Mrs":3,"Master":4,"Dr":5,"Rev":6,"Major":7,"Col":7,"Mlle":8,"Mme":8,"Don":9,"Lady":9,"Countess":9,"Capt":10,"Ms":10,"Sir":10,"Jonkheer":7}
def get_title(name):
	title_search=re.search('([A-Z,a-z]+)\.',name)
	if title_search:
		return title_search.group(1) 
	else:
		return ''
titles=titanic["Name"].apply(get_title)
#print(titles)
for k,v in title_mapping.items():
	titles[titles==k]=v

#print(pd.value_counts(titles))
titanic["titles"]=titles
alg=ExtraTreesClassifier(random_state=1,n_estimators=50,min_samples_split=4,min_samples_leaf=2)
score=cross_val_score(alg,titanic[predictoos],titanic.Survived,cv=3)
print(score.mean())