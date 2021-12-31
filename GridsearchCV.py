# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 14:39:35 2021

@author: hongyong Han

To: Do or do not. There is no try.

"""

#############################  分类算法快速筛选   #############################

import time  #为了计算程序运行的时间
start =time.perf_counter()
import warnings
warnings.filterwarnings("ignore") 

#导入工具包
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import Pipeline   #导入管道模型
from numpy import random

#导入算法包
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
 
#加载数据
data=np.loadtxt(r'C:\Users\Uaena_HY\Desktop\代码集\Testset\classifier.txt',delimiter='\t') #读入txt数据，以tab为分界
#数据集打乱
data=np.random.permutation(data)
#分输入输出数据
x = []
y = []
for line in range(76):
    x.append(data[line][:2])
    y.append(data[line][-1])
X = np.array(x)
y = np.array(y)

#模型粗选 SVC KNN RF GBC
cv=10

# SVC
pipe1= Pipeline([('scaler',StandardScaler()),('svc',SVC())])
params_1 =[ {'svc__kernel':['rbf','linear','sigmoid', 'precomputed'],
            'svc__C':random.uniform(0.1,100,size=50), 
            'svc__gamma':random.uniform(1e-3,0.1,size=50)},
            {'svc__kernel':['poly'],'svc__C':random.uniform(0.1,100,size=50),
            'svc__gamma':random.uniform(1e-6,0.1,size=50),'svc__degree':[3,4,5]} ]
grid_1 = GridSearchCV (pipe1,params_1,cv=cv,scoring='accuracy')
                       
grid_1.fit(X,y)

## KNN
pipe2= Pipeline([('scaler',StandardScaler()),('knn',KNeighborsClassifier())])
params_2 = {'knn__n_neighbors':[2,3,4,5,6,7,8,9,10]} 
grid_2 = GridSearchCV(pipe2,params_2,cv=cv,scoring='accuracy')
                        
grid_2.fit(X,y)

## RF
params_3 = {'n_estimators':[100,200,500,1000],'max_depth':[3,4,5,6,7,8,9,10]}
grid_3 = GridSearchCV (RandomForestClassifier(),params_3,cv=cv,scoring='accuracy')
grid_3.fit(X,y)

## GBC
params_4 = {'n_estimators':[100,200,500,1000],'max_depth':[3,4,5,6,7,8,9,10]}
grid_4 = GridSearchCV (GradientBoostingClassifier(),params_4,cv=cv,scoring='accuracy')
grid_4.fit(X,y)

print('\n')
print('SVM 模型最高分:{:.3f}'.format(grid_1.best_score_))
print('Ridge 最优参数：{}'.format(grid_1.best_params_))
print('\n')
print('KNN 模型最高分:{:.3f}'.format(grid_2.best_score_))
print('KNN 最优参数：{}'.format(grid_2.best_params_))
print('\n')
print('RF 模型最高分:{:.3f}'.format(grid_3.best_score_))
print('RF 最优参数：{}'.format(grid_3.best_params_))
print('\n')
print('GB 模型最高分:{:.3f}'.format(grid_4.best_score_))
print('GB 最优参数：{}'.format(grid_4.best_params_))