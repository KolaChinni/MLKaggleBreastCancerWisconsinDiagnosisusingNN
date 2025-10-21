import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler

def gradient(x,y,w,b):
    m,n=x.shape
    z=np.dot(x,w)+b
    f_wb=sigmoid(z)
    err=f_wb-y
    dj_dw=np.dot(x.T,err)
    dj_db=np.sum(err)
    dj_dw=dj_dw/m
    dj_db=dj_db/m
    return dj_dw,dj_db

def grad_descent(x,y,w,b,iter,alpha):
    for i in range(iter):
        dj_dw,dj_db=gradient(x,y,w,b)
        w=w-alpha*dj_dw
        b=b-alpha*dj_db
    return w,b

def sigmoid(z):
    return 1/(1+np.exp(-z))

def predict(x,w,b):
    z=np.dot(x,w)+b
    preds=sigmoid(z)
    return preds

data=pd.read_csv('MLKaggleBreastCancerWisconsinDiagnosisusingNN\data.csv')
data=data.drop(['id','Unnamed: 32'],axis=1)
data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
x=data.drop(['diagnosis'],axis=1)
y=data['diagnosis']

x=x.to_numpy()
y=y.to_numpy()
scaler=StandardScaler()
x=scaler.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

w=np.zeros(x.shape[1])
b=0.0
iter=1000
alpha=0.01
w,b=grad_descent(x_train,y_train,w,b,iter,alpha)
y_pred=predict(x_test,w,b)
y_pred=(y_pred>=0.5).astype(int)

print(y_pred)
classi_report=classification_report(y_test,y_pred)
confu_matrix=confusion_matrix(y_test,y_pred)
print(classi_report)
print(confu_matrix)