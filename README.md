# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries: import pandas as pd for data manipulation, from sklearn.linear_model import LogisticRegression for logistic regression modeling, and other relevant libraries for evaluation metrics.
2.Prepare the data: Load your dataset, perform any necessary preprocessing such as dropping irrelevant columns, encoding categorical variables using LabelEncoder, and splitting the data into training and testing sets.
3.Train the logistic regression model: Instantiate a LogisticRegression object, fit it to the training data using lr.fit(x_train, y_train).
4.Make predictions: Use the trained model to predict the target variable for new data points. For example, lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]]) would predict the status based on the given input features.
  
  


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: BALAJEE K.S
RegisterNumber:  212222080009
import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy Score:",accuracy)

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print("\nConfusion Matrix:\n",confusion)


from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print("\nClassification Report:\n",classification_report1)

from sklearn import metrics
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=[True,False])
cm_display.plot()
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
```

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)
![Screenshot (448)](https://github.com/balajeeakm/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131589871/c69d99a8-079a-4b09-8917-dc6243bae099)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
