#Import the necessary libraries
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
import streamlit as st
from joblib import load

#Load dataset
iris = datasets.load_iris()
data=pd.DataFrame({
'sepal length': iris.data[:,0],
'sepal width': iris.data[:,1],
'petal length': iris.data[:,2],
'petal width': iris.data[:,3], 
'species': iris.target
})
classes = iris.target_names

#Features selection 
x=data[['sepal length', 'sepal width', 'petal length', 'petal width']] #features
y=data['species']  #target

#Create Random Forest Model
#x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.3,random_state=0) 

#clf=RandomForestClassifier(n_estimators=10)  
#clf.fit(x_train, y_train)  
#y_pred=clf.predict(x_test) 
#print("Accuracy of 10 is : ", round(metrics.accuracy_score(y_test, y_pred),2))
# The best number of estimators is  10 for an accuracy of 98%.

st.title("Iris Dataset Predictor")
st.header("Welcome to the Iris Dataset !")
sepal_length= st.slider("Select the sepal length", 
                        min_value=int(min(data['sepal length'])), 
                        max_value=int(max(data['sepal length'])), 
                        value=int(np.mean(data['sepal length'])))
st.text('Selected: {}'.format(sepal_length)) 

sepal_width= st.slider("Select the sepal width", 
                        min_value=int(min(data['sepal width'])), 
                        max_value=int(max(data['sepal width'])), 
                        value=int(np.mean(data['sepal width'])))
st.text('Selected: {}'.format(sepal_width)) 

petal_length= st.slider("Select the petal length", 
                        min_value=int(min(data['petal length'])), 
                        max_value=int(max(data['petal length'])), 
                        value=int(np.mean(data['petal length'])))
st.text('Selected: {}'.format(petal_length)) 

petal_width= st.slider("Select the petal width", 
                        min_value=int(min(data['petal width'])), 
                        max_value=int(max(data['petal width'])), 
                        value=int(np.mean(data['petal width'])))
st.text('Selected: {}'.format(petal_width)) 


clf = load('irisdata.joblib')
if (st.button('Iris Predictor')):
    classs_id=clf.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    st.write(classes[classs_id][0])
   