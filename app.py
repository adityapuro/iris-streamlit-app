import streamlit as st
from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import plotly.express as px
import joblib

iris = load_iris()
data = pd.DataFrame(iris.data, columns = iris.feature_names)
data['label'] = iris.target

st.title("App")

st.table(data.head(5))
sepal_width = st.number_input("Enter the sepal width: ")
sepal_length = st.number_input("Enter the sepal length: ")
petal_width = st.number_input("Enter the petal width: ")
petal_length = st.number_input("Enter the petal length: ")

dataset = dict()

dataset["sepal length (cm)"]= sepal_length
dataset["sepal width (cm)"]= sepal_width
dataset["petal length (cm)"]= petal_length
dataset["petal width (cm)"]= petal_width

st.write(dataset)


if st.button("Predict"):
    new = pd.DataFrame([dataset])
    st.table(new)

    model = joblib.load("nb.pkl")
    pred = model.predict(new)
    # st.write(pred)
    if pred == 0:
        st.write("Setosa")
    elif pred == 1:
        st.write("Versicolor")
    elif pred == 2:
        st.write("Virginica") 
    else:
        st.error("Not a valid response")       

    


