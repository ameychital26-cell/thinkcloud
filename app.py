import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.title('Tips Application')

df = sns.load_dataset('tips')

#st.subheader('Dataset')
#st.write(df)

st.subheader('Plots')
x = st.selectbox('X-Axis',df.columns)
y = st.selectbox('Y-Axis',df.columns)

st.scatter_chart(df[[x,y]])

x_axis = df[['total_bill']]
y_axis = df['tip']

x_train, x_test, y_train,y_test = train_test_split(x_axis,y_axis)
model = LinearRegression()
model.fit(x_train,y_train)

amount = st.slider('Bill Amount',5,50)
pred = model.predict([[amount]])

if st.button('Predict'):
    st.success(f'Predicted Tip : $ {pred}')