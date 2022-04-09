# from sklearn import feature_selection
# import streamlit as st
# import pandas as pd
# import seaborn as sns
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
# header=st.container()
# data_sets=st.container()
# features=st.container()
# model_training=st.container()
# with header:
#     st.title('Titanic App')
#     st.text('Titanic Data')
# with data_sets:
#     st.header('Titanic destroyed ')
#     df=sns.load_dataset('titanic')
#     df=df.dropna()
#     st.write(df.head())
#     st.subheader('Sex, class and age SubHeader')
#     st.bar_chart(df['sex'].value_counts())
#     st.bar_chart(df['class'].value_counts())
#     st.bar_chart(df['age'].sample())
# with features:
#     st.header('App Features')
#     st.markdown('1.**Features**')
# with model_training:
#     st.header('What happened to titanic ')
#     input,display=st.columns(2)
#     max_depth=input.slider('How Much',min_value=10,max_value=100,value=20,step=5)
# n_estimators=input.selectbox('How many Tress in Random Forest',options=[50,100,200,300,'NO Limit'])
# input.write(df.columns)
# input_features=input.text_input('Which Feature we can use')
# model=RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators)
# if n_estimators=='No  Limit':
#     random_r=RandomForestRegressor(max_depth=max_depth)
# else:
#     random_r=RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators)
# X=df[[input_features]]
# y=df[['fare']]
# model.fit(X,y)
# pred=model.predict(y)
# display.subheader('Mean Absolute Error Of Model')
# display.write(mean_absolute_error(y,pred))
# display.subheader('Mean Squared Error Of Model')
# display.write(mean_squared_error(y,pred))  
# display.subheader('R Squared Score Of Model')
# display.write(r2_score(y,pred))
import streamlit as st
import seaborn as sns
st.header('Hello World')
st.text('Hello World')
df=sns.load_dataset('iris')
st.write(df['petal_length'].head())
