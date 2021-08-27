import streamlit as st
import pandas as pd
from feature_engine import transformation as vt
from feature_engine.encoding import OneHotEncoder
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
model_training = st.beta_container()

def get_data(filename):
    data = pd.read_csv(filename)
    
    return data

with header:
    st.title("Credit Card Existing Prediction Model")
    st.text("In this project I looked at a bank dataset.")

with dataset:
    st.header("Credit Card Dataset")
    st.text("I found this dataset on Kaggle")
    data = get_data(r"C:\Users\taylan.polat\Desktop\unl\MEF_Final_Proje\MLProje\BankChurners.csv")
    st.write(data.head())
    
st.subheader("Income_Category")
inc_cat = pd.DataFrame(data["Income_Category"].value_counts())
st.bar_chart(inc_cat)
    
with features:
    st.header("Features All Created")
    
    st.markdown("* **I made this first feat:** Dummies ")
    st.markdown("* **I made this second feat:** Transposed ")

with model_training:
    st.header("Time to train the model:")
    st.text("Here you get to choose the hyperparametres of Model")
    
    selected_col, disp_col = st.beta_columns(2)
    
    max_depth = selected_col.slider("what should be the max depth of the model ?",min_value = 1, max_value = 20, value = 1, step = 1)
    
    n_estimators = selected_col.selectbox('How many trees should there be ?', options = [100,200,300,400,500],index = 0)
    
    min_samples_leaf = selected_col.selectbox('How many samples should there be ?', options = [1,2,5,10,100,200,300,"No limit"],index = 0)
    #input_feature = selected_col.text_input("Which features should be used as the input feature?",'PULocationID')
    
    norm_list = ["Credit_Limit","Months_on_book","Avg_Open_To_Buy","Total_Trans_Amt"]

    tf = vt.LogTransformer(variables = norm_list)
    
    tf.fit(data)
    
    data = tf.transform(data)
    
    encoder = OneHotEncoder( top_categories=1, variables=["Attrition_Flag", "Gender"], drop_last=False)
    encoder.fit(data)
    data = encoder.transform(data)
    encoder = OneHotEncoder( top_categories=3, variables=["Marital_Status", "Card_Category"], drop_last=False)
    encoder.fit(data)
    data = encoder.transform(data)
    dummies = pd.get_dummies(data[["Education_Level","Income_Category"]],drop_first = True)
    data.drop(["Education_Level","Income_Category"],axis = 1, inplace = True)
    data = pd.concat([data,dummies],axis = 1)
    data.drop("CLIENTNUM",axis = 1,inplace = True)
    data = data.rename(columns = {"Attrition_Flag_Existing Customer":"Target"})
    
    data = data.rename(columns = {'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2':'Inactive_12_mon_2',
                                      'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1':'Inactive_12_mon_1'})
    
    X = data.drop("Target",axis = 1)
    y = data["Target"]
    
    clas = GradientBoostingClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, n_estimators=n_estimators)
    
    oversample = SMOTE()

    undersample = RandomUnderSampler()

    steps = [('o', oversample), ('u', undersample)]
    pipeline = Pipeline(steps=steps)

    X, y = pipeline.fit_resample(X, y)
    
    X.drop(["Inactive_12_mon_1","Avg_Open_To_Buy","Card_Category_Silver",
           "Inactive_12_mon_2","Total_Trans_Ct"],axis = 1, inplace = True)
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)
    
    clas.fit(X_train,y_train)
    
    prediction = clas.predict(X_test)

    disp_col.subheader("Roc Score is: ")
    disp_col.write(roc_auc_score(y_test,prediction))
    
    disp_col.subheader("Recall Score is: ")
    disp_col.write(recall_score(y_test,prediction))