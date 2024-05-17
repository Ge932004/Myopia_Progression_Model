import streamlit as st
import pandas as pd
import subprocess
import joblib
# 定义要安装的模块名称
module_name = "joblib"

# 使用 subprocess 运行 pip install 命令
subprocess.call(["pip", "install", module_name])

clf = joblib.load("clf.pkl")

# Title
st.header("Streamlit Machine Learning App")

# Input bar 1
SE_right = st.number_input("SE_right")

# Input bar 2
Age_group = st.number_input("Age_group")


# Input bar 3
School = st.number_input("School")

# Input bar 4
UDVA_group = st.number_input("UDVA_group")

# Input bar 4
Vision_correction = st.number_input("Vision_correction")

# If button is pressed
if st.button("Submit"):
    
    # Store inputs into dataframe
    X = pd.DataFrame([[SE_right, Age_group, School, UDVA_group, Vision_correction]], 
                     columns = ["SE_right", "Age_group", "School","UDVA_group","Vision_correction"])
    
    # Get prediction
    prediction = clf.predict(X)[0]
    
    # Output prediction
    st.text(f"This instance is a {prediction}")
