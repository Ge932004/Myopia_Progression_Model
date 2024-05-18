import streamlit as st
import pandas as pd
import subprocess
import shap
import matplotlib.pyplot as plt
# 定义要安装的模块名称
module_name = "joblib"

# 使用 subprocess 运行 pip install 命令
subprocess.call(["pip", "install", module_name])

# 重新导入模块
import importlib
importlib.import_module("joblib")

import joblib
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
    class StackingClassifierWrapper:
    def __init__(self, model):
        self.model = model
    def predict(self, X):
        return self.model.predict(X)   
    wrapped_model = StackingClassifierWrapper(clf)
    explainer = shap.Explainer(wrapped_model.predict, X)
    shap_values = explainer(X)

    temp = np.round(x_test, 2)
    shap.force_plot(explainer.expected_value[1], shap_values[1],temp,
         feature_names = ["SE_right", "Age_group", "School","UDVA_group","Vision_correction"], matplotlib=True, show=False)
      plt.xticks(fontproperties='Times New Roman', size=15)
      plt.yticks(fontproperties='Times New Roman', size=20)
      plt.tight_layout()
      plt.savefig("force plot.png",dpi=600) 
      pred = model.predict_proba(X)
      st.markdown("#### _Based on feature values, predicted possibility of myopia progression is {}%_".format(round(pred[0][1], 4)*100))
      st.image('myopia progression force plot.png')
    
    # Get prediction
    prediction = clf.predict(X)[0]
    
    # Output prediction
    st.text(f"This instance is a {prediction}")
