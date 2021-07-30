import streamlit as st
import pandas as pd
from PIL import Image
import pickle
from sklearn.ensemble import RandomForestClassifier
from support import load_data,user_input_features,model

#Título de portada
image = Image.open("./Imagenes/Portada.png")
st.image(image,use_column_width=True)

#Descripción de la web
st.subheader ("""Los recursos humanos tienen un gran impacto en los costes de la empresa, 
pero no todos los gastos de RR. HH. son evidentes a primera vista. Los costes obvios incluyen los gastos en material de oficina y contratación, pero también hay factores no materiales como la rotación y la insatisfacción de los empleados. Traducir estos costes en cifras muestra claramente el valor real de los recursos humanos en términos de coste para la empresa.
""")
st.header("😱 Sabías que...")
st.subheader("""Según un estudio sobre rotación de 2016, cada vez que un empleado se va le cuesta unos 43 000 € a la empresa 
(de los que 6000 corresponden a costes de contratación). Esto equivale aproximadamente al sueldo medio anual de un ejecutivo de marketing.
""")

#Presentación del modelo predictivo
st.header("🧐 ¿Te gustaría saber que empleados tienen mayor probabilidad de abandonar la empresa?")
st.header("¿QUIERES ANTICIPARTE? 🧙🏻‍♀️")
st.subheader("¡Ahora puedes!...🎉¡Es muy sencillo!🎉") 
st.text("Introduce todos los datos y descubre la probabilidad de que tu empleado abandone la empresa.")
image = Image.open("./Imagenes/Modelo predictivo.png")
st.image(image,use_column_width=True)

#Datos para el modelo predictivo
#Datos personales
st.subheader("Personal Data 👤")
col1, col2 = st.beta_columns(2)
with col1:
    Age= int(st.number_input("Age",18,70))
with col2:
    Gender = st.radio(label="Gender", options=("Female", "Male"))

col3, col4 = st.beta_columns(2)
with col3:
   MaritalStatus = st.selectbox(label="Marital Status", options=("Single", "Married", "Divorced"))   
with col4:
    Education = (st.selectbox(label="Education", options=("Bellow College","College","Bachelor","Master","Doctor")))

WorkLifeBalance = int(st.slider("Work-Life Balance", 2, 4, 1))



#Datos trabajo/administración
st.subheader("Job/Administration Data 💼")
Department = st.selectbox(label="Department", options=("Sales", "Research & Development", "Human Resources"))

col5, col6 = st.beta_columns(2)
with col5:
    JobRole= st.selectbox(label="Job Role", options=("Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Other"))
with col6:
    DistanceFromHome = int(st.number_input("Distance from home (km)",0,40))

col7, col8, col9 = st.beta_columns(3)
with col7:
    NumCompaniesWorked = int(st.number_input("Number companies worked",0,40))
with col8:
    YearsInCurrentRole = int(st.number_input("Years in current job",0,40))
with col9:
    YearsAtCompany = int(st.number_input("Years at company",0,40))



#Datos trabajo
st.subheader("Work Data 💰")
col10, col11 = st.beta_columns(2)
with col10:
    OverTime = st.radio(label="OverTime", options=("Yes", "No"))
with col11:
    BusinessTravel = st.selectbox (label="Business travel", options=("Travel_Rarely", "Travel_Frequently", "Non-Travel"))

col12, col13 = st.beta_columns(2)
with col12:
    MonthlyIncome = int(st.number_input("Monthly income",0,100000))
with col13:
    PercentSalaryHike = int(st.number_input("Percent salary hike",0,40))



#Datos satisfación
st.subheader("Satisfaction Data 🤩")
JobSatisfaction = int(st.slider("Job Satisfaction", 2, 4, 1))
EnviromentSatisfaction = int(st.slider("Environment Satisfaction", 2, 4, 1))
RelationshipSatisfaction = int(st.slider("Relationship Satisfaction", 2, 4, 1))

#Confirmar los datos del usuario
data = { "Age": Age,
    "BusinessTravel" : BusinessTravel,
    "Department" : Department,
    "DistanceFromHome" : DistanceFromHome,
    "Education" : Education,
    "EnvironmentSatisfaction" : EnviromentSatisfaction,
    "Gender": Gender,
    "JobRole" : JobRole,
    "JobSatisfaction": JobSatisfaction,
    "MaritalStatus": MaritalStatus,
    "MonthlyIncome": MonthlyIncome,
    "NumCompaniesWorked" : NumCompaniesWorked,
    "OverTime" : OverTime,
    "PercentSalaryHike" : PercentSalaryHike,
    "RelationshipSatisfaction" : RelationshipSatisfaction,
    "WorkLifeBalance" : WorkLifeBalance,
    "YearsAtCompany": YearsAtCompany,
    "YearsInCurrentRole": YearsInCurrentRole
    }

#Botón de predicción
if st.button("Predict 🔮"):
    pred = model (data)
    st.write(pred) 

#Cierre
image = Image.open("./Imagenes/Cierre.png")
st.image(image,use_column_width=True)