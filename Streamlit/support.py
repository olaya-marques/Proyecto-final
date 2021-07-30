import streamlit as st
import pandas as pd
from PIL import Image
import pickle
from sklearn.ensemble import RandomForestClassifier


def load_data():
    """
    Carga la base de datos necesaria para predecir "Attrition" del modelo usuario"
    Args: 
        no recive parámetros
    Returns:
        La base de datps
    """
    return pd.read_csv("../Data/Usuario.csv",index_col=0)


def user_input_features(data):
    """
    Guarda la información introducida por el usuario para predecir la variable "Attrition"
    Args: 
        no recive parámetros
    Returns:
        Los datos introducidos por el usuario
    """
    return pd.DataFrame(data, index=[0])


#limpieza de datos de usuario
def transformacion(data):
    """
    Transforma los datos introducidos por el usuario para que el modelo lo entienda (misma estructura y forma)
    Args: 
        Datos del usuario
    Returns:
        Los datos transformados y traducidos introducidos por el usuario
    """
    for key,value in data.items():
        if key == "OverTime" and value =="Yes":
            data["OverTime"] = 1
        elif key == "OverTime" and value =="No":
            data["OverTime"] = 0
        elif key == "Gender" and value =="Female":
            data["Gender"] = 1
        elif key == "Gender" and value =="Male":
            data["Gender"] = 0
        elif key == "Education" and value =="Below College":
            data["Education"] = 1
        elif key == "Education" and value =="College":
            data["Education"] = 2
        elif key == "Education" and value =="Bachelor":
            data["Education"] = 3
        elif key == "Education" and value =="Master":
            data["Education"] = 4
        elif key == "Education" and value =="Doctor":
            data["Education"] = 5
    return data


#Creación del modelo
def model(data):
    """
    Función que carga el modelo predictivo para predicir la variable Attrition
    Args: 
        Datos introducidos por el usuario
    Returns:
        Attrition (predicción y probabilidad)
    """
    
    #cargamos los datos del dataset original, para ello empleamos la función load_data(), definida anteriormente
    #transformamos algunos de los datos que nos da el usuario al mismo formato y estructura que como lo tenemos en el dataset original
    data2 = load_data()
    data = transformacion(data)

    #añadimos los datos que nos da el usuario al dataset original que hemos cargado en el paso anterior
    data3 = pd.concat([data2,pd.DataFrame([data])])

    #transformamos las variables categóricas en númericas mediante get_dummies
    data3 = pd.get_dummies(data3)
    
    #establecemos las variables
    x = data3.iloc[:-1].drop(["Attrition"], axis = 1)
    y = data3.iloc[:-1]["Attrition"]

    #construimos el modelo
    rf = RandomForestClassifier()
    
    #entrenamos el modelo
    rf.fit(x, y)
    
    #establecemos los datos del usuario para que los podamos pasar por el modelo predictivo
    data_user = data3.drop("Attrition", axis=1).iloc[-1].values.reshape(1, -1)
 
    #predecimos
    prediction = rf.predict(data_user)[0]
    prediction_proba = rf.predict_proba(data_user)[0, 1]
   
    #transformamos la respuesta de la predicción en un valor que el usuario pueda comprender
    dict_pred = {0 : "No", 
                1 : "Yes"}

    pred = dict_pred[prediction]
    return pred,prediction_proba