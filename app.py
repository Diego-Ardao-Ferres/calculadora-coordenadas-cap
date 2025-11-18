import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Mis mismas salidas del modelo 2
columnas_output_modelo2 = [
    "Cap_Cap",
    "Dist_B_Prom",
    "SN_Cap_Prom",
    "Cap_T_Prom",
    "Cap_V_Prom",
    "Pezon_Prom",
]

@st.cache_resource
def load_model():
    modelo = joblib.load("modelo_torso_masc.pkl")
    return modelo

modelo = load_model()

def predecir_para_paciente(altura_m, peso_kg, edad, ancho_torax_cm):
    """
    Calcula IMC, arma el vector [Altura(m), Peso, Edad, Ancho Torax, IMC]
    y devuelve dict con las predicciones.
    """
    imc = peso_kg / (altura_m ** 2)
    x_nuevo = np.array([[altura_m, peso_kg, edad, ancho_torax_cm, imc]], dtype=float)
    y_pred = modelo.predict(x_nuevo)[0]

    resultados = {nombre: valor for nombre, valor in zip(columnas_output_modelo2, y_pred)}
    return imc, resultados

# ========= INTERFAZ STREAMLIT =========

st.title("Calculadora de posición del CAP masculino")
st.write("Herramienta de investigación para estimar la ubicación del complejo areola-pezón a partir de medidas antropométricas.")

st.markdown("**⚠️ Uso académico / de investigación. No es una herramienta clínica validada ni un dispositivo médico.**")

col1, col2 = st.columns(2)

with col1:
    altura_m = st.number_input("Altura (m)", min_value=1.20, max_value=2.30, value=1.75, step=0.01)
    peso_kg = st.number_input("Peso (kg)", min_value=40.0, max_value=150.0, value=75.0, step=0.5)

with col2:
    edad = st.number_input("Edad (años)", min_value=15, max_value=80, value=30, step=1)
    ancho_torax_cm = st.number_input("Ancho de tórax (cm)", min_value=25.0, max_value=60.0, value=38.0, step=0.5)

if st.button("Calcular medidas del CAP"):
    imc, resultados = predecir_para_paciente(altura_m, peso_kg, edad, ancho_torax_cm)

    st.subheader("Resultados")
    st.write(f"**IMC calculado:** {imc:.2f} kg/m²")

    df_res = pd.DataFrame(
        [{"Medida": k, "Valor (cm)": v} for k, v in resultados.items()]
    )
    st.table(df_res)
    