import streamlit as st
import keras
import pandas as pd

st.logo('https://cdn.ips.gob.cl/files/shares/logo-ipscha-color.svg', size='large')
st.sidebar.title('Modelo de predicción NOCobro en Deep Learning')

# Cargar el modelo
@st.cache_resource()
def load_model():
    return keras.models.load_model('../models/modelo.keras')

def cargar_archivo():
    archivo = st.file_uploader('Cargar archivo CSV', type='csv')
    if archivo is not None:
        return pd.read_csv(archivo)



# Código principal de la aplicación
model = load_model()
st.header('Cargue un archivo CSV para realizar predicciones')
datos = cargar_archivo()
st.write('Seleccione el nivel de sensibilidad del modelo')
sensibilidad = st.slider('Probabilidad de NO Cobro', min_value=0.0, max_value=1.0, value= 0.7,
                         help='Ejemplo: Si selecciona 0.7, el modelo definirá como NO Cobro a todas las predicciones'
                              ' con una probabilidad mayor a 0.7')

if datos is not None:
    predicciones = model.predict(datos)
    # Usar la sensibilidad para definir si es cobro o no
    nocobro = ['No cobrará pensión' if p[0] >= sensibilidad else 'Cobrará pensión' for p in predicciones]
    st.write('Predicciones:')
    st.dataframe(pd.DataFrame({'Probabilidad de NO Cobro': predicciones[:, 0], 'Predicción': nocobro}))
