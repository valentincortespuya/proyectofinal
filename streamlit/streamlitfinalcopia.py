import pandas as pd
import streamlit as st
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import matplotlib.pyplot as plt
import plotly.express as px
st.set_page_config(layout = 'wide', initial_sidebar_state = 'collapsed', page_title = '', page_icon = '')


# Cargar el DataFrame o ajusta la ruta a tus datos
df = pd.read_csv("limpio_sin_outliers.csv", parse_dates=["Mes"])

st.title("Proyecto Final")
st.subheader('ETL+EDA', divider='rainbow')
#st.subheader('seguro que cambias algo tras el  :blue[análisis] :sunglasses:')
col5, col6 = st.columns(2)
with col5:
    st.write("Os presento a Ale")

    imagen_local = "ale.jpg"
    anchura_original = 800  # Reemplaza con el ancho original de tu imagen en píxeles
    nueva_anchura = int(anchura_original * 0.3)

    # Mostrar la imagen con la nueva anchura
    st.image(imagen_local, caption="Alejandra Rodríguez ", width=nueva_anchura)
with col6:
     st.write("Os presento a Ale. Esta preciosa mujer es Licenciada en Periodismo, Máster en Periodismo y Máster en Maketing Digital. A la vez es madre de dos niños y lleva a mi lado desde el año 2000. Desde 2007 mantenemos una economía complentamente compartida, muy a lo tuyo en mio y lo mio tuyo. Y desde SIEMPRE, pese a ser yo economista, las cuentas las ha llevado ella. ")
     st.write("En 2012 abrimos una cuenta en un banco tras mi salida de Bankia y desde entonces todo ha entrado y salido desde alli. Pero claro, comprender el funcionamiento de nuestra cuenta familiar es algo que sólo los expertos en análisis de datos y grandes profesionales de la ciencia pueden intentar entender y desde luego Ale es la única que lo entiende.")
     st.write("Hoy tras mi paso por el bootcamp y gracias a la ayuda de Dios he descifrado el ENIGMA")
st.subheader('Y qué teníamos al principio?', divider='rainbow')
st.write("Literalmente teníamos esto, muchos PDF con muchos datos cada uno de ellos, cada cual más maravillosamente distinto")
col7, col8, col9 = st.columns(3)
with col7:
    imagen_local2 = "foto1.png"
    st.image(imagen_local2, caption="pdfs llenos de movimientos", width=nueva_anchura)
with col8:
    imagen_local3 = "foto2.png"
    st.image(imagen_local3, caption="movimientos tipo", width=nueva_anchura)
with col9:
    imagen_local4 = "foto3.png"
    st.image(imagen_local4, caption="MAMA MIA", width=nueva_anchura)

st.write("Con todo este follón os podeis imaginar la ETL!! divertida y con datos chulos.  ")
imagen_local5 = "image.png"
nueva_anchura2 = int(anchura_original * 0.6)
st.image(imagen_local5, caption="si esto de ser 4 y el ipc....", width=nueva_anchura2)
# Mostrar los gráficos en Streamlit
st.title("Previsión de gasto de la familia Cortés Rodríguez")

# Agregar un párrafo explicativo
st.write("En esta aplicación, puedes explorar la previsión de gastos de la familia Cortés Rodríguez. Selecciona una columna de gastos en el menú desplegable a continuación y explora la previsión, porcentajes y más.")


# Crear un widget selectbox para que el usuario elija una columna
columna_elegida = st.selectbox("Selecciona una columna:", df.columns[1:])
col1, col2 = st.columns(2)

# Usuario 1
with col1:

    # Sumar la columna especificada por columna_elegida
    suma_columna = df[columna_elegida].sum()
    num_filas = len(df)

    # Sumar todas las filas de todas las columnas excepto la primera
    suma_filas = df.iloc[:, 1:].sum().sum()

    # Calcular el porcentaje
    porcentaje = (suma_columna / suma_filas)*100

    media_columna = suma_columna / num_filas
    # Mostrar el porcentaje medio total en una caja
    
    st.text(f"El gasto total fue de: {suma_columna}€")
    
    # Calcular el mes con el gasto máximo
    # Filtra el DataFrame para que solo contenga filas correspondientes al año 2022
    df_2022 = df[df["Mes"].dt.year == 2022]

    # Calcula la media de la columna elegida para el año 2022
    media_2022 = df_2022[columna_elegida].mean()

    # Muestra la media con dos decimales
    st.text(f"El gasto medio en {columna_elegida} en año 2022 fue de: {media_2022:.2f}€")

   
with col2:
    # Mostrar la fecha en la que ocurrió el mes máximo
    valor_maximo = df[columna_elegida].max()
    st.text(f"El valor maximo desde 2012 fue de: {valor_maximo}€")
    st.text(f"El gasto medio desde 2012 fue de: {media_columna:.2f}€")
######################################################

import plotly.express as px

# Asegúrate de que la columna 'Mes' esté en formato datetime
df['Mes'] = pd.to_datetime(df['Mes'])

# Ordena el DataFrame por la columna 'Mes' para tener un orden adecuado en el gráfico
df = df.sort_values(by='Mes')

# Crea un gráfico interactivo con Plotly
fig = px.line(df, x='Mes', y=columna_elegida, title=f'{columna_elegida} por Mes')
fig.update_traces(mode='markers+lines')
fig.update_xaxes(title_text='Mes')
fig.update_yaxes(title_text=columna_elegida)

# Muestra el gráfico en Streamlit
st.plotly_chart(fig)

###################################

# Agregar un párrafo explicativo
st.write("El gráfico a continuación muestra las previsiones de gasto basadas en datos históricos. Utilizamos Facebook Prophet, una herramienta de pronóstico de series temporales, para obtener estimaciones precisas. Explora las tendencias, patrones y proyecciones para tomar decisiones informadas sobre futuros gastos familiares. El modelo ha sido entrenado con los datos hasta 2022.")

año_2023 = df[df['Mes'].dt.year == 2023]


df = df.loc[df['Mes'].dt.year != 2023]


# Filtrar el DataFrame para la columna seleccionada
columna_df = df[["Mes", columna_elegida]].copy()
columna_df = columna_df.rename(columns={'Mes': 'ds', columna_elegida: 'y'})
columna_df['ds'] = pd.to_datetime(columna_df['ds'], format='%b-%Y') + pd.offsets.MonthBegin(0)

# Ajustar el modelo Prophet
m = Prophet()
m.fit(columna_df)

# Crear un DataFrame futuro
future = m.make_future_dataframe(periods=4, freq='M')

# Realizar la predicción
forecast = m.predict(future)

forecast_train = forecast.copy()
forecast_train["ds"] = forecast_train["ds"].apply(lambda x: x.strftime("%Y-%m-%d"))


# Filtra las fechas en el rango de enero a octubre de 2023
df1_filtered = año_2023[(año_2023['Mes'] >= '2023-01-01') & (año_2023['Mes'] <= '2023-10-31')]
df2_filtered = forecast_train[(forecast_train['ds'] >= '2023-01-01') & (forecast_train['ds'] <= '2023-10-31')]

# Asegúrate de que ambos DataFrames tengan la misma longitud
min_len = min(len(df1_filtered), len(df2_filtered))
df1_filtered = df1_filtered.iloc[:min_len]
df2_filtered = df2_filtered.iloc[:min_len]

df1_filtered.reset_index(inplace=True, drop=True)
df2_filtered.reset_index(inplace=True, drop=True)


# Combina los DataFrames en uno nuevo usando las columnas de fechas filtradas
result_df = df1_filtered[[columna_elegida]].merge(df2_filtered[['yhat']], left_index=True, right_index=True)

# Puedes ajustar el nombre de las columnas como desees
result_df.columns = ['real', 'test']


from sklearn.metrics import mean_squared_error
import numpy as np


# Asegúrate de que ambos conjuntos de datos tengan la misma longitud
real_values = result_df['real'].dropna()
predicted_values = result_df['test'].dropna()

# Calcular el MSE (Mean Squared Error)
mse = mean_squared_error(real_values, predicted_values)

# Calcular el RMSE (Root Mean Squared Error)
rmse = np.sqrt(mse)

# Mostrar el gráfico principal
st.plotly_chart(plot_plotly(m, forecast))

# Mostrar los componentes
st.plotly_chart(plot_components_plotly(m, forecast))

# Calcular el valor máximo y mínimo de 'yhat'
max_valor = forecast['yhat'].max()
min_valor = forecast['yhat'].min()
med_valor = forecast['yhat'].mean()
# Mostrar los valores máximos y mínimos

border_style = "border: 2px solid blue; padding: 5px; border-radius: 5px;"


st.markdown(f'<div style="{border_style}">Valor máximo de la predicción:                                     {max_valor:.2f}</div>', unsafe_allow_html=True)
st.markdown(f'<div style="{border_style}">Valor medio de la predicción:                                      {med_valor:.2f}</div>', unsafe_allow_html=True)
st.markdown(f'<div style="{border_style}">Valor mínimo de la predicción:                                     {min_valor:.2f}</div>', unsafe_allow_html=True)
st.markdown(f'<div style="{border_style}">El error RMSE es de:                                     {rmse:.2f}</div>', unsafe_allow_html=True)
diferencia=(rmse/media_columna)*100
st.markdown(f'<div style="{border_style}">El error supone un {diferencia:.2f}% del valor medio real</div> ', unsafe_allow_html=True)

comparacion = df1_filtered[["Mes",columna_elegida]]

col3, col4 = st.columns(2)

# Usuario 1
with col3:


    df2_filtered
with col4:
    comparacion

st.title("Próximos pasos")
st.subheader('ETL to the max!!', divider='rainbow')
#############################################################################################################################################################################################
st.write("Lo primero conseguir que mi princesa adquiera una estructura a la hora de introducir datos. Por otro lado, en la ETL, a la hora de agrupar las casi 3000 categorías, los errores han podido ser grándísmos por lo que trabajar más despacio esta parte es muy interesante. ")
st.write("Lo segundo, y lo más importante, es si no me he preocupado en los 16 años pasados de cómo van nuestras finanzas, para que preocuparme ahora")
st.write("Empezar con un proyecto en mi empresa para establecer modelos de predicción de morosidad: pero tranquilito y sin timeline")
st.title("GRACIAS")
st.subheader('Y besos y abrazos!!', divider='rainbow')
st.write(" Gracias profes: por el año y medio que en el que trabajé con vosotros y por este medio año más en el que me habeis llevado a un nuevo mundo. Sois increibles y os admiro profundamente. Yo soy lo que soy gracias a mis maestros y hoy, esos sois vosotros GRACIAS")
st.write(" Gracias a mis compis: Dilan, Stephy, Juan, David y Vero: me llevo un poquito de todos vosotros. ")
