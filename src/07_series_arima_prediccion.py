# -*- coding: utf-8 -*-
"""
Series Temporales - Ejercicio de Evaluación

07_series_arima_prediccion.py
 - Tabla de predicciones
 - Intervalos de confianza
 - Representación gráfica

Autor: Carlos Ayllón Motto
"""
# Importación de librerías
import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import shutil

warnings.filterwarnings('ignore')

from series_funciones import (
    cargar_train_test, cargar_pickle, mostrar_resultado, guardar_grafico,
    crear_tabla_prediccion_ic
    )

# =============================================================================
# 1. RUTAS DE DATOS
# =============================================================================
dir_trabajo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

dir_datos = os.path.join(dir_trabajo, 'outputs')

dir_images = os.path.join(dir_trabajo, 'images')
os.makedirs(dir_images, exist_ok=True)

graficas = os.path.join(dir_images, 'arima_prediccion')
if os.path.exists(graficas):
    shutil.rmtree(graficas)
os.makedirs(graficas, exist_ok=True)

dir_pickle_tt = os.path.join(dir_datos, 'pv_train_test.pkl')

dir_pickle_arima = os.path.join(dir_datos, 'modelo_arima_seleccion.pkl')

# =============================================================================
# 2. CARGA DE DATOS
# =============================================================================
# 2.1. TRAIN/TEST
serie, serie_train, serie_test, m, datos_tt = cargar_train_test(dir_pickle_tt)

# 2.2 MODELO ARIMA SELECCIONADO
datos_arima = cargar_pickle(dir_pickle_arima)

nombre_ari = datos_arima['nombre']

resultados = datos_arima['modelo']
    
# =============================================================================
# 3. PREDICCIÓN EN TEST + INTERVALOS DE CONFIANZA
# =============================================================================
n_test = len(serie_test)

prediccion = resultados.get_forecast(steps=n_test)

pred_ari = prediccion.predicted_mean
pred_ari = pd.Series(pred_ari.values,index=serie_test.index,name='Prediccion')

ic_test = prediccion.conf_int() # default = 95%
ic_test.columns = ['inf','sup']

# =============================================================================
# 4. TABLA PREDICCIÓN - ERROR - IC
# =============================================================================
tabla_prediccion = crear_tabla_prediccion_ic(
    nombre_ari, serie_test, pred_ari, ic_test['inf'], ic_test['sup']
    ).round(4)

tabla_pred = tabla_prediccion[[
    'observado','prediccion','error'
    ]].copy().round(3)

tabla_ic = tabla_prediccion[['prediccion', 'ic_inf', 'ic_sup']].copy().round(3)

mostrar_resultado(f'Predicción TEST - {nombre_ari}', tabla_pred)
mostrar_resultado(f'Intervalos de confianza 95% - {nombre_ari}', tabla_ic)

# =============================================================================
# 5. GRÁFICOS: TEST (REAL) + PRED + BANDAS
# =============================================================================
fecha = serie_test.index
# 5.1. Gráfico con serie completa + Predicción
plt.figure(figsize=(14, 8))
titulo = f'Predicción {nombre_ari} vs Serie TRAIN/TEST'

plt.plot(serie_test, label='TEST', color='gold', linewidth = 1.5)
plt.plot(serie_train, label='TRAIN', color='skyblue', linewidth = 1.5)
plt.plot(pred_ari, label='Predicción', color='magenta', linewidth = 1.5)

plt.axvline(x=fecha.min(),linestyle='--',color='black',linewidth = 2)
plt.title(titulo)
plt.xlabel('Periodo')
plt.ylabel('GWh')
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()

guardar_grafico(titulo, graficas)
plt.show()

# 5.2. Gráfico TEST+Predicción+IC
plt.figure(figsize=(14, 8))
titulo = f'Predicción {nombre_ari} en TEST con intervalos de confianza 95%'

plt.plot(serie_test, label='TEST', color='gold', linewidth = 1.5)
plt.plot(pred_ari, label='Predicción', color='magenta', linewidth = 1.5)
plt.plot(ic_test['inf'], linestyle='--', label='LCL', color='limegreen')
plt.plot(ic_test['sup'], linestyle='--', label='UCL', color='lime')

plt.title(titulo)
plt.xlabel('Periodo')
plt.ylabel('GWh')
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()

guardar_grafico(titulo, graficas)
plt.show()