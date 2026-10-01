# -*- coding: utf-8 -*-
"""
Series Temporales - Ejercicio de Evaluación

08_series_modelado_ets.py
 - Modelado de la serie mediante ETS en espacio de estados
 - ExponentialSmoothing vs ETSModel
 - Predicción e intervalos de confianza sobre TEST

Autor: Carlos Ayllón Motto
"""
# Importación de librerías
import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import shutil

from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from series_funciones import (
    cargar_train_test, guardar_pickle, mostrar_resultado, guardar_grafico,
    calcular_metricas_pred, crear_tabla_prediccion_ic, resumen_intervalos,
    grafico_modelo
    )

warnings.filterwarnings('ignore')

# =============================================================================
# 1. RUTAS DE DATOS
# =============================================================================
dir_trabajo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

dir_datos = os.path.join(dir_trabajo, 'outputs')

dir_images = os.path.join(dir_trabajo, 'images')
os.makedirs(dir_images, exist_ok=True)

graficas = os.path.join(dir_images, 'modelado_ets')
if os.path.exists(graficas):
    shutil.rmtree(graficas)
os.makedirs(graficas, exist_ok=True)

dir_out_pickle = os.path.join(dir_datos, 'modelo_est_esp.pkl')

dir_pickle_tt = os.path.join(dir_datos, 'pv_train_test.pkl')

# =============================================================================
# 2. CARGA DE DATOS
# =============================================================================
serie, serie_train, serie_test, m, datos_tt = cargar_train_test(dir_pickle_tt)
n_train = len(serie_train)
n_test = len(serie_test)

# =============================================================================
# 3. MODELADO SUAVIZADO ESPACIO DE ESTADOS
# =============================================================================
# Creación del modelo, libreria ETSModel
# Se configuran los estados igual que el modelo HW + error aditivo
nombre = 'ETS'
modelo_ets = ETSModel(
    serie_train, error = 'add', seasonal_periods=m, trend='add', 
    damped_trend=True, seasonal='mul', initialization_method='estimated'
    )


# =============================================================================
# 4. AJUSTE Y PARÁMETROS
# =============================================================================
res_ets = modelo_ets.fit()
res_ets.summary()

ets_params = pd.Series(
    res_ets.params.values, index = res_ets.param_names, name=nombre
    )

ets_ajuste = pd.Series({
    'AIC': res_ets.aic.round(4),
    'BIC': res_ets.bic.round(4),
    'SSE': res_ets.sse.round(4),
    'Log_Like': res_ets.llf.round(4),
    }, name = nombre)

# =============================================================================
# 5. PREDICCIÓN + ERRORES + IC
# =============================================================================
ets_prediccion = res_ets.get_prediction(
    start = serie_test.index[0], end = serie_test.index[-1],
    index = serie_test.index, simulate_repetitions=10000, rng=505
    )

ic_pred = ets_prediccion.summary_frame()

ets_pred = pd.Series(
    ic_pred['mean_numerical'], index=serie_test.index,
    name=f'prediccion_{nombre}'
    )

tabla_final = crear_tabla_prediccion_ic(
    nombre, serie_test, ets_pred, ic_pred['pi_lower'], ic_pred['pi_upper']
    ).round(4)

resumen_ic_ets = resumen_intervalos(nombre, tabla_final)

ets_metricas=calcular_metricas_pred(nombre, serie_test, ets_pred)
ets_metricas = ets_metricas.rename(ets_metricas['modelo']).drop('modelo')

# 5.1. RESULTADOS DEL MODELO
mostrar_resultado('Ajuste del modelo ETS', ets_ajuste)

mostrar_resultado('Parámetros estimados del modelo ETS', ets_params)

mostrar_resultado('Métricas de predicción ETS', ets_metricas)

mostrar_resultado('Predicción TEST e intervalos ETS', tabla_final)

mostrar_resultado('Resumen de intervalos ETS', resumen_ic_ets)

# =============================================================================
# 6. GRÁFICOS
# =============================================================================
fit_train = res_ets.fittedvalues

serie_total = pd.concat([serie_train, serie_test])
serie_modelo = pd.concat([res_ets.fittedvalues, ets_pred])

residuos_ets = res_ets.resid

# Estados
nivel = res_ets.level
tendencia = res_ets.slope
estac = res_ets.season

# 6.1 Serie observada vs ajuste/predicción
titulo = 'Serie ETSModel vs Observado'
grafico_modelo(
    titulo, serie_train, serie_test, fit_train, ets_pred, carpeta=graficas
    )

# 6.2. Gráfico TEST+Predicción+IC
plt.figure(figsize=(14, 8))
titulo = f'Predicción {nombre} en TEST con intervalos de confianza 95%'

plt.plot(serie_test, label='TEST', color='gold', linewidth = 1.5)
plt.plot(ets_pred, label='Predicción', color='magenta', linewidth = 1.5)
plt.plot(ic_pred['pi_lower'], linestyle='--', label='LCL', color='lime')
plt.plot(ic_pred['pi_upper'], linestyle='--', label='UCL', color='lime')

plt.fill_between(
    serie_test.index, ic_pred['pi_lower'], ic_pred['pi_upper'], alpha=0.15,
    color = 'lime', label='IC95% ETSModel'
    )

plt.title(titulo)
plt.xlabel('Periodo')
plt.ylabel('GWh')
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()

guardar_grafico(titulo, graficas)
plt.show()

# PANEL 2x4 - GRÁFICOS DETALLADOS
#---------------------------------
fig, axes = plt.subplots(4, 2, figsize=(15, 24))

# 1 - Serie total vs ajuste total
axes[0, 0].plot(serie_total, label='Observada', color='gold', linewidth=1.5)
axes[0, 0].plot(
    serie_modelo, label='Ajuste/Predicción', color='magenta', linewidth=1.2
    )
axes[0, 0].set_title('Serie Principal vs Serie Modelo')
axes[0, 0].set_xlabel('Fecha')
axes[0, 0].set_ylabel('GWh')
axes[0, 0].legend()

# 2 - TRAIN: observada vs suavizada (fitted)
axes[0, 1].plot(
    serie_train, label='Train observada', color='lime', linewidth=1.5
    )
axes[0, 1].plot(
    res_ets.fittedvalues, label='Suavizada (fitted)', color='skyblue',
    linewidth=1.2
    )
axes[0, 1].set_title('TRAIN: observada vs suavizada')
axes[0, 1].set_xlabel('Fecha')
axes[0, 1].set_ylabel('GWh')
axes[0, 1].legend()

# 3 - TEST: observado vs predicción
axes[1, 0].plot(
    serie_test, label='Test observado', color='gold', linewidth=1.8
    )
axes[1, 0].plot(ets_pred, label='Predicción', color='magenta', linewidth=1.5)
axes[1, 0].set_title('TEST: observado vs predicción (2025)')
axes[1, 0].set_xlabel('Fecha')
axes[1, 0].set_ylabel('GWh')

# 4 - Nivel
axes[1, 1].plot(nivel, color='teal', linewidth=1.5)
axes[1, 1].set_title('Nivel')
axes[1, 1].set_xlabel('Fecha')
axes[1, 1].set_ylabel('Nivel')

# 5 -  Tendencia
axes[2, 0].plot(tendencia, color='turquoise', linewidth=1.5)
axes[2, 0].set_title('Tendencia')
axes[2, 0].set_xlabel('Fecha')
axes[2, 0].set_ylabel('Tendencia')

# 6 - Estacionalidad
estac.index = pd.to_datetime(estac.index, format = '%Y')
axes[2, 1].plot(estac[-60:], color='royalblue', linewidth=1.5)
axes[2, 1].set_title('Estacionalidad últimos 5 años')
axes[2, 1].set_xlabel('Fecha')
axes[2, 1].set_xticks(estac.index[-60::6])
axes[2, 1].set_xticklabels(estac.index[-60::6].strftime('%Y-%m'), fontsize=8)
axes[2, 1].set_ylabel('Estacional')

# 7 - Residuos TRAIN (diferencia Observada - Suavizada)
axes[3, 0].plot(residuos_ets, color='coral', linewidth=1)
axes[3, 0].axhline(
    0, linestyle='--', color='black', linewidth=1.2, alpha = 0.4
    )
axes[3, 0].set_title('Residuos TRAIN')
axes[3, 0].set_xlabel('Fecha')
axes[3, 0].set_ylabel('Error (GWh)')

# 8 - Histograma residuos TRAIN
axes[3, 1].hist(
    residuos_ets.dropna().values, bins=15, color='wheat', edgecolor='oldlace'
    )
axes[3, 1].set_title('Distribución residuos TRAIN')
axes[3, 1].set_xlabel('Error (GWh)')
axes[3, 1].set_ylabel('Frecuencia')

titulo= f'Panel - {nombre}'
plt.suptitle(titulo, fontsize=14, weight ='bold', y=1)
plt.tight_layout()

guardar_grafico(titulo, graficas)
plt.show()

# =============================================================================
# 7. GUARDADO DEL MODELO
# =============================================================================
salida_ets = {
    'nombre': nombre,
    'modelo': modelo_ets,
    'resultados': res_ets,
    'params': ets_params,
    'metricas': ets_metricas,
    'prediccion': ets_prediccion,
    'valores_prediccion': ets_pred,
    'intervalos': ic_pred,
    'resumen_ic': resumen_ic_ets,
    }

guardar_pickle(salida_ets, dir_out_pickle)