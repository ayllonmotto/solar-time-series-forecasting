# -*- coding: utf-8 -*-
"""
Series Temporales - Ejercicio de Evaluación

04_series_suavizado.py
 - Modelo de suavizado exponencial más adecuado
 - Tabla métricas TRAIN/TEST
 - Tabla parámetros estimados (por modelo)
 - Gráfico por modelo: observada + suavizada + predicciones TEST
 - Tabla: predicciones en TEST (modelo ganador)

Autor: Carlos Ayllón Motto
"""
# Importación de librerías
import os
import pandas as pd
import matplotlib.pyplot as plt
import shutil

from statsmodels.tsa.api import ExponentialSmoothing, Holt

from series_funciones import (
    cargar_train_test, guardar_pickle, guardar_grafico, grafico_modelo, 
    calcular_metricas_pred, crear_tabla_prediccion, mostrar_resultado
    )

# =============================================================================
# 1. RUTAS DE DATOS
# =============================================================================

dir_trabajo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

dir_datos = os.path.join(dir_trabajo, 'outputs')

dir_images = os.path.join(dir_trabajo, 'images')
os.makedirs(dir_images, exist_ok=True)

graficas = os.path.join(dir_images, 'suavizado')
if os.path.exists(graficas):
    shutil.rmtree(graficas)
os.makedirs(graficas, exist_ok=True)

dir_pickle_tt = os.path.join(dir_datos, 'pv_train_test.pkl')

dir_out_pickle = os.path.join(dir_datos, 'modelo_suavizado_seleccion.pkl')

# =============================================================================
# 2. CARGA DE DATOS (PICKLE TRAIN/TEST)
# =============================================================================
serie, serie_train, serie_test, m, datos_tt = cargar_train_test(dir_pickle_tt)
n_test = len(serie_test)  # periodo de predicción

# =============================================================================
# 3. FUNCIONES AUXILIARES
# =============================================================================
def metricas_bondad(nombre: str, serie_train, fitted_train, serie_test, predic,
                    modelo_fit) -> pd.Series():
    """
    Métricas calculadas:
    - TRAIN: serie_train vs fitted_train
    - TEST : serie_test  vs pred_test
    - Incluye AIC/BIC/SSE del modelo.
    Almacena los datos en un DF
    """
    #TRAIN
    met_train = calcular_metricas_pred(nombre, serie_train, fitted_train)
    #TEST
    met_test = calcular_metricas_pred(nombre, serie_test, predic)

    metricas = {
        'Modelo': nombre,
        'AIC': modelo_fit.aic,
        'BIC': modelo_fit.bic,
        'SSE': modelo_fit.sse,
        'MAE_train': met_train['MAE'],
        'RMSE_train': met_train['RMSE'],
        'MAE_test': met_test['MAE'],
        'RMSE_test': met_test['RMSE'],
        'MAPE_test%': met_test['MAPE_%'],
        }
    return metricas
   
# =============================================================================
# 4. MODELOS SIN ESTACIONALIDAD (REFERENCIA DE COMPARACIÓN)
# =============================================================================
compara_m = [] # Lista para comparar las medidas
compara_p = [] # Lista para comparar las medidas

# 4.1 MODELO 1: (A,N) HOLT LINEAL

modelo_holt = Holt(serie_train, initialization_method='estimated').fit()
nombre_holt = 'Holt (A,N)'
fit_train_holt = modelo_holt.fittedvalues
pred_holt = modelo_holt.forecast(n_test)

# Tablas parámetros y prediccion (Holt)
params_holt = modelo_holt.params_formatted
tabla_pred_holt = crear_tabla_prediccion(nombre_holt, serie_test, pred_holt)
compara_p.append(tabla_pred_holt)
tabla_pred_holt_2m = tabla_pred_holt.asfreq('2MS').round(2)

# Bondad de ajuste
metricas_holt = metricas_bondad(nombre_holt, serie_train, fit_train_holt,
                                serie_test, pred_holt, modelo_holt)
compara_m.append(metricas_holt)

# Gráfico (Holt)
titulo = 'Modelo Holt Lineal (A,N)' 
grafico_modelo(
    titulo, serie_train, serie_test, fit_train_holt, pred_holt, 
    carpeta = graficas
    )

# 4.2. MODELO 2: (Add,N) HOLT AMORTIGUADO (damped/amortig.)

modelo_holt_am = Holt(
    serie_train, damped_trend=True, initialization_method='estimated'
    ).fit()

nombre_holt_a = 'Holt_amort (Add,N)'
fit_train_holt_am = modelo_holt_am.fittedvalues
pred_holt_am = modelo_holt_am.forecast(n_test)

# Tablas parámetros y prediccion (Holt Amort)
params_holt_am = modelo_holt_am.params_formatted
tabla_pred_holt_am = crear_tabla_prediccion(
    nombre_holt_a, serie_test, pred_holt_am
    )
compara_p.append(tabla_pred_holt_am)

tabla_pred_holt_am_2m = tabla_pred_holt_am.asfreq('2MS').round(2)

# Bondad de ajuste
metricas_holt_am = metricas_bondad(
    nombre_holt_a, serie_train, fit_train_holt_am, serie_test,
    pred_holt_am, modelo_holt_am
    )
compara_m.append(metricas_holt_am)

# Gráfico (Holt amort.)+Prediccion
titulo = 'Holt amortiguado (Add,N)'
grafico_modelo(
    titulo, serie_train, serie_test, fit_train_holt_am, pred_holt_am, 
    carpeta = graficas
    )

# =============================================================================
# 5. MODELOS CON ESTUDIO DE ESTACIONALIDAD
# =============================================================================
# 5.1. MODELO 3: (A,A) HOLT-WINTERS ADITIVO
modelo_hw_ad = ExponentialSmoothing(
    serie_train, seasonal_periods=m, trend='add', seasonal='add',
    initialization_method='estimated'
    ).fit()

nombre_hw_ad = 'HW_aditivo (A,A)'
fit_train_hw_ad = modelo_hw_ad.fittedvalues
pred_hw_ad = modelo_hw_ad.forecast(n_test)

# Tablas parámetros y prediccion (HW ADD)
params_hw_ad = modelo_hw_ad.params_formatted
tabla_pred_hw_ad = crear_tabla_prediccion(nombre_hw_ad,serie_test,pred_hw_ad)
compara_p.append(tabla_pred_hw_ad)
tabla_pred_hw_ad_2m = tabla_pred_hw_ad.asfreq('2MS').round(2)

# Bondad de ajuste
metricas_hw_ad = metricas_bondad(
    nombre_hw_ad, serie_train, fit_train_hw_ad, serie_test, pred_hw_ad, 
    modelo_hw_ad
    )
compara_m.append(metricas_hw_ad)

# Gráfico
titulo = 'Holt-Winters aditivo (A,A)'
grafico_modelo(
    titulo, serie_train, serie_test, fit_train_hw_ad, pred_hw_ad, 
    carpeta = graficas
    )

# 5.2. MODELO 4: (A,M) HOLT-WINTERS MULTIPLICATIVO
modelo_hw_mul = ExponentialSmoothing(
    serie_train, seasonal_periods = m, trend='add', seasonal='mul', 
    initialization_method='estimated'
    ).fit()
nombre_hw_mul = 'HW_multiplicativo (A,M)'
fit_train_hw_mul = modelo_hw_mul.fittedvalues
pred_hw_mul = modelo_hw_mul.forecast(n_test)

# Tablas parámetros y prediccion (HW MUL)
params_hw_mul = modelo_hw_mul.params_formatted
tabla_pred_hw_mul=crear_tabla_prediccion(nombre_hw_mul, serie_test,pred_hw_mul)
compara_p.append(tabla_pred_hw_mul)
tabla_pred_hw_mul_2m = tabla_pred_hw_mul.asfreq('2MS').round(2)

# Bondad de ajuste
metricas_hw_mul = metricas_bondad(
    nombre_hw_mul, serie_train, fit_train_hw_mul, serie_test, pred_hw_mul, 
    modelo_hw_mul
    )
compara_m.append(metricas_hw_mul)

# Gráfico (HW multiplicativo)+Predccion
titulo = 'Holt-Winters multiplicativo (A,M)'
grafico_modelo(
    titulo, serie_train, serie_test, fit_train_hw_mul, pred_hw_mul,
    carpeta = graficas
    )

# 5.3. MODELO 5: (Add,M) HW AMORTIGUADO MULTIPLICATIVO

modelo_hw_am = ExponentialSmoothing(
    serie_train, seasonal_periods=m, trend='add', damped_trend=True, 
    seasonal='mul', initialization_method='estimated'
    ).fit()

nombre_hw_am = 'HW_amort_mult (Add,M)'
fit_train_hw_am = modelo_hw_am.fittedvalues
pred_hw_am = modelo_hw_am.forecast(n_test)

# Tablas parámetros y prediccion (HW MUL)
params_hw_am = modelo_hw_am.params_formatted
tabla_pred_hw_am = crear_tabla_prediccion(nombre_hw_am, serie_test, pred_hw_am)
compara_p.append(tabla_pred_hw_am)
tabla_pred_hw_am_2m = tabla_pred_hw_am.asfreq('2MS').round(2)

# Bondad de ajuste
metricas_hw_am = metricas_bondad(
    nombre_hw_am, serie_train, fit_train_hw_am, serie_test, pred_hw_am,
    modelo_hw_am
    )
compara_m.append(metricas_hw_am)

# Gráfico (HW amort_multip)
titulo = 'Holt-Winters amortiguado multiplicativo (Add,M)'
grafico_modelo(
    titulo, serie_train, serie_test, fit_train_hw_am, pred_hw_am, 
    carpeta = graficas
    )

# =============================================================================
# 6. COMPARACIÓN Y SELECCIÓN MODELOS
# =============================================================================
# 6.1. TABLAS DE COMPARACIÓN (MÉTRICAS (TRAIN/TEST) y PREDICCIÓN)
# Comparación de los modelos
compara_metricas = pd.DataFrame(compara_m).round(3).T.copy()
compara_metricas.columns = compara_metricas.iloc[0]
compara_metricas = compara_metricas.iloc[1:]

mostrar_resultado('Comparación de modelos de suavizado', compara_metricas)

compara_predic = pd.DataFrame({
    'observado': serie_test,
    'pred_h': compara_p[0].get('prediccion'),
    'pred_h_am': compara_p[1].get('prediccion'),
    'pred_hw_ad': compara_p[2].get('prediccion'),
    'pred_hw_mul': compara_p[3].get('prediccion'),
    'pred_hw_am': compara_p[4].get('prediccion'),
    'error_abs_h': compara_p[0].get('error_abs'),
    'error_abs_h_am': compara_p[1].get('error_abs'),
    'error_abs_hw_ad': compara_p[2].get('error_abs'),
    'error_abs_hw_mul': compara_p[3].get('error_abs'),
    'error_abs_hw_am': compara_p[4].get('error_abs'),
    }).round(2)
compara_predic.index = compara_predic.index.strftime('%Y-%m')

mostrar_resultado(
    'Comparación de las predicciones de suavizado', compara_predic, 3
    )

# 6.2. Comparación gráfica de los tres modelos con estacionalidad
fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharey=True)
titulo = 'TEST vs Predicción Modelos Candidatos'

axes[0].plot(serie_test, label='TEST', color='gold')
axes[0].plot(pred_hw_ad, label='Predicción', color='lime')
axes[0].set_title('Holt-Winters Aditivo - 2025')
axes[0].set_xlabel('Fecha')
axes[0].set_ylabel('GWh')
axes[0].legend()

axes[1].plot(serie_test, label='TEST', color='gold')
axes[1].plot(pred_hw_mul, label='Predicción', color='skyblue')
axes[1].set_title('Holt-Winters Multiplicativo - 2025')
axes[1].set_xlabel('Fecha')
axes[1].set_ylabel('GWh')
axes[1].legend()

axes[2].plot(serie_test, label='TEST', color='gold')
axes[2].plot(pred_hw_am, label='Predicción', color='magenta')
axes[2].set_title('Holt-Winters Amortiguado Mult. - 2025')
axes[2].set_xlabel('Fecha')
axes[2].set_ylabel('GWh')
axes[2].legend()

axes[3].plot(pred_hw_ad, label='Predicción', color='lime')
axes[3].plot(pred_hw_mul, label='Predicción', color='skyblue')
axes[3].plot(pred_hw_am, label='Predicción', color='magenta')
axes[3].set_title('Predicciones - 2025')
axes[3].set_xlabel('Fecha')
axes[3].set_ylabel('GWh')
axes[3].legend()

plt.suptitle(titulo, fontsize=14)
plt.legend()
plt.tight_layout()
guardar_grafico('Panel '+titulo, graficas)
plt.show()

# =============================================================================
# 7. SELECCIÓN MODELO FINAL
# =============================================================================
# Renombramos el modelo tras la comparación
mod_final_nombre = nombre_hw_am
mod_final = modelo_hw_am
pred_final = mod_final.forecast(n_test)
met_final = metricas_hw_am

# =============================================================================
# 8. ANÁLISIS DETALLADO DEL MODELO FINAL
# =============================================================================
# 8.1 TABLA DE PREDICCIONES EN TEST (MODELO FINAL) Y PARÁMETROS
params_final = mod_final.params_formatted
tabla_mod_final=crear_tabla_prediccion(mod_final_nombre,serie_test,pred_final)

mostrar_resultado(
    f'Métricas del modelo seleccionado - {mod_final_nombre}',
    pd.Series(met_final)
    )

mostrar_resultado(
    f'Parámetros del modelo seleccionado - {mod_final_nombre}', params_final
    )

mostrar_resultado(
    f'Predicción TEST - {mod_final_nombre}', tabla_mod_final.round(3)
    )

# 8.2. GRÁFICOS DETALLADOS (MODELO GANADOR) - PANEL 4x2
train_suav = mod_final.fittedvalues # Suavizado

# SERIES COMPLETAS
serie_total = pd.concat([serie_train, serie_test])
serie_modelo = pd.concat([train_suav, pred_final])
# Residuos en TRAIN (real-aprox)
residuos_train = serie_train - train_suav

# ESTADOS
nivel = mod_final.level
tendencia = mod_final.trend
estac = mod_final.season.copy()
estac.index = pd.to_datetime(estac.index, format = '%Y')

fig, axes = plt.subplots(4, 2, figsize=(15, 24))

# 1 - Serie total vs ajuste total
axes[0, 0].plot(serie_total, label='Observada', color='gold')
axes[0, 0].plot(serie_modelo, label='Ajuste/Predicción', color='magenta')
axes[0, 0].set_title('Serie Principal vs Serie Modelo')
axes[0, 0].set_xlabel('Fecha')
axes[0, 0].set_ylabel('GWh')
axes[0, 0].legend()
# 2 - TRAIN: observada vs suavizada (fitted)
axes[0, 1].plot(serie_train, label='Train observada', color='lime')
axes[0, 1].plot(train_suav, label='Suavizada (fitted)', color='skyblue')
axes[0, 1].set_title('TRAIN: observada vs suavizada')
axes[0, 1].set_xlabel('Fecha')
axes[0, 1].set_ylabel('GWh')
axes[0, 1].legend()
# 3 - TEST: observado vs predicción
axes[1, 0].plot(serie_test, label='Test observado', color='gold')
axes[1, 0].plot(pred_final, label='Predicción', color='magenta')
axes[1, 0].set_title('TEST: observado vs predicción (2025)')
axes[1, 0].set_xlabel('Fecha')
axes[1, 0].set_ylabel('GWh')
axes[1, 0].legend()
# 4 - Nivel
axes[1, 1].plot(nivel, color='teal', linewidth=1.5)
axes[1, 1].set_title('Nivel')
axes[1, 1].set_xlabel('Fecha')
axes[1, 1].set_ylabel('Nivel')
# 5 - Tendencia
axes[2, 0].plot(tendencia, color='turquoise', linewidth=1.5)
axes[2, 0].set_title('Tendencia')
axes[2, 0].set_xlabel('Fecha')
axes[2, 0].set_ylabel('Tendencia')
# 6 - Estacionalidad
axes[2, 1].plot(estac[-60:], color='royalblue', linewidth=1.5)
axes[2, 1].set_title('Estacionalidad últimos 5 años')
axes[2, 1].set_xlabel('Fecha')
axes[2, 1].set_ylabel('Estacional')
axes[2, 1].set_xticks(estac.index[-60::6])
axes[2, 1].set_xticklabels(estac.index[-60::6].strftime('%Y-%m'), fontsize=8)
# 7 - Residuos TRAIN (diferencia Observada - Suavizada)
axes[3, 0].plot(residuos_train, color='coral', linewidth=1)
axes[3, 0].axhline(0, linestyle='--', color='black',linewidth=1.2, alpha = 0.4)
axes[3, 0].set_title('Residuos TRAIN')
axes[3, 0].set_xlabel('Fecha')
axes[3, 0].set_ylabel('Error (GWh)')
# 8 - Histograma residuos TRAIN
axes[3, 1].hist(
    residuos_train.dropna().values, bins=15, color='wheat', edgecolor='oldlace'
    )
axes[3, 1].set_title('Distribución residuos TRAIN')
axes[3, 1].set_xlabel('Error (GWh)')
axes[3, 1].set_ylabel('Frecuencia')

titulo= f'Panel - {mod_final_nombre}'
plt.suptitle(titulo, fontsize=14, weight ='bold', y=1)
plt.tight_layout()
guardar_grafico(titulo, graficas)
plt.show()

# 8.3 GRÁFICO MODELO SELECCIONADO
titulo = f'Modelo Seleccionado: {mod_final_nombre}'
grafico_modelo(
    titulo, serie_train, serie_test, train_suav, pred_final,
    carpeta = graficas
    )

# =============================================================================
# 9. GUARDAR ARCHIVO PARA REUTILIZAR (PICKLE)
# =============================================================================
salida = {
    'nombre': mod_final_nombre,
    'modelo': mod_final,    
    'prediccion': pred_final,
    'tabla prediccion': tabla_mod_final,
    'params': params_final,
    'metricas': met_final,
    }
guardar_pickle(salida, dir_out_pickle)