# -*- coding: utf-8 -*-
"""
Series Temporales - Ejercicio de Evaluación

06_series_arima_automatico.py
 - Modelo Arima con ajuste automático
 - Comparación con el modelo manual 

Autor: Carlos Ayllón Motto
"""
# Importación de librerías
import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pmdarima as pm
import shutil

from series_funciones import (
    cargar_train_test, cargar_pickle, guardar_pickle, mostrar_resultado, 
    guardar_grafico, comprobacion_resid, info_modelo, calcular_metricas_pred
    )

warnings.filterwarnings('ignore')

# =============================================================================
# 1. RUTAS DE DATOS
# =============================================================================
dir_trabajo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

dir_datos = os.path.join(dir_trabajo, 'outputs')

dir_images = os.path.join(dir_trabajo, 'images')
os.makedirs(dir_images, exist_ok=True)

graficas = os.path.join(dir_images, 'arima_auto')
if os.path.exists(graficas):
    shutil.rmtree(graficas)
os.makedirs(graficas, exist_ok=True)

dir_pickle_tt = os.path.join(dir_datos, 'pv_train_test.pkl')

dir_pickle_auto = os.path.join(dir_datos, 'modelo_arima_auto.pkl')

dir_pickle_manual = os.path.join(dir_datos, 'modelo_arima_manual.pkl')

dir_pickle_seleccion = os.path.join(dir_datos, 'modelo_arima_seleccion.pkl')

# =============================================================================
# 2. CARGA DE DATOS
# =============================================================================
# 2.1 TRAIN / TEST
serie, serie_train, serie_test, m, datos_tt = cargar_train_test(dir_pickle_tt)

# 2.2 MODELO MANUAL (script 05)
datos_manual = cargar_pickle(dir_pickle_manual)

modelo_L = datos_manual['Modelo1']
nombre_L = modelo_L.nombre
res_L = modelo_L.resultados

modelo_N = datos_manual['Modelo2']
nombre_N = modelo_N.nombre
res_N = modelo_N.resultados

# =============================================================================
# 3. FUNCIONES AUXILIARES
# =============================================================================
# DF para guardar los modelos y compararlos
compara_modelos = []
compara_ljung = []
compara_metricas = []

def metricas_error(nombre_modelo: str, serie_train, serie_test, 
                   pred_test: dict, pred_train: dict) -> pd.Series():
    """
    Métricas calculadas:
        - TEST : serie_test  vs prediccion_test
        - MAE
        - RMSE
        - MAPE
    """
    met_train = calcular_metricas_pred(nombre_modelo, serie_train, pred_train)
    met_test = calcular_metricas_pred(nombre_modelo, serie_test, pred_test)

    metricas = {
        'Modelo': nombre_modelo,
        'MAE_train': met_train['MAE'],
        'RMSE_train': met_train['RMSE'],
        'MAPE_train_%': met_train['MAPE_%'],
        'MAE_test': met_test['MAE'],
        'RMSE_test': met_test['RMSE'],
        'MAPE_test_%': met_test['MAPE_%'],
        }
    compara_metricas.append(metricas)
    return metricas

# =============================================================================
# 4. AJUSTE AUTOMÁTICO (AUTO_ARIMA) SOBRE TRAIN
# =============================================================================
# 4.1. AUTOARIMA - AIC
# Fijamos d=1 y D=1 según (estaciónarización para la serie)
# El ajuste automático seleccionará (p,q,P,Q) minimizando AIC (parsimonia).
mod_auto_aic_pm = pm.auto_arima(
    serie_train, seasonal=True, m=m, d=1, D=1, start_p=0, start_q=0, max_p=3,
    max_q=3, start_P=0, start_Q=0, max_P=2, max_Q=2, 
    information_criterion='aic', stepwise=True, trace=True,
    error_action='ignore', suppress_warnings=True
    )

orden_aic = mod_auto_aic_pm.order #(p,d,q)
orden_est_aic = mod_auto_aic_pm.seasonal_order  #(P,D,Q,m)

aic_auto_nombre = (
    f'AUTOARIMA_aic({orden_aic[0]},{orden_aic[1]},{orden_aic[2]})'
    f'({orden_est_aic[0]},{orden_est_aic[1]},{orden_est_aic[2]})'
    )
# Cambio de pdmarima a statsmodels
mod_auto_aic_sm = sm.tsa.ARIMA(
    serie_train, order=orden_aic, seasonal_order=orden_est_aic
    )

# 4.2. AUTOARIMA - BIC
mod_auto_bic_pm = pm.auto_arima(
    serie_train, seasonal=True, m=m, d=1, D=1, start_p=0, start_q=0, max_p=3,
    max_q=3, start_P=0, start_Q=0, max_P=2, max_Q=2, 
    information_criterion='bic',stepwise=True, trace=True,
    error_action='ignore', suppress_warnings=True
    )

orden_bic = mod_auto_bic_pm.order #(p,d,q)
orden_est_bic = mod_auto_bic_pm.seasonal_order  #(P,D,Q,m)

bic_auto_nombre = (
    f'AUTOARIMA_bic({orden_bic[0]},{orden_bic[1]},{orden_bic[2]})'
    f'({orden_est_bic[0]},{orden_est_bic[1]},{orden_est_bic[2]})'
    )

# Cambio de pdmarima a statsmodels
mod_auto_bic_sm = sm.tsa.ARIMA(
    serie_train, order=orden_bic, seasonal_order=orden_est_bic
    )

# =============================================================================
# 5. DIAGNÓSTICO Y TABLA INFO MODELOS
# =============================================================================
# 5.1. MODELOS AUTOMATICOS (AIC Y BIC)
res_auto_aic = mod_auto_aic_sm.fit()
res_auto_aic.summary()

ljung_auto_aic = comprobacion_resid(aic_auto_nombre, res_auto_aic, m, graficas)
compara_ljung.append(ljung_auto_aic)

info_auto_aic = info_modelo(aic_auto_nombre, res_auto_aic, ljung_auto_aic)
compara_modelos.append(info_auto_aic)

res_auto_bic = mod_auto_bic_sm.fit()
res_auto_bic.summary()

ljung_auto_bic = comprobacion_resid(bic_auto_nombre, res_auto_bic, m, graficas)
compara_ljung.append(ljung_auto_bic)

info_auto_bic = info_modelo(bic_auto_nombre, res_auto_bic, ljung_auto_bic)
compara_modelos.append(info_auto_bic)

compara_auto_info = pd.DataFrame(
    compara_modelos
    ).set_index('modelo').T.round(3).dropna()

compara_auto_lb = pd.DataFrame(
    compara_ljung
    ).set_index('modelo').T.round(3).dropna()

# 5.2. MODELOS MANUALES (L Y N)
res_L.summary()
ljung_L = comprobacion_resid(nombre_L, res_L, m, graficas)

compara_ljung.append(ljung_L)
info_L = info_modelo(nombre_L, res_L, ljung_L)

compara_modelos.append(info_L)

res_N.summary()
ljung_N = comprobacion_resid(nombre_N, res_N, m, graficas)

compara_ljung.append(ljung_N)
info_N = info_modelo(nombre_N, res_N, ljung_N)

compara_modelos.append(info_N)

# =============================================================================
# 6. TABLAS DE COMPARACIÓN MODELO AUTO VS _L
# =============================================================================
compara_info = pd.concat([
    info_auto_bic.drop('modelo').rename(bic_auto_nombre),
    info_L.drop('modelo').rename(nombre_L)
], axis=1).round(3)

compara_lb = pd.concat([
    ljung_auto_bic.drop('modelo').rename(bic_auto_nombre),
    ljung_L.drop('modelo').rename(nombre_L)
], axis=1).round(3)

mostrar_resultado('Comparación AutoARIMA BIC vs ARIMA manual', compara_info)
mostrar_resultado(
    'Diagnóstico Ljung-Box - AutoARIMA BIC vs ARIMA manual', compara_lb
    )

# =============================================================================
# 7. COMPARACIÓN PREDICCIÓN AUTO VS _L (TRAIN/TEST)
# =============================================================================
n = len(serie_test)

# Predicciones Modelos
prediccion_L = res_L.get_forecast(steps=n)
prediccion_auto_aic = res_auto_aic.get_forecast(steps=n)
prediccion_auto_bic = res_auto_bic.get_forecast(steps=n)

pred_L = prediccion_L.predicted_mean
pred_L = pd.Series(pred_L.values,index=serie_test.index,name='prediccion_L_L')

pred_auto_aic = prediccion_auto_aic.predicted_mean
pred_auto_aic = pd.Series(
    pred_auto_aic.values, index=serie_test.index, name='prediccion_auto_aic'
    )

pred_auto_bic = prediccion_auto_bic.predicted_mean
pred_auto_bic = pd.Series(
    pred_auto_bic.values, index=serie_test.index, name='prediccion_auto_bic'
    )

# Valores ajustados en TRAIN
L_train = pd.Series(
    res_L.fittedvalues, index=serie_train.index, name='prediccion_L_train_L'
    )

auto_aic_train = pd.Series(
    res_auto_aic.fittedvalues, index=serie_train.index,
    name='prediccion_auto_aic_train'
    )

auto_bic_train = pd.Series(
    res_auto_bic.fittedvalues, index=serie_train.index,
    name='prediccion_auto_bic_train'
    )

# METRICAS COMPARATIVAS
met_auto_aic = metricas_error(
    aic_auto_nombre, serie_train, serie_test, pred_auto_aic, auto_aic_train
    )

met_auto_bic = metricas_error(
    bic_auto_nombre, serie_train, serie_test, pred_auto_bic, auto_bic_train
    )

met_L = metricas_error(nombre_L, serie_train, serie_test, pred_L, L_train)

compara_error = pd.DataFrame(compara_metricas).set_index('Modelo').T.round(3)

mostrar_resultado('Métricas de predicción - modelos ARIMA', compara_error)

# =============================================================================
# 8. GRÁFICO COMPARACIÓN
# =============================================================================
# GRÁFICO 1 - SERIE COMPLETA
plt.figure(figsize=(14, 8))
titulo = 'Observación Real vs Auto_Arima vs Arima_L'

plt.plot(serie_train, label='TRAIN', color='lime', linewidth = 1.5)
plt.plot(serie_test, label='TEST', color='gold', linewidth = 1.5)

plt.plot(auto_aic_train, color='skyblue', linewidth = 1)
plt.plot(
    pred_auto_aic, label='Auto_Arima_AIC',color='skyblue', linewidth = 1.5
    )

plt.plot(auto_bic_train, color='skyblue', linewidth = 1)
plt.plot(
    pred_auto_bic, label='Auto_Arima_BIC', color='skyblue', linewidth = 1.5
    )

plt.plot(L_train, color='magenta', linewidth = 1)
plt.plot(pred_L, label='Arima_L', color='magenta',linewidth = 1.5)

plt.axvline(
    x=serie_test.index.min(), linestyle='--', color='black', linewidth = 2
    )

plt.xlabel('Fecha')
plt.ylabel('GWh')
plt.title(titulo)
plt.legend()
plt.tight_layout()

guardar_grafico(titulo, graficas)
plt.show()

# GRÁFICO 2 - TEST
plt.figure(figsize=(14, 8))
titulo = 'Comparación de predicción en TEST'
plt.plot(serie_test, label='TEST (real)')
plt.plot(pred_auto_aic, label='AUTO_AIC (pred)')
plt.plot(pred_auto_bic, label='AUTO_BIC (pred)')
plt.plot(pred_L, label='ARIMA_L (pred)')

plt.title(titulo)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()

guardar_grafico(titulo, graficas)
plt.show()

# =============================================================================
# 9. GUARDADO DEL MODELO AUTOMÁTICO Y SELECCIONADO
# =============================================================================
# 9.1. SELECCIÓN FINAL
arima_seleccion = modelo_L

mostrar_resultado('Modelo ARIMA seleccionado', arima_seleccion.nombre)

# 9.2. GUARDADO DE MODELOS
salida_auto = {
    'nombre': bic_auto_nombre,
    'mod_auto_pm': mod_auto_bic_pm,
    'mod_auto_sm': mod_auto_bic_sm,
    'params': info_auto_bic,
    'metricas': met_auto_bic,
    'prediccion': prediccion_auto_bic,
    'valores_prediccion': pred_auto_bic,
    }

guardar_pickle(salida_auto, dir_pickle_auto)
    
salida_seleccion = {
    'nombre': arima_seleccion.nombre,
    'modelo': arima_seleccion.resultados,
    'params': info_L,
    'metricas': met_L,
    'prediccion': prediccion_L,
    'valores_prediccion': pred_L  
    }

guardar_pickle(salida_seleccion, dir_pickle_seleccion)