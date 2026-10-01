# -*- coding: utf-8 -*-
"""
Series Temporales - Ejercicio de Evaluación

05_series_arima_manual.py
 - Representación de la serie (TRAIN)
 - Correlogramas ACF y PACF
 - Diferenciación estacional (m=12) y diferenciación regular
 - Propuesta y ajuste manual de modelos ARIMA estacionales
 - Diagnóstico: comprobar que los residuos están incorrelados
 - Tabla de parámetros estimados + gráficos de diagnóstico
 - Tabla de comparación de modelos

Autor: Carlos Ayllón Motto
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import shutil

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from series_funciones import (
    cargar_train_test, guardar_pickle, mostrar_resultado, guardar_grafico,
    revision_dif, comprobacion_resid, info_modelo
    )

warnings.filterwarnings('ignore')

# =============================================================================
# 1. RUTAS DE DATOS
# =============================================================================

dir_trabajo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

dir_datos = os.path.join(dir_trabajo, 'outputs')

dir_images = os.path.join(dir_trabajo, 'images')
os.makedirs(dir_images, exist_ok=True)

graficas = os.path.join(dir_images, 'arima')
if os.path.exists(graficas):
    shutil.rmtree(graficas)
os.makedirs(graficas, exist_ok=True)

dir_pickle_tt = os.path.join(dir_datos, 'pv_train_test.pkl')

dir_pickle_modelo = os.path.join(dir_datos, 'modelo_arima_manual.pkl')

# =============================================================================
# 2. CARGA DE DATOS
# =============================================================================
serie, serie_train, serie_test, m, datos_tt = cargar_train_test(dir_pickle_tt)

# =============================================================================
# 3. FUNCIÓN DE SELECCIÓN
# =============================================================================
# DF de información
modelos_seleccionados = []
modelos_ari = []

def seleccion_arima(modelos: dict) -> pd.DataFrame():
    """
     - Entrada: Lista con todos los modelos evaluados
     - Se establecen criterios de selección
     - Por comparación se añaden los modelos que cumplan las condiciones
     - Salida: df con los modelos aceptados
    """
    candidatos = pd.DataFrame(modelos).round(4)

    condicion_p = (candidatos['no_signif'] == 0)
    condicion_lb = (candidatos['Contraste_LB'] == 'ok')
    condicion_lb3 = (candidatos['P_LB(3)'] > 0.05)

    # Establecemos los filtros para las condiciones
    filtro_1 = condicion_p & condicion_lb
    filtro_2 = condicion_p & condicion_lb3  # bajo nivel de restricción

    seleccion = candidatos[filtro_1]    # primer filtrado

    extras = pd.DataFrame()
    if len(seleccion) < 5:  # para añadir más opciones
        extras = candidatos[filtro_2]

    # agrupamos los modelos seleccionados
    seleccion = pd.concat([seleccion, extras]).drop_duplicates()

    # orden de observación y presentación
    seleccion = seleccion.sort_values(
        by=['Contraste_LB', 'AIC'], ascending=[False, True]
        )
    seleccion = seleccion.set_index('modelo').T
    return seleccion

# =============================================================================
# 4. REPRESENTACIÓN DE LA SERIE (TRAIN)
# =============================================================================
plt.figure(figsize=(14, 8))
titulo = 'Serie TRAIN - Solar fotovoltaica (GWh)'
plt.plot(serie_train, label = 'TRAIN', color='gold', linewidth=1.5)
plt.title(titulo)
plt.xlabel('Fecha')
plt.ylabel('GWh')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()

guardar_grafico(titulo, graficas)
plt.show()

# =============================================================================
# 5. CORRELOGRAMAS (SERIE SIN DIFERENCIAR)
# =============================================================================
lag_graficas = 36

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
plot_acf(serie_train, lags=lag_graficas, ax=ax1)
ax1.set_title('Función de Autocorrelación (ACF) - TRAIN (sin diferenciar)')

plot_pacf(serie_train, lags=lag_graficas, ax=ax2)
ax2.set_title(
    'Función de Autocorrelación Parcial (PACF) - TRAIN (sin diferenciar)'
    )

for ax in [ax1, ax2]: # rejilla para visualización cada 6 meses
    ax.set_xticks(range(0, lag_graficas + 1, 6))
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    
titulo = 'Correlogramas Serie TRAIN sin diferenciar'
plt.suptitle(titulo, fontsize=14)    
plt.tight_layout()

guardar_grafico(titulo, graficas)
plt.show()

# =============================================================================
# 6. DIFERENCIACIÓN - SERIE ESTACIONARIA
# =============================================================================
# 6.1. DIFERENCIACIÓN ESTACIONAL
dif_est = serie_train.diff(m)

# Gráfica diferenciación estacional
plt.figure(figsize=(14, 8))
titulo = 'Serie en diferencia estacional (m = 12)'
plt.plot(dif_est, color='skyblue', linewidth=1.2)
plt.axhline(0, linestyle='--', color='black', linewidth=1.2)
plt.title(titulo)
plt.xlabel('Fecha')
plt.ylabel('Diferencia (GWh)')
plt.tight_layout()

guardar_grafico(titulo, graficas)
plt.show()

dif_est = dif_est.dropna()
a_dif1 = revision_dif(dif_est)

mostrar_resultado('ADF - Diferenciación estacional D=1', a_dif1)
# Correlograma dif. estacional
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
plot_acf(dif_est, lags=lag_graficas, ax=ax1)
ax1.set_title('ACF - diferencia estacional (m = 12)')

plot_pacf(dif_est, lags=lag_graficas, ax=ax2)
ax2.set_title('PACF - diferencia estacional (m = 12)')
for ax in [ax1, ax2]: # rejilla para visualización cada 6 meses
    ax.set_xticks(range(0, lag_graficas + 1, 6))
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)

titulo = 'Correlogramas Serie TRAIN con diferenciación estacional (D=1)'
plt.suptitle(titulo, fontsize=14)    
plt.tight_layout()

guardar_grafico(titulo, graficas)
plt.show()

# 6.2. DIFERENCIACIÓN REGULAR
# No estacionariedad en media tras revision de autocorrelograma y parámetros
# Aplica doble diferenciación (d=1 y D=1) - añado la regular +1
# Comprobación diferenciación regular
dif_est_2 = dif_est.diff(1)

# Gráfica diferenciación regular y estacional
plt.figure(figsize=(14, 8))
titulo = 'Serie con doble diferenciación (regular y estacional)'
plt.plot(dif_est_2, color='lightcoral', linewidth=1.2)
plt.axhline(0, linestyle='--', color='black', linewidth=1.2)
plt.title(titulo)
plt.xlabel('Fecha')
plt.ylabel('Diferencia (GWh)')
plt.tight_layout()

guardar_grafico(titulo, graficas)
plt.show()

dif_est_2 = dif_est_2.dropna()
a_dif2 = revision_dif(dif_est_2)

mostrar_resultado('ADF - Diferenciación regular y estacional d=1, D=1', a_dif2)

# Correlograma dif. doble
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
plot_acf(dif_est_2, lags=lag_graficas, ax=ax1)
ax1.set_title('ACF - doble diferenciación (d=1 y D=1)')

plot_pacf(dif_est_2, lags=lag_graficas, ax=ax2)
ax2.set_title('PACF - doble diferenciación (d=1 y D=1)')

for ax in [ax1, ax2]: # rejilla para visualización cada 6 meses
    ax.set_xticks(range(0, lag_graficas + 1, 6))
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    
titulo = 'Correlogramas Serie TRAIN con doble diferenciación (D=1, d=1)'
plt.suptitle(titulo, fontsize=14)    
plt.tight_layout()

guardar_grafico(titulo, graficas)
plt.show()

# =============================================================================
# 7. AJUSTE MANUAL DE MODELOS ARIMA ESTACIONALES
# =============================================================================
# La identificación final la hacemos mirando los correlogramas.
# Empezamos por modelos sencillos (parsimonia) y comprobamos los residuos.
# Se mantiene en todos los modelos la diferencia estacional y regular: d=1,D=1

# 7.1 MODELO CANDIDATO A 
#------------------------------------------------------
# serie estacionaria (establecemos 2 diferenciaciones d=1 y D=1)
# ACF: lag1 y lag12 -> q=1 y Q=1 (media movil estacional y regular)
modelo_A_name = 'A_ARIMA(0,1,1)(0,1,1)'

modelo_A = sm.tsa.ARIMA(serie_train, order=(0,1,1), seasonal_order=(0,1,1,12))

resultados_A = modelo_A.fit()
resultados_A.summary()

# Comprobación de residuos incorrelados+información
ljung_A = comprobacion_resid(modelo_A_name, resultados_A, m, graficas)
ljung_A_1 = ljung_A.rename(ljung_A['modelo']).drop('modelo')

tabla_info_A = info_modelo(modelo_A_name, resultados_A, ljung_A_1, modelos_ari)
tabla_info_A = tabla_info_A.rename(
    tabla_info_A['modelo']
    ).drop('modelo').dropna()

# 7.2 MODELO CANDIDATO B (doble diferencia: d=1, D=1)
#------------------------------------------------------
# Mantiene Q=1 (medias moviles en estacionalidad anual)
# p=1 (dependencia del valor mensual)
modelo_B_name = 'B_ARIMA(1,1,0)(0,1,1)'

modelo_B = sm.tsa.ARIMA(serie_train, order=(1,1,0), seasonal_order=(0,1,1,12))
resultados_B = modelo_B.fit()
resultados_B.summary()

ljung_B = comprobacion_resid(modelo_B_name, resultados_B, m, graficas)
ljung_B_1 = ljung_B.rename(ljung_B['modelo']).drop('modelo')

tabla_info_B = info_modelo(modelo_B_name, resultados_B, ljung_B_1, modelos_ari)
tabla_info_B = tabla_info_B.rename(
    tabla_info_B['modelo']
    ).drop('modelo').dropna()

# 7.3 MODELO CANDIDATO C (doble diferencia: d=1, D=1)
#------------------------------------------------------
# Mantiene Q=1 (medias moviles en estacionalidad anual)
# p=1 no aporta -> P=1 (dependencia del valor estacional)
modelo_C_name = 'C_ARIMA(0,1,1)(1,1,1)'

modelo_C = sm.tsa.ARIMA(serie_train, order=(0,1,1), seasonal_order=(1,1,1,12))
resultados_C = modelo_C.fit()
resultados_C.summary()

ljung_C = comprobacion_resid(modelo_C_name, resultados_C, m, graficas)
ljung_C_1 = ljung_C.rename(ljung_C['modelo']).drop('modelo')

tabla_info_C = info_modelo(modelo_C_name, resultados_C, ljung_C_1, modelos_ari)
tabla_info_C = tabla_info_C.rename(
    tabla_info_C['modelo']
    ).drop('modelo').dropna()

# 7.4 MODELO CANDIDATO D
#------------------------------------------------------
# Mantiene Q=1 (medias moviles en estacionalidad anual)
# p=1 y q=1 en AR
modelo_D_name = 'D_ARIMA(1,1,1)(0,1,1)'

modelo_D = sm.tsa.ARIMA(serie_train, order=(1,1,1), seasonal_order=(0,1,1,12))

resultados_D = modelo_D.fit()
resultados_D.summary()

ljung_D = comprobacion_resid(modelo_D_name, resultados_D, m, graficas)
ljung_D_1 = ljung_D.rename(ljung_D['modelo']).drop('modelo')

tabla_info_D = info_modelo(modelo_D_name, resultados_D, ljung_D_1, modelos_ari)
tabla_info_D = tabla_info_D.rename(
    tabla_info_D['modelo']
    ).drop('modelo').dropna()

# 7.5 MODELO CANDIDATO E
#------------------------------------------------------
# P=1 # Solo un parametro estacional
# p=1 y q=1 en AR
modelo_E_name = 'E_ARIMA(1,1,1)(1,1,0)'

modelo_E = sm.tsa.ARIMA(serie_train, order=(1,1,1), seasonal_order=(1,1,0,12))

resultados_E = modelo_E.fit()
resultados_E.summary()

ljung_E = comprobacion_resid(modelo_E_name, resultados_E, m, graficas)
ljung_E_1 = ljung_E.rename(ljung_E['modelo']).drop('modelo')

tabla_info_E = info_modelo(modelo_E_name, resultados_E, ljung_E_1, modelos_ari)
tabla_info_E = tabla_info_E.rename(
    tabla_info_E['modelo']
    ).drop('modelo').dropna()

# 7.6 MODELO CANDIDATO F
#------------------------------------------------------
# P=1 Q=1 Reforzamos parte estacional
# p=1 y q=1
modelo_F_name = 'F_ARIMA(1,1,1)(1,1,1)'

modelo_F = sm.tsa.ARIMA(serie_train, order=(1,1,1), seasonal_order=(1,1,1,12))

resultados_F = modelo_F.fit()
resultados_F.summary()

ljung_F = comprobacion_resid(modelo_F_name, resultados_F, m, graficas)
ljung_F_1 = ljung_F.rename(ljung_F['modelo']).drop('modelo')

tabla_info_F = info_modelo(modelo_F_name, resultados_F, ljung_F_1, modelos_ari)
tabla_info_F = tabla_info_F.rename(
    tabla_info_F['modelo']
    ).drop('modelo').dropna()

# 7.7 MODELO CANDIDATO G
#------------------------------------------------------
# Aumento de complejidad en AR (medias)
modelo_G_name = 'G_ARIMA(0,1,2)(0,1,1)'

modelo_G = sm.tsa.ARIMA(serie_train, order=(0,1,2), seasonal_order=(0,1,1,12))

resultados_G = modelo_G.fit()
resultados_G.summary()

ljung_G = comprobacion_resid(modelo_G_name, resultados_G, m, graficas)
ljung_G_1 = ljung_G.rename(ljung_G['modelo']).drop('modelo')

tabla_info_G = info_modelo(modelo_G_name, resultados_G, ljung_G_1, modelos_ari)
tabla_info_G = tabla_info_G.rename(
    tabla_info_G['modelo']
    ).drop('modelo').dropna()

# 7.8 MODELO CANDIDATO H
#------------------------------------------------------
# Aumento de complejidad en AR p=2
modelo_H_name = 'H_ARIMA(2,1,0)(0,1,1)'

modelo_H = sm.tsa.ARIMA(serie_train, order=(2,1,0), seasonal_order=(0,1,1,12))

resultados_H = modelo_H.fit()
resultados_H.summary()

ljung_H = comprobacion_resid(modelo_H_name, resultados_H, m, graficas)
ljung_H_1 = ljung_H.rename(ljung_H['modelo']).drop('modelo')

tabla_info_H = info_modelo(modelo_H_name, resultados_H, ljung_H_1, modelos_ari)
tabla_info_H = tabla_info_H.rename(
    tabla_info_H['modelo']
    ).drop('modelo').dropna()

# 7.9 MODELO CANDIDATO I
#------------------------------------------------------
# Aumento de complejidad en AR p=2
# P=1
modelo_I_name = 'I_ARIMA(2,1,1)(0,1,1)'

modelo_I = sm.tsa.ARIMA(serie_train, order=(2,1,1), seasonal_order=(0,1,1,12))

resultados_I = modelo_I.fit()
resultados_I.summary()

ljung_I = comprobacion_resid(modelo_I_name, resultados_I, m, graficas)
ljung_I_1 = ljung_I.rename(ljung_I['modelo']).drop('modelo')

tabla_info_I = info_modelo(modelo_I_name, resultados_I, ljung_I_1, modelos_ari)
tabla_info_I = tabla_info_I.rename(
    tabla_info_I['modelo']
    ).drop('modelo').dropna()

# 7.10 MODELO CANDKDATO J
#------------------------------------------------------
# Q = 2 / Aumento MA estacional
modelo_J_name = 'J_ARIMA(1,1,0)(0,1,2)'

modelo_J = sm.tsa.ARIMA(serie_train, order=(1,1,0), seasonal_order=(0,1,2,12))

resultados_J = modelo_J.fit()
resultados_J.summary()

ljung_J = comprobacion_resid(modelo_J_name, resultados_J, m, graficas)
ljung_J_1 = ljung_J.rename(ljung_J['modelo']).drop('modelo')
tabla_info_J = info_modelo(modelo_J_name, resultados_J, ljung_J_1, modelos_ari)
tabla_info_J = tabla_info_J.rename(
    tabla_info_J['modelo']
    ).drop('modelo').dropna()

# 7.11 MODELO CANDKDATO K
#------------------------------------------------------
# Modificación qxp
modelo_K_name = 'K_ARIMA(0,1,1)(0,1,2)'

modelo_K = sm.tsa.ARIMA(serie_train, order=(0,1,1), seasonal_order=(0,1,2,12))

resultados_K = modelo_K.fit()
resultados_K.summary()

ljung_K = comprobacion_resid(modelo_K_name, resultados_K, m, graficas)
ljung_K_1 = ljung_K.rename(ljung_K['modelo']).drop('modelo')

tabla_info_K = info_modelo(modelo_K_name, resultados_K, ljung_K_1, modelos_ari)
tabla_info_K = tabla_info_K.rename(
    tabla_info_K['modelo']
    ).drop('modelo').dropna()

# 7.12 MODELO CANDKDATO L
#------------------------------------------------------
# Añadimos complejidad en no estacional
modelo_L_name = 'L_ARIMA(1,1,1)(0,1,2)'

modelo_L = sm.tsa.ARIMA(serie_train, order=(1,1,1), seasonal_order=(0,1,2,12))
resultados_L = modelo_L.fit()
resultados_L.summary()

ljung_L = comprobacion_resid(modelo_L_name, resultados_L, m, graficas)
ljung_L_1 = ljung_L.rename(ljung_L['modelo']).drop('modelo')

tabla_info_L = info_modelo(modelo_L_name, resultados_L, ljung_L_1, modelos_ari)
tabla_info_L = tabla_info_L.rename(
    tabla_info_L['modelo']
    ).drop('modelo').dropna()

# 7.13 MODELO CANDKDATO M
#------------------------------------------------------
# prueba con P=1
modelo_M_name = 'M_ARIMA(1,1,1)(1,1,2)'

modelo_M = sm.tsa.ARIMA(serie_train, order=(1,1,1), seasonal_order=(1,1,2,12))

resultados_M = modelo_M.fit()
resultados_M.summary()

ljung_M = comprobacion_resid(modelo_M_name, resultados_M, m, graficas)
ljung_M_1 = ljung_M.rename(ljung_M['modelo']).drop('modelo')

tabla_info_M = info_modelo(modelo_M_name, resultados_M, ljung_M_1, modelos_ari)
tabla_info_M = tabla_info_M.rename(
    tabla_info_M['modelo']
    ).drop('modelo').dropna()

# 7.14 MODELO CANDKDATO N
#------------------------------------------------------
modelo_N_name = 'N_ARIMA(1,1,1)(2,1,0)'

modelo_N = sm.tsa.ARIMA(serie_train, order=(1,1,1), seasonal_order=(2,1,0,12))

resultados_N = modelo_N.fit()
resultados_N.summary()

ljung_N = comprobacion_resid(modelo_N_name, resultados_N, m, graficas)
ljung_N_1 = ljung_N.rename(ljung_N['modelo']).drop('modelo')

tabla_info_N = info_modelo(modelo_N_name, resultados_N, ljung_N_1, modelos_ari)
tabla_info_N = tabla_info_N.rename(
    tabla_info_N['modelo']
    ).drop('modelo').dropna()

# 7.15 INFORMACIÓN DE TODOS LOS MODELOS
# Mucho redondeo para espacio en el documento
modelos_arima = pd.DataFrame(modelos_ari).round(2).T

# =============================================================================
# 8. FILTRADO DE MODELOS
# =============================================================================
# Condicion 1: 0 parámetros no significativos
# Condición 2: residuos incorrelados
# Condición 3: residuo no estacional incorrelado
modelos_seleccionados = seleccion_arima(modelos_ari)

mostrar_resultado(
    'Modelos ARIMA seleccionados para comparación', modelos_seleccionados
    )

# =============================================================================
# 9. SELECCIÓN DEL MODELO MANUAL
# =============================================================================
# Tablas de comparacion retardos regulares
retardos = pd.DataFrame({
    'modelo': modelos_seleccionados.columns,
    'P_LB(3)': modelos_seleccionados.loc['P_LB(3)'].values,
    'H0(3)': np.where(
        modelos_seleccionados.loc['P_LB(3)'] > 0.05, 'no rechaza H0',
        'se rechaza H0'
        ),
    'P_LB(reg)': modelos_seleccionados.loc['P_LB(reg)'].values,
    'H0(reg)': np.where(
        modelos_seleccionados.loc['P_LB(reg)'] > 0.05, 'no rechaza H0', 
        'se rechaza H0'
        )
    })
retardos = retardos.set_index('modelo')

mostrar_resultado('Contraste Ljung-Box - modelos candidatos', retardos)

# 9.1. Selección -> Parsimonia+contraste
mod_arima_nombre = modelo_L_name
res_arima = resultados_L

ljung_arima = ljung_L

tabla_info_arima = info_modelo(mod_arima_nombre, res_arima, ljung_arima)

mod_arima_nombre2 = modelo_N_name
res_arima2 = resultados_N

ljung_arima2 = ljung_N

tabla_info_arima2 = info_modelo(mod_arima_nombre2, res_arima2, ljung_arima2)

comparacion_final_manual = pd.concat([
    tabla_info_arima.rename(mod_arima_nombre),
    tabla_info_arima2.rename(mod_arima_nombre2)
    ], axis=1)

comparacion_final_manual = comparacion_final_manual.drop(
    index='modelo', errors='ignore'
    )

mostrar_resultado(
    'Comparación final ARIMA manual - modelos L y N', comparacion_final_manual
    )

Modelo_1 = pd.Series({
    'nombre': mod_arima_nombre,
    'resultados': res_arima,
    'contraste_ljungbox': ljung_arima,
    'info': tabla_info_arima,
    })

Modelo_1.resultados.summary()

Modelo_2 = pd.Series({
    'nombre': mod_arima_nombre2,
    'resultados': res_arima2,
    'contraste_ljungbox': ljung_arima2,
    'info': tabla_info_arima2,
    })

# =============================================================================
# 10. GUARDADO DEL MODELO MANUAL SELECCIONADO (PICKLE)
# =============================================================================
salida = {
    'Modelo1': Modelo_1,
    'Modelo2': Modelo_2,
    'modelos_arima': modelos_arima,
    'modelos_seleccionados': modelos_seleccionados,
    'retardos': retardos,
    }

guardar_pickle(salida, dir_pickle_modelo)