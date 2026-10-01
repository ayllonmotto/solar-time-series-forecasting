# -*- coding: utf-8 -*-
"""
Series Temporales - Ejercicio de Evaluación

09_series_comparacion_final.py
 - Comparación final: Suavizado vs ARIMA vs ETS
 - Evaluación en TEST (año 2025)
 - Métricas, errores e intervalos de confianza
 - Gráficos comparativos

Autor: Carlos Ayllón Motto
"""
# Importación de librerías
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil

from series_funciones import (
    cargar_train_test, cargar_pickle, mostrar_resultado, guardar_grafico,
    calcular_metricas_pred, crear_tabla_prediccion_ic, resumen_intervalos
    )

warnings.filterwarnings('ignore')

# =============================================================================
# 1. RUTAS DE DATOS
# =============================================================================

dir_trabajo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

dir_datos = os.path.join(dir_trabajo, 'outputs')

dir_graficos = os.path.join(dir_trabajo, 'images')
os.makedirs(dir_graficos, exist_ok=True)

graficas = os.path.join(dir_graficos, 'comparacion_modelos')

if os.path.exists(graficas):
    shutil.rmtree(graficas)
os.makedirs(graficas, exist_ok=True)

# Datos generados por los scripts 03, 04, 06 y 08
dir_pickle_tt = os.path.join(dir_datos, 'pv_train_test.pkl')
dir_pickle_suav = os.path.join(dir_datos, 'modelo_suavizado_seleccion.pkl')
dir_pickle_arima = os.path.join(dir_datos, 'modelo_arima_seleccion.pkl')
dir_pickle_ets = os.path.join(dir_datos, 'modelo_est_esp.pkl')

# =============================================================================
# 2. CARGA DE DATOS
# =============================================================================
# 2.1. TRAIN / TEST
serie, serie_train, serie_test, m, datos_tt = cargar_train_test(dir_pickle_tt)

n_train = len(serie_train)
n_test = len(serie_test)

# 2.2. SUAVIZADO
suav = cargar_pickle(dir_pickle_suav)
suav_modelo = suav['modelo']
suav_nombre = suav['nombre']
suav_prediccion = suav['prediccion']
suav_prediccion = pd.Series(
    suav_prediccion.values, index=serie_test.index,name='Prediccion_Suavizado'
    )

# 2.3. ARIMA
arima = cargar_pickle(dir_pickle_arima)
arima_modelo = arima['modelo']
arima_nombre = arima['nombre']
arima_prediccion = arima['valores_prediccion']
arima_prediccion = pd.Series(
    arima_prediccion.values, index=serie_test.index, name='Prediccion_ARIMA'
    )

# 2.4. ETS
ets = cargar_pickle(dir_pickle_ets)
ets_modelo = ets['modelo']
ets_nombre = ets['nombre']
ets_prediccion = ets['valores_prediccion']
ets_prediccion = pd.Series(
    ets_prediccion.values, index=serie_test.index, name='Prediccion_ETS'
    )

# =============================================================================
# 3. TABLA DE COMPARACIÓN (TEST)
# =============================================================================
tabla_prediccion_error = pd.DataFrame({
    'Observado': serie_test,
    'Pred_Suavizado': suav_prediccion,
    'Pred_ARIMA': arima_prediccion,
    'Pred_ETS': ets_prediccion,
    'Error_Suavizado': serie_test - suav_prediccion,
    'Error_ARIMA': serie_test - arima_prediccion,
    'Error_ETS': serie_test - ets_prediccion,
    '|Error_S|': (serie_test - suav_prediccion).abs(),
    '|Error_A|': (serie_test - arima_prediccion).abs(),
    '|Error_ETS|': (serie_test - ets_prediccion).abs(),
    }).round(4)

# SUBTABLAS PREDICCIÓN / ERROR (DOSSIER)
tabla_prediccion = tabla_prediccion_error[[
    'Observado', 'Pred_Suavizado','Pred_ARIMA', 'Pred_ETS'
    ]].copy()

tabla_error = tabla_prediccion_error[[
    'Error_Suavizado', 'Error_ARIMA', 'Error_ETS', '|Error_S|', '|Error_A|',
    '|Error_ETS|'
    ]].copy()

mostrar_resultado('Predicciones finales en TEST', tabla_prediccion)

mostrar_resultado('Errores de predicción en TEST', tabla_error)

# =============================================================================
# 4. MÉTRICAS (TEST) - MAE y RMSE (+ MAPE)
# =============================================================================
metricas_error = []
metricas_error_suav = calcular_metricas_pred(
    suav_nombre, serie_test, suav_prediccion
    )
metricas_error_arima = calcular_metricas_pred(
    arima_nombre,serie_test,arima_prediccion
    )
metricas_error_ets = calcular_metricas_pred(
    ets_nombre, serie_test, ets_prediccion
    )

metricas_error.append(metricas_error_suav)
metricas_error.append(metricas_error_arima)
metricas_error.append(metricas_error_ets)

tabla_metricas = pd.DataFrame(metricas_error).set_index('modelo').round(4)

mostrar_resultado('Métricas finales de comparación', tabla_metricas)

# =============================================================================
# 5. INTERVALOS DE CONFIANZA (ARIMA Y ETS)
# =============================================================================
# 5.1. ARIMA
arima_ic = arima['prediccion'].conf_int()
arima_ic.columns = ['ic_inf', 'ic_sup']
tabla_arima_ic = crear_tabla_prediccion_ic(
    arima_nombre, serie_test, arima_prediccion, arima_ic['ic_inf'],
    arima_ic['ic_sup']
    ).round(4)

# 5.2. ETS
ets_ic = ets['prediccion'].summary_frame()[['pi_lower', 'pi_upper']]
ets_ic.columns = ['ic_inf', 'ic_sup']
tabla_ets_ic = crear_tabla_prediccion_ic(
    ets_nombre, serie_test, ets_prediccion, ets_ic['ic_inf'], ets_ic['ic_sup']
    ).round(4)

# 5.3. Tabla Intervalos - Comparación
tabla_ic = pd.DataFrame({
    'Observado': serie_test,
    'Error_ARIMA': serie_test - arima_prediccion,
    'Error_ets': serie_test - ets_prediccion,
    'IC_ARIMA':list(
        zip(arima_ic['ic_inf'].round(2), arima_ic['ic_sup'].round(2))
        ),
    'IC_ETS': list(
        zip(ets_ic['ic_inf'].round(2), ets_ic['ic_sup'].round(2))
        ),
    }).round(2)

mostrar_resultado('Comparación de intervalos ARIMA vs ETS', tabla_ic)

# 5.4. Resumen Intervalos
resumen_ic_arima = resumen_intervalos('ARIMA', tabla_arima_ic)
resumen_ic_ets = resumen_intervalos('ETS', tabla_ets_ic)

tabla_ic_resumen = pd.concat([resumen_ic_arima, resumen_ic_ets], axis=1)
tabla_ic_resumen.index.name = 'Metrica'

mostrar_resultado('Resumen de intervalos ARIMA vs ETS', tabla_ic_resumen)
# =============================================================================
# 6. GUARDADO DE TABLAS
# =============================================================================
dir_tablas = os.path.join(dir_datos, 'tablas')
os.makedirs(dir_tablas, exist_ok=True)

tabla_prediccion.to_csv(
    os.path.join(dir_tablas, '09_predicciones_test.csv'), sep=';', decimal=',',
    encoding='utf-8-sig'
    )

tabla_error.to_csv(
    os.path.join(dir_tablas, '09_errores_test.csv'), sep=';', decimal=',',
    encoding='utf-8-sig'
    )

tabla_metricas.to_csv(
    os.path.join(dir_tablas, '09_metricas_test.csv'), sep=';', decimal=',',
    encoding='utf-8-sig'
    )

tabla_arima_ic.to_csv(
    os.path.join(dir_tablas, '09_intervalos_arima.csv'), sep=';', decimal=',',
    encoding='utf-8-sig'
    )

tabla_ets_ic.to_csv(
    os.path.join(dir_tablas, '09_intervalos_ets.csv'), sep=';', decimal=',',
    encoding='utf-8-sig'
    )

tabla_ic.to_csv(
    os.path.join(dir_tablas, '09_comparacion_intervalos.csv'), sep=';',
    decimal=',', encoding='utf-8-sig'
    )

tabla_ic_resumen.to_csv(
    os.path.join(dir_tablas, '09_resumen_intervalos.csv'), sep=';',
    decimal=',', encoding='utf-8-sig'
    )

# =============================================================================
# 7. GRÁFICOS (evidencia visual)
# =============================================================================
abs_suav = tabla_error['|Error_S|']
abs_ari = tabla_error['|Error_A|']
abs_ets = tabla_error['|Error_ETS|']
error_suav = tabla_error['Error_Suavizado']
error_ari = tabla_error['Error_ARIMA']
error_ets = tabla_error['Error_ETS']

test_color = 'gold' # coincidencia de colores para las gráficas
s_color = 'magenta'
a_color = 'skyblue'
e_color = 'lime'

# 7.1 TEST + predicciones
plt.figure(figsize=(14, 8))
titulo = 'Comparación Predicción en TEST (2025)'

plt.plot(serie_test, label='TEST', color=test_color, linewidth=2)
plt.plot(
    suav_prediccion, label=f'Pred_{suav_nombre}', color=s_color, linewidth=2
    )
plt.plot(
    arima_prediccion, label=f'Pred_{arima_nombre}', color=a_color, linewidth=2
    )
plt.plot(
    ets_prediccion, label=f'Pred_{ets_nombre}', color=e_color, linewidth=2
    )

plt.title(titulo)
plt.xlabel('Fecha')
plt.ylabel('GWh')
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()

guardar_grafico(titulo, graficas)
plt.show()

# 7.2 ERROR ABSOLUTO
plt.figure(figsize=(14, 8))
titulo = 'Error Absoluto en TEST (2025)'

plt.plot(abs_suav, label=f'|Error| {suav_nombre}', color=s_color, linewidth=2)
plt.plot(abs_ari, label=f'|Error| {arima_nombre}', color=a_color, linewidth=2)
plt.plot(abs_ets, label=f'|Error| {ets_nombre}', color=e_color, linewidth=2)

plt.title(titulo)
plt.xlabel('Fecha')
plt.ylabel('|Error| (GWh)')
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()

guardar_grafico(titulo, graficas)
plt.show()

# 7.3 ERROR POR MES - BARRAS + LÍNEA 0
x = np.arange(len(serie_test.index))
width = 0.3

plt.figure(figsize=(14, 8))
titulo = 'Error (Observado - Predicción) por mes'

plt.bar(x-width, error_suav, width,label=f'Error {suav_nombre}', color=s_color)
plt.bar(x, error_ari, width, label=f'Error {arima_nombre}', color=a_color)
plt.bar(x+width, error_ets, width, label=f'Error {ets_nombre}', color=e_color)

plt.axhline(0, linestyle='--', linewidth=2)
plt.title(titulo)
plt.xlabel('Mes')
plt.ylabel('Error (GWh)')
plt.xticks(
    ticks=x, labels=[
        d.strftime('%Y-%m') for d in tabla_prediccion_error.index
    ], rotation=45
    )
plt.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()

guardar_grafico(titulo, graficas)
plt.show()

# 7.4 INTERVALOS DE CONFIANZA (ARIMA-ETS)
plt.figure(figsize=(14, 8))
titulo = 'TEST (2025): ARIMA vs ETSModel con IC95%'
plt.plot(serie_test, label='TEST', color=test_color, linewidth=2)
plt.plot(
    arima_prediccion, label=f'Pred_{arima_nombre}', color=a_color, linewidth=2
    )
plt.plot(
    ets_prediccion, label=f'Pred_{ets_nombre}', color=e_color, linewidth=2
    )

# Sombreado IC
plt.fill_between(
    serie_test.index, arima_ic['ic_inf'], arima_ic['ic_sup'], alpha=0.15, 
    color=a_color, label='IC95% ARIMA'
    )
plt.fill_between(
    serie_test.index, ets_ic['ic_inf'], ets_ic['ic_sup'], alpha=0.1,
    color=e_color, label='IC95% ETSModel'
    )

plt.title(titulo)
plt.xlabel("Fecha")
plt.ylabel("GWh")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()

guardar_grafico(titulo, graficas)
plt.show()