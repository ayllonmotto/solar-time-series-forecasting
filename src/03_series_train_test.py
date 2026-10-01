# -*- coding: utf-8 -*-
"""
Series Temporales - Ejercicio de Evaluación

03_series_train_test.py
 - Partición TRAIN/TEST (m=12)
 - Reserva del último periodo estacional (TEST) para evaluación de predicciones

Autor: Carlos Ayllón Motto
"""
# Importación de librerías
import os
import pandas as pd
import matplotlib.pyplot as plt
import shutil

from series_funciones import (
    cargar_pickle, guardar_pickle, guardar_grafico, mostrar_resultado,
    asegurar_serie_mensual
    )

# =============================================================================
# 1. RUTAS DE DATOS
# =============================================================================

dir_trabajo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

dir_datos = os.path.join(dir_trabajo, 'outputs')

dir_images = os.path.join(dir_trabajo, 'images')
os.makedirs(dir_images, exist_ok=True)

graficas = os.path.join(dir_images, 'train_test')
if os.path.exists(graficas):
    shutil.rmtree(graficas)
os.makedirs(graficas, exist_ok=True)

dir_pickle_in = os.path.join(dir_datos, 'series_energia_solar.pkl')

dir_pickle_out = os.path.join(dir_datos, 'pv_train_test.pkl')

dir_csv_out = os.path.join(dir_datos, 'pv_train_test.csv')

# =============================================================================
# 2. CARGA DE DATOS (PICKLE DATOS DEPURADOS)
# =============================================================================
datos = cargar_pickle(dir_pickle_in)

datos_input = datos['datos_input']   # DataFrame depurado + Fecha index
  
# 2.2. SERIE PRINCIPAL (PV GWh)
pv_gwh = datos_input['solar_fotovoltaica_gwh'].copy()
pv_gwh = asegurar_serie_mensual(pv_gwh)

# =============================================================================
# 3. PARTICIÓN TRAIN/TEST
# =============================================================================
# Serie estacional mensual - reservamos 1 periodo: m = 12
m = 12 # TEST 11,1% aproximado de los datos
test = pv_gwh.iloc[-m:]
train = pv_gwh.iloc[:-m]

# Control básico de tamaños y rangos
info_particion = {
    'm': m,
    'n_total': int(pv_gwh.shape[0]),
    'n_train': int(train.shape[0]),
    'n_test': int(test.shape[0]),
    'train_inicio': str(train.index.min().date()),
    'train_fin': str(train.index.max().date()),
    'test_inicio': str(test.index.min().date()),
    'test_fin': str(test.index.max().date())
}

mostrar_resultado('Partición temporal TRAIN / TEST', info_particion)

mostrar_resultado(
    'Periodo TEST reservado para validación', test.to_frame('PV_GWh')
    )

# =============================================================================
# 4. GRÁFICO TRAIN vs TEST
# =============================================================================
plt.figure(figsize=(14, 8))
titulo = 'Partición TRAIN/TEST - Solar fotovoltaica (GWh)'
plt.plot(train, label='TRAIN (ajuste)', color = 'gold', linewidth=1.5)
plt.plot(test, label='TEST (validación)', color = 'skyblue', linewidth=1.5)

# Línea de corte
plt.axvline(x=test.index.min(), linestyle='--', color='magenta', linewidth=2)
plt.title(titulo)
plt.xlabel('Fecha')
plt.ylabel('GWh')
plt.legend()
guardar_grafico(titulo, graficas)
plt.show()

# =============================================================================
# 5. GUARDADO DATOS PARA REUTILIZAR (PICKLE + CSV)
# =============================================================================
train_test = {
    'serie': pv_gwh,
    'train': train,
    'test': test,
    'm': m,
    'particion': info_particion
    }

guardar_pickle(train_test, dir_pickle_out)

train_test_df = pd.concat([
    train.to_frame('PV_GWh').assign(Particion='TRAIN'),
    test.to_frame('PV_GWh').assign(Particion='TEST')
    ])
train_test_df = train_test_df.reset_index().set_index('Fecha')
train_test_df.index = pd.to_datetime(train_test_df.index)

train_test_df.to_csv(
    dir_csv_out, index=True, encoding='utf-8-sig', sep=';', decimal = ','
    )