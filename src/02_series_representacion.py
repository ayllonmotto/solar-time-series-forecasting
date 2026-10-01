# -*- coding: utf-8 -*-
"""
Series Temporales - Ejercicio de Evaluación

02_series_representacion.py
 - Representación gráfica de la serie (solar fotovoltaica)
 - Descomposición estacional de la serie principal
 - Tabla de coeficientes estacionales
 - Comparación con otras series solares

Autor: Carlos Ayllón Motto

"""
# Importación de librerías
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

from statsmodels.tsa.seasonal import seasonal_decompose

from series_funciones import (
    cargar_pickle, guardar_grafico, mostrar_resultado
    )

# =============================================================================
# 1. RUTAS DE DATOS
# =============================================================================

dir_trabajo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

dir_datos = os.path.join(dir_trabajo, 'outputs')

dir_images = os.path.join(dir_trabajo, 'images')
os.makedirs(dir_images, exist_ok=True)

graficas = os.path.join(dir_images, 'serie_base')
if os.path.exists(graficas):
    shutil.rmtree(graficas)
os.makedirs(graficas, exist_ok=True)

dir_pickle = os.path.join(dir_datos, 'series_energia_solar.pkl')

dir_tabla_est = os.path.join(dir_datos, 'tabla_coef_estacionalidad_PV.csv')

# =============================================================================
# 2. CARGA DE DATOS PICKLE
# =============================================================================
datos = cargar_pickle(dir_pickle)
datos_input = datos['datos_input'].copy()


# =============================================================================
# 3. CREACIÓN SERIES Y OBJETOS PARA GRÁFICAS
# =============================================================================
# Serie principal
pv_gwh = datos_input['solar_fotovoltaica_gwh'].copy()
pv_gwh.index = pd.to_datetime(pv_gwh.index)
pv_gwh.name = 'Solar Fotovoltaica (GWh)'

# DataFrame comparación (GWh)
solar_gwh = datos_input[[
    'solar_total_gwh', 'solar_termica_gwh', 'solar_fotovoltaica_gwh']
    ].copy()
solar_gwh.index = pd.to_datetime(solar_gwh.index)
solar_gwh.columns = [
    'Solar Total (GWh)', 'Solar Térmica (GWh)', 'Solar Fotovoltaica (GWh)'
    ]
color_solar = ['lime', 'gold', 'skyblue']

# DataFrame comparación (%)
solar_pct = datos_input[[
    'solar_total_pct', 'solar_termica_pct', 'solar_fotovoltaica_pct'
    ]].copy()
solar_pct.index = pd.to_datetime(solar_pct.index)
solar_pct.columns = [
    'Solar Total (%)', 'Solar Térmica (%)', 'Solar Fotovoltaica (%)'
    ]
color_solarpct = ['lime', 'gold', 'skyblue']

# =============================================================================
# 4. REPRESENTACIÓN GRÁFICA
# =============================================================================
# 4.1 Series solares (GWh)
solar_gwh.plot(figsize=(14, 8), color=color_solar)
titulo = 'Generación neta mensual - Series solares (GWh) - España'
plt.title(titulo)
plt.xlabel('Fecha')
plt.ylabel('GWh')
plt.legend()
guardar_grafico(titulo, graficas)
plt.show()

# 4.2 Serie principal (PV en GWh)
pv_gwh.plot(figsize=(14, 8), color='gold')
titulo = 'Generación neta mensual - Solar fotovoltaica (GWh) - España'
plt.title(titulo)
plt.xlabel('Fecha')
plt.ylabel('GWh')
plt.legend()
guardar_grafico(titulo, graficas)
plt.show()

# 4.3 Serie principal (2025)
solar_25 = pv_gwh[-12:]
solar_25.plot(figsize=(14, 8), color='gold')
titulo = 'Generación neta mensual - Solar fotovoltaica (GWh) - España'
plt.title(titulo)
plt.xlabel('Mes')
plt.ylabel('GWh')
plt.legend()
guardar_grafico(titulo, graficas)
plt.show()

# 4.4 Serie principal (2021-2025)
solar_2125 = pv_gwh['2021':'2025']
solar_2125.plot(figsize=(14, 8), color='gold')
titulo = 'Generación neta mensual - Solar fotovoltaica (GWh) - España'
plt.title(titulo)
plt.xlabel('Mes')
plt.ylabel('GWh')
plt.legend()
guardar_grafico(titulo, graficas)
plt.show()

# 4.5 Series solares (%)
solar_pct.plot(figsize=(14, 8), color=color_solarpct)
titulo = 'Series solares mensuales (%) - España'
plt.title(titulo)
plt.xlabel('Fecha')
plt.ylabel('%')
plt.legend()
guardar_grafico(titulo, graficas)
plt.show()

# =============================================================================
# 5. DESCOMPOSICIÓN ESTACIONAL
# =============================================================================
# Decisión del modelo:
modelo = 'multiplicative' if (pv_gwh.dropna() > 0).all() else 'additive'
# Descomposición estacional modelo multiplicativo
des_est = seasonal_decompose(pv_gwh, model=modelo, period=12)

# Representación de componentes: Serie, Tendencia, coef_est, Ruido
plt.rc('figure', figsize=(16, 14))
plt.rc('font', size=11)

fig = des_est.plot()
titulo = 'Descomposición Estacional - Solar Fotovoltaica (GWh)'
fig.suptitle(titulo, y=1)

colores = ['gold', 'turquoise', 'royalblue', 'coral'] # por gráfica
labels = ['Observed', 'Trend', 'Seasonal', 'Resid']
titulos = ['Serie Observada', 'Tendencia', 'Estacionalidad', 'Ruido']

for i, ax in enumerate(fig.get_axes()):
    line = ax.get_lines()[0]
    line.set_color(colores[i])
    if i == 3:
        for line in ax.get_lines():
            line.set_color(colores[i])
    ax.set_ylabel(labels[i], rotation=90, labelpad=20)
    ax.set_title(titulos[i])

plt.tight_layout()
guardar_grafico(titulo, graficas)
plt.show()

# 5.1. TABLA DE COEFICIENTES ESTACIONALES
meses = [
    'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
    'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'
    ]
coef_est = des_est.seasonal.dropna().iloc[-12:]

tabla_est = pd.DataFrame({
    'Mes_num': coef_est.index.month,
    'Mes': [meses[m - 1] for m in coef_est.index.month],
    'Coef_estacional': coef_est.values.round(4)
    })

tabla_est['Variacion_%'] = ((tabla_est['Coef_estacional'] - 1) * 100).round(2)
tabla_est = tabla_est.set_index('Mes_num')
tabla_est.to_csv(
    dir_tabla_est, index=False, encoding='utf-8-sig', sep=';', decimal = ','
    )

mostrar_resultado('Coeficientes estacionales - Solar fotovoltaica', tabla_est)

                        
# 5.2. GRÁFICO SERIE AJUSTADA ESTACIONAL
pv_ajust = pv_gwh / des_est.seasonal # definimos antes modelo multi

plt.figure(figsize=(14, 8))
titulo = f'Serie original - ajustada estacionalmente ({modelo}) - PV (GWh)'

plt.plot(pv_gwh, label='Serie original', color='gold', linewidth=1.5)
plt.plot(pv_ajust, label='Serie ajustada', color='magenta', linewidth=1.5)
plt.plot(des_est.trend, label='Tendencia', color='turquoise', linewidth=2.5)

plt.title(titulo)
plt.xlabel('Fecha')
plt.ylabel('GWh')
plt.legend()
guardar_grafico(titulo, graficas)
plt.show()

# 5.3.. GRÁFICO ESTACIONAL POR AÑO
# Usamos el DF para el grafico anual
solar_estacional = solar_gwh.copy()
solar_estacional['Año'] = solar_estacional.index.year
solar_estacional['Mes'] = solar_estacional.index.month
pv_anual = solar_estacional['Solar Fotovoltaica (GWh)']

plt.figure(figsize=(14, 8))
titulo = 'Gráfico estacional por año - Solar fotovoltaica (GWh)'

sns.lineplot(
    x=solar_estacional['Mes'], y=pv_anual, hue=solar_estacional['Año'],
    palette='Set3', estimator=None, linewidth=2, legend='full'
    )

plt.xlabel('Mes')
plt.ylabel('PV (GWh)')
plt.title(titulo)
plt.legend(title='Año', loc='upper left', bbox_to_anchor=(1, 1))
guardar_grafico(titulo, graficas)
plt.show()