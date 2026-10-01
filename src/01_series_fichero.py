# -*- coding: utf-8 -*-
"""
Series Temporales - Ejercicio de Evaluación

01_series_fichero.py
Conversión de Excel (exportación Eurostat) a:
- CSV columnas
- Pickle con todas las series + metadatos

Autor: Carlos Ayllón Motto
"""
# Importación de librerías
import os
import re
import numpy as np
import pandas as pd
import warnings

from series_funciones import guardar_pickle, mostrar_resultado

warnings.filterwarnings('ignore')

# =============================================================================
# 1. RUTAS DE DATOS
# =============================================================================

dir_trabajo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

dir_datos = os.path.join(dir_trabajo, 'data')
nombre_excel = 'nrg_cb_pem__custom_20190130.xlsx'
dir_excel = os.path.join(dir_datos, nombre_excel)

dir_salida = os.path.join(dir_trabajo, 'outputs')
os.makedirs(dir_salida, exist_ok=True)

dir_csv = os.path.join(dir_salida, 'series_energia_solar.csv')
dir_pickle = os.path.join(dir_salida, 'series_energia_solar.pkl')
dir_reporte = os.path.join(dir_salida, 'reporte_series.txt')

# =============================================================================
# 2. FUNCIONES DE EXTRACCIÓN Y CONVERSIÓN
# =============================================================================
# 2.1. EXTRAER LOS DATOS Y CONVERTIRLO A SERIE (PANDAS)
#------------------------------------------------------
def extraer_eurostat(dir_excel, hoja, geo='Spain') -> pd.DataFrame():
    
    datos = pd.read_excel(dir_excel, sheet_name=hoja, header=None)

    fila_fecha = datos.index[datos[0].astype(str).str.strip().eq('TIME')][0]
    cabecera = datos.iloc[fila_fecha]

    cols_mes = []
    for c, v in cabecera.items():
        v = str(v).strip()
        if re.match(r'^\d{4}[-M]\d{2}$', v):
            cols_mes.append(c)

    fila_geo = datos.index[datos[0].astype(str).str.strip().eq(geo)][0]

    fechas = []
    for c in cols_mes:
        fechas.append(str(cabecera[c]).strip())

    valores = list(datos.loc[fila_geo, cols_mes].values)
    
    datos_out = pd.DataFrame({
        'Fecha': fechas,
        'Valor': valores
        })

    return datos_out

# 2.2. DEPURAR Y CONVERTIR LOS VALORES A FLOAT
#------------------------------------------------------
def depurar_valores(datos: pd.DataFrame) -> pd.DataFrame:
    df_valores = datos.copy()

    def convertir(x):
        if pd.isna(x):
            return np.nan

        # Si ya viene numérico, respetarlo
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)

        x = str(x).strip()

        if x in ['', ':', 'nan', 'NaN', 'None']:
            return np.nan

        x = x.replace(' ', '')
        x = re.sub(r'[a-zA-Z]+', '', x)

        # Caso europeo: 1.234,56
        if ',' in x and '.' in x:
            x = x.replace('.', '').replace(',', '.')
        # Caso decimal con coma: 1234,56
        elif ',' in x:
            x = x.replace(',', '.')
        # Caso decimal con punto: 1234.56 -> se deja tal cual

        try:
            return float(x)
        except:
            return np.nan

    df_valores['Valor'] = df_valores['Valor'].apply(convertir)
    return df_valores

# 2.3. INDEXAR LAS FECHAS DE LOS DATOS
#------------------------------------------------------
def index_fechas(datos: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte 'Fecha' a datetime
    - replace + strip
    - to_datetime con format %Y-%m
    """
    df_fecha = datos.copy()
    
    df_fecha['Fecha'] = (
        df_fecha['Fecha'].astype(str).str.replace('T', '-', regex=False)
                         .str.replace('M', '-', regex=False).str.strip()
                         )

    df_fecha['Fecha'] = pd.to_datetime(df_fecha['Fecha'], format='%Y-%m')

    df_fecha = df_fecha.sort_values('Fecha').set_index('Fecha')
    df_fecha.index.name = 'Fecha'
    return df_fecha

# 2.5. CONSTRUIR LA SERIE CON EL FORMATO ADECUADO
#------------------------------------------------------
def construir_serie(dir_excel: str, hoja: str, geo: str) -> pd.Series:
    df_serie = extraer_eurostat(dir_excel, hoja, geo=geo)
    df_serie = depurar_valores(df_serie)
    df_serie = index_fechas(df_serie)
    serie = df_serie['Valor'].copy()
    return serie

# =============================================================================
# 3. HOJAS DE SERIES
# =============================================================================
hojas = {
    'solar_total_gwh': 'Sheet 1',
    'solar_total_pct': 'Sheet 2',
    'solar_termica_gwh': 'Sheet 3',
    'solar_termica_pct': 'Sheet 4',
    'solar_fotovoltaica_gwh': 'Sheet 5',  # SERIE PRINCIPAL
    'solar_fotovoltaica_pct': 'Sheet 6'
    }

# =============================================================================
# 4. LECTURA, DATOS Y FECHA INDEXADA
# =============================================================================
geo = 'Spain' # Se podría cambiar por otros paises si se genera el archivo

series = {}
for nombre, hoja in hojas.items():
    series[nombre] = construir_serie(dir_excel, hoja, geo)

datos_input = pd.concat(series, axis=1).sort_index()

# =============================================================================
# 5. REPORTE DE CONTROL (FECHAS-MISSINGS)
# =============================================================================
freq_inferida = pd.infer_freq(datos_input.index)
reporte = {
    'archivo': nombre_excel,
    'geo': geo,
    'inicio': str(datos_input.index.min()),
    'fin': str(datos_input.index.max()),
    'n_filas': int(datos_input.shape[0]),
    'freq_inferida': str(freq_inferida),
    'missing_por_columna': datos_input.isna().sum().to_dict(),
    'negativos_por_columna': (datos_input < 0).sum().to_dict()
}

# Visualización rápida de los datos procesados
muestra_datos = pd.concat([datos_input.head(3), datos_input.tail(3)])

mostrar_resultado('Muestra del dataset procesado', muestra_datos)
mostrar_resultado('Reporte de control de datos', reporte)

with open(dir_reporte, 'w', encoding='utf-8') as archivo:
    archivo.write(str(reporte))
    
# =============================================================================
# 6. EXPORTAR A CSV
# =============================================================================
datos_input.reset_index().to_csv(
    dir_csv, index=False, encoding='utf-8-sig', sep=';', decimal = ','
    )

# =============================================================================
# 7. PICKLE (REUTILIZAR)
# =============================================================================
datos_salida = {
    'datos_input': datos_input,
    'hojas': hojas,
    'dir_excel': dir_excel,
    'geo': geo,
    'freq_inferida': freq_inferida
}

guardar_pickle(datos_salida, dir_pickle)