# -*- coding: utf-8 -*-
"""
Series Temporales - Ejercicio de Evaluación

series_funciones.py
 Funciones auxiliares comunes del pipeline:
  - rutas
  - carga/guardado de pickle
  - exportación CSV
  - guardado de gráficos
  - normalización de series temporales
  - métricas de predicción
  - tablas homogéneas de predicción

Autor: Carlos Ayllón Motto
"""

#######################################################
# IMPORTACIÓN DE LIBRERÍAS
#######################################################
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    )

#######################################################
# PICKLE
#######################################################
def guardar_pickle(datos, ruta_pickle: str) -> None:
    """
    Guarda en formato pickle.
    """
    carpeta = os.path.dirname(ruta_pickle)
    if carpeta:
        os.makedirs(carpeta, exist_ok=True)

    with open(ruta_pickle, 'wb') as archivo:
        pickle.dump(datos, archivo)


def cargar_pickle(ruta_pickle: str):
    """
    Carga desde un pickle.
    """
    with open(ruta_pickle, 'rb') as archivo:
        return pickle.load(archivo)


#######################################################
# SERIES TEMPORALES
#######################################################
def asegurar_serie_mensual(serie, frec='MS'):
    salida = serie.copy()
    salida.index = pd.to_datetime(salida.index)
    salida = salida.asfreq(frec)
    salida.index.name = 'Fecha'
    return salida


def cargar_train_test(ruta_pickle: str, frec='MS'):
    datos = cargar_pickle(ruta_pickle)
    serie = datos.get('serie', None)
    train = asegurar_serie_mensual(datos['train'], frec=frec)
    test = asegurar_serie_mensual(datos['test'], frec=frec)
    m = int(datos['m'])

    if serie is not None:
        serie = asegurar_serie_mensual(serie, frec=frec)

    return serie, train, test, m, datos


#######################################################
# GRÁFICOS
#######################################################
def limpiar_nombre_archivo(nombre: str) -> str:
    """
    Limpia un texto para usarlo como nombre de archivo.
    """
    return (
        str(nombre)
        .replace(' ', '_')
        .replace('-', '')
        .replace('/', '_')
        .replace(':', '_')
        .replace('\\', '_')
        .replace('*', '')
        .replace('?', '')
        .replace('"', '')
        .replace('<', '')
        .replace('>', '')
        .replace('|', '')
        .replace('__', '_')
    )


def guardar_grafico(titulo: str, carpeta: str) -> str:
    """
    Guarda el gráfico actual en una carpeta específica.
    Parámetros:
     - nombre_archivo: Nombre del archivo (ej: "Serie observada")
     - Ruta
    Salida:
     - Ruta completa del archivo guardado.
    """
    os.makedirs(carpeta, exist_ok=True)
    nombre_limpio = limpiar_nombre_archivo(titulo)
    
    n_imagen = len([
        x for x in os.listdir(carpeta)
        if os.path.isfile(os.path.join(carpeta, x))
        ]) + 1
    
    nombre_archivo = f'{n_imagen:02}_{nombre_limpio}.png'
    ruta = os.path.join(carpeta, nombre_archivo)
    
    plt.savefig(ruta, dpi=400, bbox_inches='tight')

    return ruta


#######################################################
# MÉTRICAS DE PREDICCIÓN
########################################################
def calcular_metricas_pred(nombre: str, serie, prediccion) -> pd.Series:
    """
    Calcula métricas de predicción.
    Devuelve:
    - MAE
    - RMSE
    - MAPE
    """
    metricas = pd.Series({
        'modelo': nombre,
        'MAE': mean_absolute_error(serie, prediccion),
        'RMSE': np.sqrt(mean_squared_error(serie, prediccion)),
        'MAPE_%': (mean_absolute_percentage_error(serie, prediccion))*100
        })
    
    return metricas


#######################################################
# TABLAS HOMOGÉNEAS DE PREDICCIÓN
#######################################################
def crear_tabla_prediccion(nombre: str, serie, prediccion) -> pd.DataFrame:
    """
    Construye una tabla homogénea de predicción.

    Columnas base:
    - observado
    - prediccion
    - error
    - error_abs
    - ic_inf
    - ic_sup
    - modelo
    """
    observado = pd.Series(serie).copy()
    prediccion = pd.Series(prediccion, index=observado.index).copy()

    tabla = pd.DataFrame({
        'observado': observado,
        'prediccion': prediccion,
        'error': observado - prediccion,
        'error_abs': (observado - prediccion).abs(),
        })
       
    tabla.index.name = 'Fecha'
    
    return tabla


def crear_tabla_prediccion_ic(nombre: str, serie, prediccion, ic_inf, 
                              ic_sup) -> pd.DataFrame():
    """
    añade los intervalos de confianza a la tabla de predicción
    """
    tabla = crear_tabla_prediccion(nombre, serie, prediccion).copy()
    tabla['ic_inf'] = pd.Series(ic_inf, index=tabla.index)
    tabla['ic_sup'] = pd.Series(ic_sup, index=tabla.index)

    return tabla


def resumen_intervalos(nombre: str, tabla_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Requiere columnas:
     - observado
     - ic_inf
     - ic_sup
    """
    validos = tabla_pred[['observado', 'ic_inf', 'ic_sup']].dropna()
    
    meses = (
        (validos['observado'] >= validos['ic_inf'])
        & (validos['observado'] <= validos['ic_sup'])
        )
    
    dentro = meses.sum()
    nombre_mes = validos.index[meses].strftime('%b')
    cobertura = (meses.mean()*100).round(2)
    amplitud = (validos['ic_sup'] - validos['ic_inf']).mean()
    
    resumen_ic = pd.DataFrame({
        f'{nombre}':[nombre_mes.tolist(), dentro, cobertura, amplitud]
        }, index = ['Meses', 'Total', 'Cobertura', 'Amplitud'])
    
    return resumen_ic


#######################################################
# VISUALIZACIÓN DE RESULTADOS EN CONSOLA
#######################################################
def mostrar_resultado(titulo: str, datos, decimales: int=4) -> None:
    """
    Muestra de forma homogénea resultados relevantes en consola.

    Admite:
    - DataFrame
    - Series
    - diccionarios
    - valores simples

    Solo redondea valores numéricos, manteniendo sin cambios
    textos, fechas y otros tipos de datos.
    """
    print('\n' + '=' * 80)
    print(titulo)
    print('=' * 80)

    def redondear(valor):
        if (
            isinstance(valor, (int, float, np.integer, np.floating))
            and not isinstance(valor, bool)
        ):
            return round(valor, decimales)
        return valor
    
    # DataFrame
    if isinstance(datos, pd.DataFrame):
        salida = datos.copy().map(redondear)
        print(salida.to_string())

    # Series
    elif isinstance(datos, pd.Series):
        salida = datos.copy().map(redondear)
        print(salida.to_string())

    # Diccionarios
    elif isinstance(datos, dict):
        salida = pd.Series(datos).map(redondear)
        print(salida.to_string())

    # Otros tipos
    else:
        print(datos)


#######################################################
# FUNCIONES SUAVIZADO
#######################################################
def grafico_modelo(titulo: str, train, test, ajuste, prediccion, carpeta=None):
    """
    - titulo: Nombre del gráfico
    - serie_train y serie_test: Serie Original
    - fitted_train (TYPE): Suavizado
    - pred_test (TYPE): Predicción
    Salida: gráfico del modelo seleccionado
    """
    plt.figure(figsize=(14, 8))
    
    plt.plot(train, label='Train', color='lime', linewidth=1.5)
    plt.plot(test, label='Test', color='gold', linewidth=1.5)
    
    plt.plot(ajuste, label='Suavizada', color='skyblue', linewidth=1)
    plt.plot(
        prediccion, label='Predicción (TEST)', color='magenta', linewidth=1.5
        )
    
    # Punto de corte TRAIN/TEST
    plt.axvline(x=test.index.min(), linestyle='--', color='black', linewidth=2)
    plt.xlabel('Fecha')
    plt.ylabel('GWh')
    
    plt.title(titulo)
    plt.legend()

    if carpeta is not None:
        guardar_grafico(titulo, carpeta)

    plt.show()


#######################################################
# FUNCIONES ARIMA
#######################################################
def revision_dif(difer: dict) -> pd.Series:
    """
    Se obtienen los parámetros para analizar el resultado de la
    diferenciación con el criterio de raiz unitaria.
    """
    a_dif = adfuller(difer)
    
    resumen_dif = pd.Series({
        'Estadístico ADF': a_dif[0],
        'p_valor': a_dif[1],
        'Retardos': a_dif[2],
        'Observaciones': a_dif[3],
        'Crítico 1%': a_dif[4]['1%'],
        'Crítico 5%': a_dif[4]['5%'],
        'Crítico 10%': a_dif[4]['10%']
    })
    
    return resumen_dif


def grafico_residuos(modelo: str, resultados: pd.Series, carpeta: str):
    """
    Panel gráfico con los residuos de un modelo
     - Res. estandarizados
     - Q-Q
     - Histograma
     - Correlograma
    """
    fig = resultados.plot_diagnostics(lags=36, figsize=(14, 8))
    
    axes = fig.get_axes()
    
    for ax in axes:
        ax.grid(True, linestyle='--', alpha=0.5)
        axes[3].set_xticks(range(0, 37, 6))

    titulo = f'Diagnostico Residuos - {modelo}'
    plt.suptitle(titulo, fontsize=14)
    plt.tight_layout()
    
    guardar_grafico(titulo, carpeta)
    plt.show()


def comprobacion_resid(nombre_modelo: str, resultados: pd.Series, m: int, 
                       carpeta: str) -> pd.Series:
    """
    Analisis gráfico y numérico de la incorrelación en residuos
    Retardos:
    - Bajo: primero relevante sabiendo que es (m-grados de lib)>0
    - Estacionales (12, 24): estructura anual en serie mensual (m=12).
    - Verificación (1–12): control adicional hasta lag 12.
    - Retardo 24: diagnóstico de información.
    - rechazos en el 1er periodo
    - Comprobación ACF
    Validez (incorrelados)
    - bajo(p_valor) > 0.05
    - estacional(p_valor) > 0.05 (12)
    """
    residuos = resultados.resid.dropna()
    
    ret_lista = list(range(1, m * 2 + 1, 1))
    
    modelo_sm = resultados.model

    # Grados de libertad
    ljb_gl = (
        modelo_sm.k_ar_params + modelo_sm.k_ma_params +
        modelo_sm.k_seasonal_ar_params + modelo_sm.k_seasonal_ma_params
        )

    # Contraste de Ljung-Box para los retardos
    ljungb_contraste = acorr_ljungbox(
        residuos, lags=ret_lista, boxpierce=True, model_df=ljb_gl,
        return_df=True
        )
    ljungb_contraste2 = acorr_ljungbox(
        residuos, lags=ret_lista, boxpierce=True, model_df=0, return_df=True
        )

    lb_pvalor = ljungb_contraste.lb_pvalue
    lb_pvalor_inicio = ljungb_contraste2.lb_pvalue

    # Retardos relevantes - p_valor
    lb_lag3 = lb_pvalor_inicio.loc[3]
    lag_bajo = max(6, ljb_gl + 1)      # primer retardo relevante
    lb_lag_bajo = lb_pvalor.loc[lag_bajo]
    lag_estac = m                      # relevante por estacionalidad m=12
    lb_estacional1 = lb_pvalor.loc[lag_estac]
    lb_estacional2 = lb_pvalor.loc[lag_estac * 2]  # información y desempate

    lb_1_12 = lb_pvalor.loc[1:12]
    rechazos = (lb_1_12 <= 0.05).sum()
    ret_rechazados = lb_1_12[lb_1_12 <= 0.05].index.tolist()

    # Condiciones de validez de contraste
    lb_ok = lb_pvalor > 0.05
    contraste_valido = (
        ((lb_lag3 > 0.05) or (lb_lag_bajo > 0.05)) and (lb_estacional1 > 0.05)
        )

    # Revisión ACF
    raiz = (len(residuos)) ** 0.5
    limite = (2 / raiz) * 1.05   # Añado un 5% de margen (IC)
    
    acf_valores = acf(residuos, nlags=m + 1, fft=False)
    acf_valido = ('ok' if np.all(np.abs(acf_valores) <= limite) else 'no')

    ljungb = pd.Series({
        'modelo': nombre_modelo,
        'n_coef_modelo': ljb_gl,
        'P_LB(3)': lb_lag3,
        'Retardo_reg(gl)': lag_bajo,
        'P_LB(ret_reg)': lb_lag_bajo,
        'H0_Retardo_reg': 'ok' if lb_ok.loc[lag_bajo] else 'no',
        'LB(estac)': lag_estac,
        'P_LB(estac)': lb_estacional1,
        'H0_LB(estac)': 'ok' if lb_ok.loc[lag_estac] else 'no',
        'LB(estac2)': lag_estac * 2,
        'P_LB(estac2)': lb_estacional2,
        'P_LB(1:12)_rechazo': rechazos,
        'Retardos_rechazados': str(ret_rechazados),
        'Hipotesis_H0': 'ok' if contraste_valido else 'no',
        'ACF_limite_Retardos': 'ok' if acf_valido else 'no',
    })

    grafico_residuos(nombre_modelo, resultados, carpeta)
    
    return ljungb


def info_modelo(modelo: str, resultados: dict, ljungb: dict, 
                modelos_ari: list=None) -> pd.Series:
    """
    - Tabla con información relevante del modelo
    - Útil para comparaciones entre modelos
    - (AIC/BIC/LLF/sigma2 + parámetros estacionales y Ljung).
    - (opcional modelos_ari para tablas de comparación)
    """
    params = resultados.params

    # 1. Extraer AIC, BIC y Sigma2 del modelo
    aic = resultados.aic       # Valores menores parsimonia
    bic = resultados.bic
    log_lik = resultados.llf   # Bondad de ajuste Log-Likelihood
    sigma2 = params['sigma2']  # Varianza error modelo completo

    # p-values > 0.05
    p_values = resultados.pvalues.drop('sigma2')
    n_no_significativos = (p_values > 0.05).sum()

    # coeficientes
    ar_12 = params.get('ar.S.L12', np.nan)
    ma_12 = params.get('ma.S.L12', np.nan)
    ar_24 = params.get('ar.S.L24', np.nan)
    ma_24 = params.get('ma.S.L24', np.nan)
    p_ar12 = p_values.get('ar.S.L12', np.nan)
    p_ma12 = p_values.get('ma.S.L12', np.nan)
    p_ar24 = p_values.get('ar.S.L24', np.nan)
    p_ma24 = p_values.get('ma.S.L24', np.nan)

    # ljung
    lb_3 = ljungb.get('P_LB(3)', np.nan)
    ret_regular = ljungb.get('Retardo_reg(gl)', np.nan)
    lb_regular = ljungb.get('P_LB(ret_reg)', np.nan)
    ret_estac = ljungb.get('LB(estac)', np.nan)
    lb_estac = ljungb.get('P_LB(estac)', np.nan)
    lb_rechazos = ljungb.get('P_LB(1:12)_rechazo', np.nan)
    lb_valido = ljungb.get('Hipotesis_H0', 'no')

    informacion_modelo = pd.Series({
        'modelo': modelo,
        'AIC': aic,
        'BIC': bic,
        'Log-Likelihood': log_lik,
        'Varianza': sigma2,
        'Coef_AR12': ar_12,
        'pv_AR12': p_ar12,
        'Coef_MA12': ma_12,
        'pv_MA12': p_ma12,
        'Coef_AR24': ar_24,
        'pv_AR24': p_ar24,
        'Coef_MA24': ma_24,
        'pv_MA24': p_ma24,
        'no_signif': n_no_significativos,
        'P_LB(3)': lb_3,
        'LB(reg)': ret_regular,
        'P_LB(reg)': lb_regular,
        'LB(estac)': ret_estac,
        'P_LB(estac)': lb_estac,
        'LB_1:12_rechazos': lb_rechazos,
        'Contraste_LB': lb_valido
    })

    if modelos_ari is not None:
        modelos_ari.append(informacion_modelo)

    return informacion_modelo