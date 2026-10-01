# -*- coding: utf-8 -*-
"""
Pipeline completo - Series Temporales

Ejecuta secuencialmente:
01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 08 -> 09

Muestra en terminal:
- Etapa que se está ejecutando.
- Salida generada por cada script.
- Archivos de datos/modelos generados o actualizados.
- Gráficos generados o actualizados.
- Tiempo de ejecución por etapa y total.

Autor: Carlos Ayllón Motto
"""

from pathlib import Path
import runpy
import sys
import time


# =============================================================================
# 1. RUTAS
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / 'src'

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

OUTPUT_DIR = BASE_DIR / 'outputs'
IMAGES_DIR = BASE_DIR / 'images'


# =============================================================================
# 2. PIPELINE
# =============================================================================
PIPELINE = [
    ('01', 'Preparación de datos', '01_series_fichero.py'),
    ('02', 'Representación y estacionalidad', '02_series_representacion.py'),
    ('03', 'Partición TRAIN / TEST', '03_series_train_test.py'),
    ('04', 'Suavizado exponencial', '04_series_suavizado.py'),
    ('05', 'ARIMA manual', '05_series_arima_manual.py'),
    ('06', 'ARIMA automático y selección', '06_series_arima_automatico.py'),
    ('07', 'Predicción ARIMA', '07_series_arima_prediccion.py'),
    ('08', 'Modelado ETS', '08_series_modelado_ets.py'),
    ('09', 'Comparación final', '09_series_comparacion_final.py'),
]


# =============================================================================
# 3. FUNCIONES AUXILIARES
# =============================================================================
def obtener_archivos(carpeta):
    """Devuelve archivo -> fecha de modificación."""
    
    if not carpeta.exists():
        return {}

    return {
        archivo: archivo.stat().st_mtime
        for archivo in carpeta.rglob('*')
        if archivo.is_file()
    }


def cambios_generados(antes, despues):
    """Detecta archivos nuevos o modificados."""

    cambios = []

    for archivo, fecha in despues.items():

        if archivo not in antes:
            cambios.append(('NUEVO', archivo))

        elif fecha > antes[archivo]:
            cambios.append(('ACTUALIZADO', archivo))

    return cambios


def mostrar_cambios(titulo, cambios):

    if not cambios:
        return

    print(f'\n{titulo}')

    for estado, archivo in cambios:
        print(f'  [{estado}] {archivo.relative_to(BASE_DIR)}')


# =============================================================================
# 4. EJECUCIÓN DE UNA ETAPA
# =============================================================================
def ejecutar_etapa(paso, nombre, script):

    script_path = SRC_DIR / script

    if not script_path.exists():
        raise FileNotFoundError(f'No se encuentra el script: {script_path}')

    print('\n' + '=' * 80)
    print(f'PASO {paso} | {nombre}')
    print('=' * 80)
    print(f'Script: {script}')
    print('-' * 80)

    # Estado previo
    outputs_antes = obtener_archivos(OUTPUT_DIR)
    images_antes = obtener_archivos(IMAGES_DIR)

    inicio = time.perf_counter()

    # ---------------------------------------------------------
    # Ejecución
    # ---------------------------------------------------------
    runpy.run_path(str(script_path), run_name='__main__')

    tiempo = time.perf_counter() - inicio

    # Estado posterior
    outputs_despues = obtener_archivos(OUTPUT_DIR)
    images_despues = obtener_archivos(IMAGES_DIR)

    cambios_outputs = cambios_generados(outputs_antes, outputs_despues)
    cambios_images = cambios_generados(images_antes, images_despues)

    # ---------------------------------------------------------
    # Resumen de la etapa
    # ---------------------------------------------------------
    mostrar_cambios('ARCHIVOS / RESULTADOS GENERADOS:', cambios_outputs)

    mostrar_cambios('GRÁFICOS GENERADOS:', cambios_images)

    print('\n' + '-' * 80)
    print(f'PASO {paso} FINALIZADO CORRECTAMENTE ({tiempo:.2f} s)')


# =============================================================================
# 5. EJECUCIÓN DEL PIPELINE COMPLETO
# =============================================================================
def ejecutar_pipeline():

    print('\n' + '=' * 80)
    print('PIPELINE | SERIES TEMPORALES')
    print('=' * 80)

    print(f'\nPython: {sys.executable}')
    print(f'Proyecto: {BASE_DIR}')

    inicio_total = time.perf_counter()

    for paso, nombre, script in PIPELINE:
        ejecutar_etapa(paso, nombre, script)

    tiempo_total = time.perf_counter() - inicio_total

    print('\n' + '=' * 80)
    print('PIPELINE FINALIZADO CORRECTAMENTE')
    print(f'Tiempo total: {tiempo_total:.2f} s')
    print('=' * 80)


# =============================================================================
# 6. EJECUCIÓN
# =============================================================================
if __name__ == '__main__':
    ejecutar_pipeline()