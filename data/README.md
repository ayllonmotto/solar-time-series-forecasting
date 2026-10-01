# Datos

## Fuente

**Eurostat**  
Dataset: **Net electricity generation by type of fuel – monthly data**  
Código: `nrg_cb_pem`

Extracción personalizada:

https://ec.europa.eu/eurostat/databrowser/view/nrg_cb_pem__custom_20190130/default/table

## Fichero esperado

`src/01_series_fichero.py` espera:

```text
data/nrg_cb_pem__custom_20190130.xlsx
```

El fichero original no se incluye en este paquete. Debe descargarse desde Eurostat y guardarse con ese nombre antes de ejecutar el pipeline completo.

## Alcance utilizado

- País: Spain
- Periodicidad: mensual
- Periodo: 2017-01 a 2025-12
- Serie principal: Solar photovoltaic
- Unidad principal: Gigawatt-hour (GWh)

La fase descriptiva utiliza además solar total, solar térmica y sus porcentajes.

Los CSV procesados y resultados seleccionados se incluyen en `outputs/`.
