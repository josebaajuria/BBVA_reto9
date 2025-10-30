# INGESTA Y LIMPIEZA
# In: Datos/Originales/DowJones/Los 26 csv proporcionados
# Out: Datos/Limpios/LOS DATASETS LIMPIOS

import packages.Preprocesamiento as ppr
import pandas as pd
import numpy as np 
import os
import pytz
import pandas_market_calendars as mcal
from datetime import time
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import gmean

### CARGA DE DATOS
#Hacemos un bucle para meter todos los dataframes (26) en una lista y los vamos modificando.
carpeta = "Datos/Originales/DowJones"

# Lista de los archivos CSV en la carpeta
archivos = [f for f in os.listdir(carpeta) if f.endswith('.csv')]

lista_dfs = []

for i, archivo in enumerate(archivos):
    ruta_completa = os.path.join(carpeta, archivo)
    lista_dfs.append(pd.read_csv(ruta_completa))


### LIMPIEZA
# DEFINIMOS FUNCIÓN PARA DETERMINAR SI EL REGISTRO DE CADA INSTANCIA ESTÁ EN EL HORARIO DE APERTURA DEL MERCADO EN NUEVA YORK (9:30-16:30).
#Pasamos la fecha dada a horario de Nueva York, y si la hora es antes de las 9:30, el registro se pasa al día anterior, y si está después de las 16:30, se pasa al día siguiente.
# Ahora (tarda ~1min): 
# 1. Guardamos los días que está abierto ese mercado (el NYSE).
# 2. Ordenamos los dfs por la fecha.
# 3. La variable date la convertimos a tipo fecha (estaba como string).
# 4. Creamos la variable trading_day aplicando la función anterior, es decir, pasamos al horario de Nueva York y arreglamos los registros fuera de horario.
# 5. Por último, filtramos los dfs (variable trading_day) por los días que está abierto el NYSE.

# Calendario oficial NYSE
nyse = mcal.get_calendar("NYSE")
valid_days = nyse.valid_days(start_date="2018-01-01", end_date="2021-12-31").tz_convert('America/New_York').normalize()

# Rango horario del NYSE
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 30)

for df in lista_dfs:
    df.sort_values('date',inplace=True)

    # Convertir la columna 'date' a datetime con zona UTC
    df["date"] = pd.to_datetime(df["date"], utc=True)

    # Reasignar registros fuera del rango horario
    df["trading_day"] = df["date"].apply(ppr.assign_trading_day, MARKET_CLOSE=MARKET_CLOSE, MARKET_OPEN=MARKET_OPEN).dt.normalize()

    df["trading_day"] = df["trading_day"].dt.date
    valid_set = set(valid_days.date)

    # Filtrar solo días hábiles
    df = df[df["trading_day"].isin(valid_set)].copy()
    
    lista_dfs[i] = df

#MIRAMOS LA FECHA MÁS PEQUEÑA Y MÁS GRANDE DE CADA DF PARA PONERLOS TODOS EN EL MISMO RANGO
fechas = {'minima':[], 'maxima':[]}
for i,df in enumerate(lista_dfs):
    fecha_min = df['trading_day'].min()
    fecha_max = df['trading_day'].max()

    fechas['minima'].append(fecha_min)
    fechas['maxima'].append(fecha_max)


    print(f'Fecha min del df{i}: {fecha_min} y max: {fecha_max}')

print(f"La fecha más reciente para empezar las series: {pd.Series(fechas['minima']).max()} | La última fecha para acabar las series: {pd.Series(fechas['maxima']).min()}")

# LA SERIE QUE MAS TARDE EMPIEZA ES EN 2020-06-02, ASI QUE FILTRAMOS TODAS POR ESA FECHA PARA ADELANTE.\
# TODAS ACABAN EN 2021-07-01.
fecha_inicio = pd.Timestamp("2020-06-02", tz="America/New_York").date()

for i, df in enumerate(lista_dfs):
    df = df[df['trading_day'] >= fecha_inicio]
    lista_dfs[i] = df

##FRECUENCIA DIARIA
# PASAR A INDEX Y A DIARIO
for i, df in enumerate(lista_dfs):
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df.set_index("trading_day", inplace=True)
    simbolo = df['symbol'].values[0]
    df.drop(columns=['date','exchange','symbol'], inplace=True)
    df = df.resample("D").mean()
    df.loc[:,'symbol'] = simbolo

    lista_dfs[i] = df

##QUITAR NA-S
for i,df in enumerate(lista_dfs):
    df['close'] = df['close'].ffill().bfill()
    
    lista_dfs[i] = df

##AJUSTAMOS SPLITS
for i, df in enumerate(lista_dfs):
    # Descargamos splits reales
    symbol = df['symbol'].iloc[0]
    splits = yf.Ticker(symbol).splits

    if splits.empty:
        lista_dfs[i] = df
        continue

    # Normalizamos tz de splits para que coincida con df.index
    # Si splits.index viene sin tz, lo localizamos; si viene con UTC, lo convertimos
    if splits.index.tz is None:
        splits.index = splits.index.tz_localize("America/New_York")
    else:
        splits.index = splits.index.tz_convert("America/New_York")

    # Definimos el rango con tz NY
    fecha_min = pd.Timestamp("2020-06-02", tz="America/New_York")
    fecha_max = pd.Timestamp("2021-07-01", tz="America/New_York")

    # Filtramos splits dentro de la ventana
    splits = splits.loc[(splits.index >= fecha_min) & (splits.index <= fecha_max)]
    if splits.empty:
        lista_dfs[i] = df
        continue

    print(f"\n Splits reales para {symbol} entre {fecha_min.date()} y {fecha_max.date()}:")
    print(splits)

    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    else:
        df.index = df.index.tz_convert("America/New_York")

    # Ajustamos precios retrospectivamente
    for split_date, factor in splits.items():
        mask = df.index <= split_date
        for col in ['open','high','low','close']:
            if col in df.columns:
                df.loc[mask, col] /= factor

    lista_dfs[i] = df

##CALCULAMOS RENTABILIDAD DIARIA Y JUNTAMOS TODOS EN UN MISMO DF
dfs_pct = []

for i, df in enumerate(lista_dfs):
    # Quedarse solo con 'close' y 'symbol'
    df = df[["close", "symbol"]].copy()

    # Calcular variación porcentual diaria
    df["rent"] = df["close"].pct_change()

    # Eliminar la primera fila (NaN en pct_change)
    df = df.dropna(subset=["rent"])

    # Guardar en la lista
    dfs_pct.append(df)

# Unir todos los DataFrames en uno solo
df_final = pd.concat(dfs_pct)

# Hay problemas con fechas, lo arreglamos
df_final.index = df_final.index.astype(str)
df_final.index = df_final.index.map(lambda x: pd.to_datetime(x).replace(tzinfo=None))

# Creamos dos df nuevos con los simbolos como columnas, el de precios_close tendrá los valores de close y el de rentabilidades los valores de rentabilidad diaria.
grouped = df_final.groupby([df_final.index, 'symbol']).agg({
        'close': 'first',
        'rent':  'first'
    })

precios_close = grouped['close'].unstack('symbol')
rentabilidades = grouped['rent'].unstack('symbol')

print(precios_close,rentabilidades)

# USAMOS FUNCIÓN PARA CALCULAR VOLATILIDADES Y RENTABILIDADES DE LOS ACTIVOS
df_riesgo_rent = ppr.analisis_completo(precios=precios_close, rentabilidades=rentabilidades)
print(df_riesgo_rent)

## CARPETA DE DATOS LIMPIOS
path_clean_data = os.path.join('Datos','Limpios')
if not os.path.exists(path_clean_data):
  os.makedirs(path_clean_data)

## GUARDAR DATOS LIMPIOS 
file_clean = os.path.join(path_clean_data,'Precios_Close.csv')
ppr.save_clean_data(precios_close, file_clean)

file_clean = os.path.join(path_clean_data,'Rentabilidades.csv')
ppr.save_clean_data(rentabilidades, file_clean)

file_clean = os.path.join(path_clean_data,'Analisis_activos.csv')
ppr.save_clean_data(df_riesgo_rent, file_clean)

cartera_p = precios_close[['AAPL','MSFT','CAT']]
file_clean = os.path.join(path_clean_data,'Cartera_Close.csv')
ppr.save_clean_data(cartera_p, file_clean)

cartera_r = rentabilidades[['AAPL','MSFT','CAT']]
file_clean = os.path.join(path_clean_data,'Cartera_Rent.csv')
ppr.save_clean_data(cartera_r, file_clean)

# EN EL NOTEBOOK Simulacion_Monte_Carlo_AZUL_CLARO, se encuentra el mismo proceso de limpieza más explicado,
# además de gráficos y de la función de Monte Carlo con sus resultados y gráficos