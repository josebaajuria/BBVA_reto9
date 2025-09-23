# INGESTA
# In: Datos/Originales/DowJones
# Out: Datos/Limpios/LOS DATASETS LIMPIOS

import packages.Preprocesamiento as ppr
import os
import pandas as pd
import datetime as dt

### CARGA DE DATOS
df = pd.read_csv('Datos/Originales/DowJones/x.csv')

### LIMPIEZA
## TIPOS DE VARIABLES

print(df.dtypes)

## DUPLICADOS
print(df[df.duplicated()].shape) #no hay

## TRATAR NAs
print(ppr.mirar_NA(df))

## CARPETA DE DATOS LIMPIOS
path_clean_data = os.path.join('Datos','Limpios')
if not os.path.exists(path_clean_data):
  os.makedirs(path_clean_data)

## GUARDAR DATOS LIMPIOS 
file_clean = os.path.join(path_clean_data,'y.csv')
ppr.save_clean_data(df, file_clean)

