## ENTRADA: CSV DEL FIREWALL LIMPIO
## IMPORTANTE: PRIMERO EJECUTAR EL _01_Ingesta_Limpieza.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Datos/Limpios/y.csv')

