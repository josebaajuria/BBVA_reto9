import pandas as pd
import numpy as np

# Función para arreglar las instancias fuera de horario de mercado
def assign_trading_day(ts, MARKET_CLOSE, MARKET_OPEN):
    local_time = ts.tz_convert('America/New_York').time()
    date = ts.tz_convert('America/New_York').date()

    if local_time < MARKET_OPEN:
        # Antes de la apertura → se asigna al día anterior
        new_date = pd.Timestamp(date) - pd.Timedelta(days=1)
    elif local_time > MARKET_CLOSE:
        # Después del cierre → se asigna al día siguiente
        new_date = pd.Timestamp(date) + pd.Timedelta(days=1)
    else:
        new_date = pd.Timestamp(date)

    return new_date

# Función para calcular rentabilidades anualizadas
def rtb_anualizada(tipo, n, p0, p1, dias_habiles):
    if tipo == 'd':
        r = (p1 - p0) / p0
        return ((1 + r) ** (dias_habiles / n)) - 1

    elif tipo == 'm':
        r = (p1 - p0) / p0
        return ((1 + r) ** (12 / n)) - 1

    elif tipo == 'y':
        r = (p1 - p0) / p0
        return ((1 + r) ** (1 / n)) - 1

    else:
        return("Por favor, introduzca 'd', 'm' o 'y' en el tipo")

# Hacemos una función que calcula las volatilidades diaria y anualizada, las rentabilidades media y anualizada. 
# Devuelve un dataframe en el que salen esas variables por símbolo.
def analisis_completo(precios, rentabilidades):

    # Cálculo de riesgo, rentabilidad media y Sharpe simulado
    volatilidad   = rentabilidades.std()
    volatilidad_anual = volatilidad * np.sqrt(252)
    rentabilidad  = rentabilidades.mean()
    df_riesgo_rent = pd.DataFrame({
        'Rentabilidad (media diaria)': rentabilidad,
        'Volatilidad (media diaria)': volatilidad,
        'Volatilidad_anual':  volatilidad_anual
    })
    df_riesgo_rent['Sharpe_sim'] = (
        df_riesgo_rent['Rentabilidad (media diaria)'] /
        df_riesgo_rent['Volatilidad (media diaria)']
    )

    # Rentabilidad anualizada (tipo 'd', 252 días hábiles)
    dias_habiles_anuales = 252
    n_periodo = len(precios.index)
    rent_anual = {
        sym: rtb_anualizada('d',
                            n_periodo,
                            precios[sym].iloc[0],
                            precios[sym].iloc[-1],
                            dias_habiles_anuales)
        for sym in precios.columns
    }
    df_riesgo_rent['Rent_anualizada'] = pd.Series(rent_anual)

    df_riesgo_rent = df_riesgo_rent.sort_values('Sharpe_sim', ascending=False)
    
    return df_riesgo_rent

def save_clean_data(df, file):
    df.to_csv(file, index=True, encoding='UTF-8')
    return None