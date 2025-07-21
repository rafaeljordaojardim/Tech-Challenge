import yfinance as yf
import pandas as pd
import numpy as np

# Lista de FIIs (com ".SA" para a B3 via yfinance)
fiis = ["ALZR11.SA", "BTLG11.SA", "HGRU11.SA", "RBRP11.SA",
        "RECR11.SA", "VGIP11.SA", "VISC11.SA", "VRTA11.SA", "XPLG11.SA"]

# Período de 12 meses
dados = yf.download(fiis, start="2024-07-01", end="2025-07-01", interval="1d")["Close"]
print("Dados baixados com sucesso!")

# Preenchendo valores faltantes com o último valor válido
dados = dados.fillna(method='ffill')

# Calculando retornos mensais
retornos_mensais = dados.resample("ME").last().pct_change().dropna()

# Calculando retorno e risco para cada FII
resultado = pd.DataFrame(index=retornos_mensais.columns)
resultado['Retorno_Medio_Anual'] = (1 + retornos_mensais.mean()) ** 12 - 1
resultado['Risco_Anual'] = retornos_mensais.std() * np.sqrt(12)

# Calculando matriz de covariância dos retornos mensais
cov_matrix = retornos_mensais.cov()

# Salvando os dados em CSVs
resultado.to_csv("fiis_dados.csv", float_format="%.6f")
cov_matrix.to_csv("fiis_cov_matrix.csv", float_format="%.8f")

# Exibir resultado
print(resultado.sort_values(by="Retorno_Medio_Anual", ascending=False))
print("\nMatriz de Covariância salva em 'fiis_cov_matrix.csv'")
