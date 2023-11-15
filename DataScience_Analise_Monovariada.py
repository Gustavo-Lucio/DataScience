# coding: utf-8

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression

# ,index_col=['Ano', 'Mês']

# df = pd.read_csv('D:/DS/Vendas.csv',sep=';', encoding='ISO-8859-1')
# df['Data'] = pd.to_datetime(df['Ano'].astype(str) + '-' + df['Mês'].astype(str), format='%Y-%m')
# df.set_index('Data', inplace=True)
# df.head()

serie = [10.5,11.2,12,13.5,15.2,14.8,14.6,15.3,16,17.2,16.5,18,10.8,11.4,12.2,13.7,15.6,15,14.9,15.8,16.3,17.4,16.8,18.5,11.2,12,12.8,14.2,16,15.4,15.2,16.1,17,17.8,17.1,19,11.5,12.3,13.2]

serie = np.array(serie)


# 1 - Exercicio
media = round(serie.mean(),2)

subgrupo = 12
medianas = [np.median(serie[i:i + subgrupo]) for i in range(0, len(serie), subgrupo)]

desvioPadrao = serie.std()

variancia = np.var(serie)
desvio_padrao = np.std(serie)

venda_min = serie.min()
venda_max = serie.max()


# 2 - Exercicio
quartis = np.percentile(serie, [25, 50, 75])

q1 = quartis[0]  # Primeiro quartil (25%)
q2 = quartis[1]  # Segundo quartil (mediana - 50%)
q3 = quartis[2]  # Terceiro quartil (75%)
iqr = q3 - q1

serie_media = np.zeros(len(serie))
serie_media[:] = serie.mean()

serie_limite_superior = np.zeros(len(serie))
serie_limite_superior[:] = serie.mean() + serie.std()

serie_limite_inferior = np.zeros(len(serie))
serie_limite_inferior[:] = serie.mean() - serie.std()

print(f'A média de vendas durante os anos 2024 à 2027 é de {media} (em unidades monetárias).')

print(f'A mediana do ano de 2024 é de {medianas[0]}.')
print(f'A mediana do ano de 2025 é de {medianas[1]}.')
print(f'A mediana do ano de 2026 é de {medianas[2]}.')
print(f'A mediana do ano de 2027 é de {medianas[3]}.')

print(f'Variância: {variancia}')
print(f'Desvio Padrão: {desvio_padrao}')

print(f'O mês com a menor venda foi Jan/2024 e o valor de venda foi: {venda_min}')
print(f'O mês com a maior venda foi Dez/2026 e o valor de venda foi:: {venda_max}')

print("Primeiro Quartil (Q1):", q1)
print("Segundo Quartil (Mediana - Q2):", q2)
print("Terceiro Quartil (Q3):", q3)
print("Intervalo Interquartil (IQR):", iqr)

#Monta o gráfico
sns.lineplot(data=serie, label='Observações')
sns.lineplot(data=serie_media, label='Média')
sns.lineplot(data=serie_limite_superior, label='Limite superior')
sns.lineplot(data=serie_limite_inferior, label='Limite inferior')
plt.title("Exemplo de gráfico de controle")
plt.legend()
plt.show()


#Forecasting ~~~~~~~~~~
meses = np.arange(1, len(serie) + 10).reshape(-1, 1)
modelo = LinearRegression()
modelo.fit(meses[:len(serie)], serie)
previsao_2028 = modelo.predict(meses[len(serie):])

print("Previsão de Vendas de abrir de 2027 até 2028:")

for i, valor in enumerate(previsao_2028):
    print(f"Previsão para o mês {i + 1}: {round(valor,2)}")

print(" ")

erro_medio = np.mean(previsao_2028 - media)
print(f'O Erro médio da analise de tendencia é: {round(erro_medio,2)}')

erro_percentual_medio = np.mean((previsao_2028 - media) / media) * 100
print(f'O Erro médio da analise de tendencia é: {round(erro_percentual_medio,2)} %')
print(" ")

# Previsao_2023
meses = np.arange(1, len(previsao_2028) + 61).reshape(-1, 1)
modelo = LinearRegression()
modelo.fit(meses[:len(previsao_2028)], previsao_2028)
previsao_2032 = modelo.predict(meses[len(previsao_2028):])

print("Previsão de Vendas de abrir de 2028 até 2032:")
for i, valor in enumerate(previsao_2032):
    print(f"Previsão para o mês {i + 1}: {round(valor,2)}")

print(" ")
