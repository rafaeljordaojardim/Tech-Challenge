import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns


# Atualiza e importa os dados - PESSOA 1

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

print("\n")


# === Configuração do AG
N_IND = 100
N_GEN_WITHOUT_IMPROVE_FITNESS = 0
MAX_N_GEN_WITHOUT_IMPROVE_FITNESS = 1000
TAXA_CROSSOVER = 0.7
TAXA_MUTACAO = 0.5
AMPLITUDE_MUTACAO = 0.15
NUM_GENERATION_COUNTER = 1
ELITE = 1
max_fitness = -np.inf

# Para gráficos
historico_melhor_fitness = []
historico_fitness_medio = []
historico_risco = []
historico_retorno = []

np.random.seed(42)
random.seed(42)

# === Dados dos FIIs ===
dados = pd.read_csv("fiis_dados.csv", index_col=0)
tickers = dados.index.tolist()
retornos = dados["Retorno_Medio_Anual"].values
risco = dados["Risco_Anual"].values

# Carregar matriz de covariância
cov_matrix = pd.read_csv("fiis_cov_matrix.csv", index_col=0)
cov_matrix = cov_matrix.loc[tickers, tickers]
cov_matrix_values = cov_matrix.values

def calcular_retorno(weights):
    weights = weights / weights.sum()
    return np.dot(weights, retornos)

def calcular_risco(weights):
    weights = weights / weights.sum()
    return np.sqrt(weights @ cov_matrix_values @ weights.T)

def calcular_diversificacao(weights):
    weights = weights / weights.sum()
    weights = np.where(weights == 0, 1e-6, weights)
    entropia = -np.sum(weights * np.log(weights))
    return entropia

def get_weights(item):
    weights = np.array(item)
    weights = weights / weights.sum()
    return weights

ALPHA = 0.6
BETA = 0.2
GAMMA = 0.2

def fitness(item):
    weights = get_weights(item)
    retorno = calcular_retorno(weights)
    risco = calcular_risco(weights)
    diversificacao = calcular_diversificacao(weights)
    retorno_norm = retorno
    risco_norm = risco
    diversificacao_norm = diversificacao / np.log(len(weights))
    score = ALPHA * retorno_norm - BETA * risco_norm + GAMMA * diversificacao_norm
    return score

def criar_individuo():
    ind = np.random.rand(len(tickers))
    ind /= ind.sum()
    return ind

def crossover(p1, p2):
    if random.random() < TAXA_CROSSOVER:
        ponto = random.randint(random.randint(0, len(p1) - 2), len(p1) - 1)
        f1 = np.concatenate([p1[:ponto], p2[ponto:]])
        f2 = np.concatenate([p2[:ponto], p1[ponto:]])
        return f1 / f1.sum(), f2 / f2.sum()
    else:
        return p1.copy(), p2.copy()

def mutacao_com_parametros(ind, taxa_mut, amp_mut):
    ind = ind.copy()
    for _ in range(len(ind)):
        if random.random() < taxa_mut:
            i, j = np.random.choice(len(ind), 2, replace=False)  # escolhe dois ativos diferentes
            delta = np.random.uniform(0, amp_mut) * ind[i]  # fração do ativo i
            ind[i] -= delta
            ind[j] += delta
    ind /= ind.sum()
    return ind

# === Execução principal ===
pop = [criar_individuo() for _ in range(N_IND)]
best_result_first_generation = None

while True:
    if N_GEN_WITHOUT_IMPROVE_FITNESS > 20:
        taxa_mut = 0.5
        amp_mut = 0.3
    else:
        taxa_mut = TAXA_MUTACAO
        amp_mut = AMPLITUDE_MUTACAO

    fitnesses = [fitness(ind) for ind in pop]
    elite_indices = np.argsort(fitnesses)[-ELITE:]
    elite = [pop[i] for i in elite_indices]

    if NUM_GENERATION_COUNTER % 10 == 0:
        n_substituir = math.ceil(N_IND * 0.1)
        indices_piores = np.argsort(fitnesses)[:n_substituir]
        for i in indices_piores:
            pop[i] = criar_individuo()

    nova_pop = elite[:]
    while len(nova_pop) < N_IND:
        pais = random.choices(pop, weights=fitnesses, k=2)
        filho1, filho2 = crossover(pais[0], pais[1])
        filho1 = mutacao_com_parametros(filho1, taxa_mut, amp_mut)
        filho2 = mutacao_com_parametros(filho2, taxa_mut, amp_mut)
        nova_pop.extend([filho1, filho2])

    pop = nova_pop[:N_IND]
    melhor_idx = np.argmax(fitnesses)
    best = pop[melhor_idx]
    best_fitness = fitnesses[melhor_idx]

    # ⏺️ Armazenar histórico para gráficos
    media_fitness = np.mean(fitnesses)
    historico_melhor_fitness.append(best_fitness)
    historico_fitness_medio.append(media_fitness)
    historico_retorno.append(calcular_retorno(best))
    historico_risco.append(calcular_risco(best))

    if best_fitness > max_fitness:
        N_GEN_WITHOUT_IMPROVE_FITNESS = 0
        max_fitness = best_fitness
    else:
        N_GEN_WITHOUT_IMPROVE_FITNESS += 1
        if N_GEN_WITHOUT_IMPROVE_FITNESS >= MAX_N_GEN_WITHOUT_IMPROVE_FITNESS:
            #print("🔚 Algoritmo parou por não melhorar o fitness.")
            break
        #print(f"Melhor fitness não melhorou: {best_fitness:.6f}")

    retorno_esperado = np.dot(best, retornos)
    risco_real = np.sqrt(best @ cov_matrix_values @ best.T)
    diversificacao = calcular_diversificacao(best)
    div_norm = diversificacao / np.log(len(best))
    if (NUM_GENERATION_COUNTER == 1) and (best_result_first_generation is None):
        best_result_first_generation = best
    #print(f"Geração {NUM_GENERATION_COUNTER}: Fitness = {best_fitness:.6f}, Retorno = {retorno_esperado:.6f}, Risco = {risco_real:.6f}, Diversificação = {div_norm:.4f}")
    NUM_GENERATION_COUNTER += 1

print("\n")

# === GRÁFICO 1: Evolução do Fitness ===
plt.figure(figsize=(10, 5))
plt.plot(historico_melhor_fitness, label="Melhor Fitness", marker='o')
plt.plot(historico_fitness_medio, label="Fitness Médio", linestyle='--', marker='x')
plt.xlabel("Geração")
plt.ylabel("Fitness")
plt.title("Evolução do Fitness ao Longo das Gerações")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n")
# === GRÁFICO 2: Risco x Retorno (últimos indivíduos) ===
plt.figure(figsize=(8, 6))
plt.scatter(historico_risco, historico_retorno, c='blue', label='Evolução Portfólio')
plt.xlabel('Risco (Desvio Padrão)')
plt.ylabel('Retorno Esperado')
plt.title('Evolução Risco x Retorno da Melhor Carteira')
plt.legend()
plt.grid(True)
plt.show()

print("\n")
# === Resultado Final ===
def show_result(individuo, titulo="Resultado"):
    weights = get_weights(individuo)
    retorno_esperado = calcular_retorno(weights)
    risco_real = calcular_risco(weights)
    diversificacao = calcular_diversificacao(weights)
    diversificacao_norm = diversificacao / np.log(len(weights))
    fitness_final = fitness(individuo)

    df_resultado = pd.DataFrame({
        "Ticker": tickers,
        "Peso (%)": weights * 100
    }).sort_values(by="Peso (%)", ascending=False)

    print(f"\n📊 {titulo}")
    print(df_resultado.to_string(index=False, float_format="%.2f"))
    print("\n🔍 Métricas da Carteira:")
    print(f"Retorno esperado     : {retorno_esperado:.2%}")
    print(f"Risco esperado       : {risco_real:.2%}")
    print(f"Diversificação (norm): {diversificacao_norm:.4f}")
    print(f"Fitness final        : {fitness_final:.6f}")

print("======================================")
show_result(best_result_first_generation, "Best result of first generation")
print("======================================")
best = max(pop, key=fitness)
show_result(best, f"Melhor Carteira Encontrada of {NUM_GENERATION_COUNTER} generation")