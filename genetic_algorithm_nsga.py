!pip install deap
import copy
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# === Dados ===
dados = pd.read_csv("fiis_dados.csv", index_col=0)
tickers = dados.index.tolist()
retornos = dados["Retorno_Medio_Anual"].values
cov_matrix = pd.read_csv("fiis_cov_matrix.csv", index_col=0)
cov_matrix = cov_matrix.loc[tickers, tickers].values

def criar_individuo_diversificado():
    ind = np.random.dirichlet(np.ones(len(tickers)) * 2.0)
    return creator.Individuo(ind.tolist())

# === Funções de Avaliação ===
def calcular_retorno(weights):
    weights = weights / weights.sum()
    return np.dot(weights, retornos)

def calcular_risco(weights):
    weights = weights / weights.sum()
    return np.sqrt(weights @ cov_matrix @ weights.T)

def calcular_diversificacao(weights):
    weights = weights / weights.sum()
    weights = np.where(weights == 0, 1e-6, weights)
    entropia = -np.sum(weights * np.log(weights))
    return entropia / np.log(len(weights))

def avaliar(ind):
    w = np.array(ind)
    w = np.clip(w, 0, 1)
    w /= np.sum(w)
    retorno = calcular_retorno(w)
    risco = calcular_risco(w)
    diversificacao = calcular_diversificacao(w)
    return -retorno, risco, -diversificacao  # Negativos porque DEAP minimiza

def get_weights(item):
    weights = np.array(item)
    weights = weights / weights.sum()
    return weights

# === Configuração DEAP ===
N = len(tickers)
POP_SIZE = 100
N_GEN = 1000

creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0, -1.0))  # Minimizar retorno, maximizar risco, minimizar diversificação
creator.create("Individuo", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_float", lambda: random.uniform(0.01, 1))
toolbox.register("individuo", criar_individuo_diversificado)
toolbox.register("populacao", tools.initRepeat, list, toolbox.individuo)

toolbox.register("evaluate", avaliar)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0.0, up=1.0, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=0.0, up=1.0, eta=20.0, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

# === Criar população inicial e avaliar ===
pop = toolbox.populacao(n=POP_SIZE)
for ind in pop:
    ind.fitness.values = toolbox.evaluate(ind)

# Gráfico: População inicial - Retorno vs Risco
retornos_pop = [-ind.fitness.values[0] for ind in pop]
riscos_pop = [ind.fitness.values[1] for ind in pop]

plt.figure(figsize=(8,6))
plt.scatter(riscos_pop, retornos_pop, c='orange', label='População Inicial')
plt.xlabel('Risco (Desvio Padrão)')
plt.ylabel('Retorno Esperado')
plt.title('População Inicial - Retorno vs Risco')
plt.legend()
plt.grid(True)
plt.show()

# === Exemplo de Mutação ===
print("\n🔧 Exemplo de Mutação em 2 indivíduos:")
for i in range(2):
    ind = criar_individuo_diversificado()
    print(f"\n🎯 Indivíduo original {i+1}:")
    print(np.round(get_weights(ind) * 100, 2))

    ind_mutado = copy.deepcopy(ind)
    toolbox.mutate(ind_mutado)
    ind_mutado[:] = get_weights(ind_mutado)

    print(f"✨ Indivíduo mutado {i+1}:")
    print(np.round(get_weights(ind_mutado) * 100, 2))

# Gráfico: Depois da mutação - Retorno vs Risco dos dois indivíduos mutados
retornos_mutados = []
riscos_mutados = []

for i in range(2):
    # Criar e mutar indivíduo
    ind = criar_individuo_diversificado()
    ind_mutado = copy.deepcopy(ind)
    toolbox.mutate(ind_mutado)
    ind_mutado[:] = get_weights(ind_mutado)

    fit_vals = toolbox.evaluate(ind_mutado)
    retornos_mutados.append(-fit_vals[0])
    riscos_mutados.append(fit_vals[1])

plt.figure(figsize=(8,6))
plt.scatter(riscos_mutados, retornos_mutados, c='green', label='Indivíduos Mutados')
plt.xlabel('Risco (Desvio Padrão)')
plt.ylabel('Retorno Esperado')
plt.title('Indivíduos Após Mutação - Retorno vs Risco')
plt.legend()
plt.grid(True)
plt.show()

# === Execução do NSGA-II ===
hof = tools.ParetoFront()

pop, logbook = algorithms.eaMuPlusLambda(
    pop, toolbox, mu=POP_SIZE, lambda_=POP_SIZE,
    cxpb=0.7, mutpb=0.3, ngen=N_GEN,
    halloffame=hof, verbose=True
)

# Gráfico Fronteira de Pareto final
retornos_hof = [-ind.fitness.values[0] for ind in hof]
riscos_hof = [ind.fitness.values[1] for ind in hof]

plt.figure(figsize=(8, 6))
plt.scatter(riscos_hof, retornos_hof, c='blue', label='Fronteira de Pareto')
plt.xlabel('Risco (Desvio Padrão)')
plt.ylabel('Retorno Esperado')
plt.title('Fronteira de Pareto - Portfólios Ótimos')
plt.legend()
plt.grid(True)
plt.show()

# Função para converter a população em uma tabela
def popula_para_tabela(populacao):
    registros = []
    for individuo in populacao:
        weights = get_weights(individuo)
        fit_retorno, fit_risco, fit_divers = individuo.fitness.values

        registro = {
            "Retorno (%)": -fit_retorno * 100,
            "Risco (%)": fit_risco * 100,
            "Diversificação (%)": -fit_divers * 100,
        }

        for i, ticker in enumerate(tickers):
            registro[f"Peso_{ticker} (%)"] = weights[i] * 100

        registros.append(registro)

    return pd.DataFrame(registros)

# Exibir resultados de forma mais detalhada
def show_result(ind, titulo="Resultado"):
    weights = np.array(ind)
    weights /= weights.sum()

    retorno = calcular_retorno(weights) * 100
    risco = calcular_risco(weights) * 100
    diversificacao = calcular_diversificacao(weights) * 100

    df = pd.DataFrame({
        "Ticker": tickers,
        "Peso (%)": weights * 100  # Pesos em %
    }).sort_values("Peso (%)", ascending=False)

    print(f"\n📊 {titulo}")
    print(df[df["Peso (%)"] > 1.0].to_string(index=False, float_format="%.2f"))

    print("\n🔍 Métricas:")
    print(f"Retorno esperado     : {retorno:.2f}%")
    print(f"Risco esperado       : {risco:.2f}%")
    print(f"Diversificação (norm): {diversificacao:.2f}%")

# Gerar a tabela com todas as carteiras
df_carteiras = popula_para_tabela(pop)

# Exibir as melhores carteiras ordenadas por Retorno
print("\n📊 Melhores carteiras por Retorno:")
print(df_carteiras.sort_values(by="Retorno (%)", ascending=False).to_string(float_format="%.2f"))

# Exibir a melhor carteira encontrada
best_ind = hof[0]
show_result(best_ind, "Melhor Carteira Encontrada")

df_carteiras.sort_values(by="Retorno (%)", ascending=False).to_csv("resultados_carteiras.csv", index=False, float_format="%.2f")
print("\nResultados salvos no arquivo 'resultados_carteiras.csv'.")