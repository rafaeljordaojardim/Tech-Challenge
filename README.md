# 🧬 Otimização de Carteira de FIIs com Algoritmo Genético

## 📌 Sobre o Projeto

Este projeto foi desenvolvido para o Tech Challenge da pós-graduação e utiliza *Algoritmos Genéticos* para otimizar uma carteira de Fundos Imobiliários (FIIs) listados na B3.

🎯 O objetivo é encontrar a alocação ideal que *equilibre retorno esperado, risco (volatilidade) e diversificação*. A solução automatiza a coleta de dados históricos e realiza testes evolutivos em milhares de combinações, buscando a mais eficiente.

## ⚙️ Como Funciona

A aplicação se divide em duas etapas principais: coleta de dados e otimização.

### 1️⃣ Coleta e Preparo dos Dados

- Utiliza a API do yfinance para buscar dados históricos dos FIIs.
- Calcula o retorno esperado de cada fundo e a matriz de covariância entre eles.
- Essas métricas são fundamentais para avaliar risco e retorno.

### 2️⃣ Otimização com Algoritmo Genético

- O coração do projeto, onde acontece o processo evolutivo.

#### 🧠 Função de Aptidão (Fitness)

O algoritmo usa o *Índice de Sharpe* para medir a qualidade de cada carteira. Ele avalia o retorno ajustado pelo risco:

$$
S(x) = \frac{R(x) - R_f}{\sigma(x)}
$$

Onde:

- $R(x)$ = retorno esperado da carteira  
- $R_f$ = taxa livre de risco  
- $\sigma(x)$ = risco da carteira (desvio padrão)

#### 🔄 Ciclos Evolutivos

A cada geração, o algoritmo executa:

1. *Seleção* — Carteiras com melhor índice de Sharpe são escolhidas.
2. *Cruzamento* — Combina carteiras para formar novas soluções.
3. *Mutação* — Adiciona variações aleatórias para explorar melhor o espaço de soluções.

## 📈 Resultados

Ao fim da execução, o script apresenta:

- A alocação ideal por FII.
- Gráficos que mostram o desempenho ao longo das gerações.
- Métricas financeiras da carteira ótima.

## 🚀 Como Executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/rafaeljordaojardim/Tech-Challenge.git