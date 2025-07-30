# 🧬 Otimização de Carteira de FIIs com Algoritmo Genético

Este projeto utiliza **algoritmos genéticos (AG)** para encontrar a composição ideal de uma carteira de **Fundos Imobiliários (FIIs)** listados na B3, equilibrando **retorno esperado**, **risco (volatilidade)** e **diversificação**. 

Dados históricos são coletados automaticamente via API do Yahoo Finance (`yfinance`), e os resultados são exibidos com gráficos e métricas financeiras.

---

## 📈 Objetivo

> Encontrar a melhor combinação de FIIs para maximizar o retorno esperado, minimizar o risco e garantir boa diversificação.

---

## ⚙️ Tecnologias e Bibliotecas

- Python 3.x
- `pandas`
- `numpy`
- `yfinance`
- `matplotlib`
- `seaborn`
- `math`
- `random`

---

## 🧪 Funcionalidades

- 📦 **Coleta automática** de dados históricos dos FIIs da B3 via `yfinance`.
- 📊 Cálculo de **retorno médio anual** e **risco anualizado** (desvio padrão).
- 🧠 Execução de um **algoritmo genético customizado**, com:
  - Seleção por fitness
  - Crossover
  - Mutação adaptativa
  - Elite preservation
- 🔍 Avaliação do portfólio com função fitness baseada em:
  - Retorno (α)
  - Risco (β)
  - Diversificação (γ via entropia)
- 📈 Gráficos de:
  - Evolução do fitness
  - Relação risco x retorno da melhor carteira
- 📋 Geração de CSVs para persistência dos dados (retornos e covariância)