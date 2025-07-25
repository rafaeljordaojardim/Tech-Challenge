import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Algoritmo Genético - Ajuste de Pesos", layout="centered")

st.title("Algoritmo Genético - Ajuste de Pesos")

ALPHA = st.slider("Peso do retorno", 0.0, 1.0, 0.6, 0.05)
BETA = st.slider("Peso do risco", 0.0, 1.0, 0.2, 0.05)
GAMMA = st.slider("Peso da diversificação", 0.0, 1.0, 0.2, 0.05)

soma_pesos = round(ALPHA + BETA + GAMMA, 2)

st.write(f"Soma atual dos pesos: {soma_pesos}")

if soma_pesos != 1.0:
    st.warning("⚠️ A soma dos pesos deve ser igual a 1.0")
else:
    st.success("✅ Pesos válidos!")

    # Simulação de dados (exemplo)
    np.random.seed(42)
    resultados = []
    for i in range(100):
        retorno = np.random.uniform(5, 15)
        risco = np.random.uniform(2, 10)
        equilibrio = np.random.uniform(0.7, 1.0)
        score = retorno * ALPHA - risco * BETA + equilibrio * GAMMA
        resultados.append((retorno, risco, equilibrio, score))

    df = pd.DataFrame(resultados, columns=["Retorno", "Risco", "Equilibrio", "Score"])
    df = df.sort_values(by="Score", ascending=False)

    st.subheader("Top 10 Resultados")
    st.dataframe(df.head(10))

    # Gráfico
    st.subheader("Distribuição dos Scores")
    fig, ax = plt.subplots()
    ax.plot(df["Score"].values, label="Score", color="blue")
    ax.set_title("Scores Calculados")
    ax.set_xlabel("Indivíduo")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
