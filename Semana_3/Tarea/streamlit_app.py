"""
Laboratorio Final — Starbucks Rewards
App para que el equipo de negocio explore a quién enviar el cupón de $5:
curva Qini del T-learner y tabla de políticas por % de la base contactada.

Correr localmente:  streamlit run streamlit_app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Starbucks Rewards — ¿A quién enviar el cupón?", page_icon="☕", layout="wide")

GREEN = "#00704A"
ORANGE = "#C4512F"
GRAY = "#9A988E"
FEATURES = ["recency_days", "frequency", "monetary", "email_open_rate", "tenure_days"]

# Ruta relativa a ESTE archivo: funciona igual en local y en Streamlit Cloud,
# sin importar desde qué carpeta se ejecute la app.
DATA_PATH = Path(__file__).parent / "data" / "uplift_campaign.csv"

MODELOS = {
    "Regresión logística": lambda: LogisticRegression(max_iter=1000),
    "Árbol (max_depth=5, leaf=200)": lambda: DecisionTreeClassifier(max_depth=5, min_samples_leaf=200, random_state=42),
    "Random Forest (d=6, leaf=100)": lambda: RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=100, n_jobs=-1, random_state=42),
    "Gradient Boosting (d=3)": lambda: GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, min_samples_leaf=100, random_state=42),
}


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def entrenar_t_learner(df: pd.DataFrame, nombre_modelo: str, test_size: float, seed: int):
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["treatment_group"])

    # Modelo de riesgo/churn (para comparar en la curva Qini)
    churn = 1 - train["responded_60d"]
    modelo_riesgo = LogisticRegression(max_iter=1000).fit(train[FEATURES], churn)

    # T-learner: un modelo por grupo
    train_t = train[train["treatment_group"] == "treatment"]
    train_c = train[train["treatment_group"] == "control"]
    crear = MODELOS[nombre_modelo]
    modelo_tratado = crear().fit(train_t[FEATURES], train_t["responded_60d"])
    modelo_control = crear().fit(train_c[FEATURES], train_c["responded_60d"])

    test = test.copy()
    X_test = test[FEATURES]
    test["prob_churn"] = modelo_riesgo.predict_proba(X_test)[:, 1]
    test["p_tratado"] = modelo_tratado.predict_proba(X_test)[:, 1]
    test["p_control"] = modelo_control.predict_proba(X_test)[:, 1]
    test["uplift_estimado"] = test["p_tratado"] - test["p_control"]
    return train, test


def valor_de_la_politica(test_df, seleccionados):
    sel = test_df.loc[seleccionados]
    valor_tratado = sel.loc[sel["treatment_group"] == "treatment", "utilidad_neta_60d"].mean()
    valor_control = sel.loc[sel["treatment_group"] == "control", "margin_60d"].mean()
    return valor_tratado - valor_control, len(sel)


def curva_qini(test_df, score_col, steps=40):
    orden = test_df.sort_values(score_col, ascending=False).reset_index(drop=True)
    es_tratado = (orden["treatment_group"] == "treatment").values
    respondio = orden["responded_60d"].values
    acum_t = np.cumsum(np.where(es_tratado, respondio, 0))
    acum_c = np.cumsum(np.where(~es_tratado, respondio, 0))
    n_t = np.cumsum(es_tratado)
    n_c = np.cumsum(~es_tratado)
    n = len(orden)
    xs, ys = [0.0], [0.0]
    for i in np.linspace(1, n, steps).astype(int):
        tasa_t = acum_t[i - 1] / n_t[i - 1] if n_t[i - 1] > 0 else 0
        tasa_c = acum_c[i - 1] / n_c[i - 1] if n_c[i - 1] > 0 else 0
        xs.append(i / n)
        ys.append((tasa_t - tasa_c) * n)
    return np.array(xs), np.array(ys)


def coeficiente_qini(xs, ys, n):
    area_modelo = np.trapezoid(ys, xs)
    area_azar = np.trapezoid(np.linspace(0, ys[-1], len(xs)), xs)
    return (area_modelo - area_azar) / n


@st.cache_data
def tabla_de_politicas(test_df, score_col, paso):
    orden = test_df.sort_values(score_col, ascending=False)
    filas = []
    for pct in range(paso, 101, paso):
        n_c = int(len(test_df) * pct / 100)
        v, n = valor_de_la_politica(test_df, orden.head(n_c).index)
        filas.append({"% contactado": pct, "Valor por cliente ($)": round(v, 2),
                      "Clientes contactados": n, "Valor total ($)": round(v * n, 2)})
    return pd.DataFrame(filas)


# --- Título y carga de datos ---
st.title("☕ Starbucks Rewards — ¿A quién enviar el cupón de $5?")
st.caption("Esta app ayuda a decidir qué porcentaje de clientes debe recibir el cupón para maximizar "
           "la utilidad neta, contactando solo a quienes el cupón realmente hace comprar.")

df = load_data()

with st.sidebar:
    st.header("Parámetros")
    nombre_modelo = st.selectbox("Modelo dentro del T-learner", list(MODELOS.keys()), index=2)
    pct_contactar = st.slider("% de la base a contactar", min_value=5, max_value=100, value=20, step=5)
    paso_tabla = st.radio("Granularidad de la tabla", [5, 10], index=1, format_func=lambda x: f"cada {x}%")
    escalar = st.checkbox("Mostrar valores escalados a la base completa (40,000)", value=True)

with st.spinner("Entrenando modelos..."):
    train, test = entrenar_t_learner(df, nombre_modelo, test_size=0.30, seed=42)

factor = len(df) / len(test) if escalar else 1.0
base_txt = "base completa" if escalar else "base de prueba"

# --- KPIs de la política elegida ---
n_sel = int(len(test) * pct_contactar / 100)
idx_sel = test.sort_values("uplift_estimado", ascending=False).head(n_sel).index
v_cli, n = valor_de_la_politica(test, idx_sel)
v_todos, n_todos = valor_de_la_politica(test, test.index)
idx_riesgo = test.sort_values("prob_churn", ascending=False).head(n_sel).index
v_riesgo, n_r = valor_de_la_politica(test, idx_riesgo)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Clientes a contactar", f"{n * factor:,.0f}")
c2.metric("Valor por cliente", f"${v_cli:+.2f}")
c3.metric(f"Valor total ({base_txt})", f"${v_cli * n * factor:+,.0f}",
          delta=f"${(v_cli * n - v_riesgo * n_r) * factor:+,.0f} vs. targeting por riesgo")
c4.metric("Si se contacta a todos", f"${v_todos * n_todos * factor:+,.0f}")

col_izq, col_der = st.columns([3, 2])

# --- Curva Qini ---
with col_izq:
    st.subheader("Curva Qini")
    xs_u, ys_u = curva_qini(test, "uplift_estimado")
    xs_r, ys_r = curva_qini(test, "prob_churn")
    q_u = coeficiente_qini(xs_u, ys_u, len(test))
    q_r = coeficiente_qini(xs_r, ys_r, len(test))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs_u, ys_u, color=GREEN, linewidth=3, label=f"T-learner · {nombre_modelo} (Qini={q_u:+.3f})")
    ax.plot(xs_r, ys_r, color=ORANGE, linewidth=3, label=f"Riesgo/churn (Qini={q_r:+.3f})")
    ax.plot([0, 1], [0, ys_u[-1]], color=GRAY, linestyle="--", label="Al azar")
    ax.axvline(pct_contactar / 100, color="black", linestyle=":", linewidth=1.5, label=f"Política elegida ({pct_contactar}%)")
    ax.set_xlabel("% de la base contactada")
    ax.set_ylabel("Respuestas incrementales acumuladas")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    st.caption("Cuanto más se separa la curva verde de la diagonal, mejor ordena el modelo a los clientes "
               "que el cupón convence. El modelo de riesgo apenas supera al azar.")

# --- Tabla de políticas ---
with col_der:
    st.subheader("Tabla de políticas")
    tabla = tabla_de_politicas(test, "uplift_estimado", paso_tabla).copy()
    tabla["Valor total ($)"] = (tabla["Valor total ($)"] * factor).round(0)
    tabla["Clientes contactados"] = (tabla["Clientes contactados"] * factor).round(0).astype(int)
    optimo = tabla.loc[tabla["Valor total ($)"].idxmax(), "% contactado"]

    def resaltar(fila):
        if fila["% contactado"] == optimo:
            return ["background-color: #d7ede3; font-weight: bold"] * len(fila)
        if fila["% contactado"] == pct_contactar:
            return ["background-color: #f3e2dc"] * len(fila)
        return [""] * len(fila)

    st.dataframe(tabla.style.apply(resaltar, axis=1).format({"Valor por cliente ($)": "{:+.2f}",
                                                             "Valor total ($)": "{:+,.0f}",
                                                             "Clientes contactados": "{:,}"}),
                 hide_index=True, width="stretch")
    st.success(f"Óptimo en esta tabla: contactar al **{optimo}%** de la base.")

    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.plot(tabla["% contactado"], tabla["Valor total ($)"], marker="o", color=GREEN)
    ax2.axhline(0, color=GRAY, linestyle="--")
    ax2.axvline(pct_contactar, color="black", linestyle=":")
    ax2.set_xlabel("% contactado")
    ax2.set_ylabel(f"Valor total ($, {base_txt})")
    ax2.grid(alpha=0.2)
    st.pyplot(fig2)

with st.expander("¿Cómo se calcula el valor?"):
    st.markdown(
        "- **Uplift estimado** = P(compra con cupón) − P(compra sin cupón), con dos modelos entrenados por separado (T-learner).\n"
        "- **Valor de una política**: dentro de los clientes seleccionados, utilidad neta de los que recibieron cupón "
        "(margen − $5 si compró − $0.05 de envío) menos el margen de los que no lo recibieron.\n"
        "- Todo se mide sobre el 30% de clientes que los modelos **no vieron** al entrenar."
    )
