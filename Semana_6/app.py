"""App de segmentación de clientes — Lab Marketing Analytics.

Ejecutar local:  streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Segmentación de clientes", page_icon="📊", layout="wide")

ATRIBUTOS = ["ingreso_anual", "gasto_total", "compras_tienda", "compras_web", "compras_catalogo",
             "visitas_web_mes", "recencia_dias", "hijos_pequenos", "edad"]
SEMILLA = 42
RUTA_DATOS = Path(__file__).parent / "data" / "clientes_retail.csv"


# ---------- Datos ----------
@st.cache_data
def leer_csv(fuente) -> pd.DataFrame:
    df = pd.read_csv(fuente)
    if "respondio_ultima_campana" in df and df["respondio_ultima_campana"].dtype == object:
        df["respondio_ultima_campana"] = (df["respondio_ultima_campana"].astype(str).str.strip().str.lower()
                                          .isin(["1", "si", "sí", "yes", "true"]).astype(int))
    return df


@st.cache_data
def estandarizar(df: pd.DataFrame) -> np.ndarray:
    return StandardScaler().fit_transform(df[ATRIBUTOS])


@st.cache_data
def curva_silueta(Xz: np.ndarray) -> pd.DataFrame:
    filas = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit(Xz)
        filas.append({"k": k, "silueta": silhouette_score(Xz, km.labels_), "inercia": km.inertia_})
    return pd.DataFrame(filas)


def ordenar_por_gasto(df, etiquetas, prefijo):
    """Renombra grupos para que 1 = menor gasto promedio. El ruido de DBSCAN (-1) queda como 'Ruido'."""
    s = pd.Series(etiquetas, index=df.index)
    validos = s[s != -1]
    orden = df.loc[validos.index].groupby(validos)["gasto_total"].mean().sort_values().index
    mapa = {o: f"{prefijo}{i + 1}" for i, o in enumerate(orden)}
    mapa[-1] = "Ruido"
    return s.map(mapa)


st.title("📊 Segmentación de clientes")
st.caption("¿A qué grupos de clientes les hablamos distinto, y cuánto vale cada grupo?")

subido = st.sidebar.file_uploader("Cargar otro clientes_retail.csv (opcional)", type="csv")
if subido is not None:
    clientes = leer_csv(subido)
elif RUTA_DATOS.exists():
    clientes = leer_csv(RUTA_DATOS)
else:
    st.warning("No encontré data/clientes_retail.csv. Súbalo desde la barra lateral.")
    st.stop()

faltan = [c for c in ATRIBUTOS + ["respondio_ultima_campana"] if c not in clientes.columns]
if faltan:
    st.error(f"Faltan columnas en el archivo: {faltan}")
    st.stop()

Xz = estandarizar(clientes)

# ---------- Controles ----------
st.sidebar.header("Técnica")
tecnica = st.sidebar.radio("Algoritmo", ["k-means", "Jerárquico (Ward)", "DBSCAN"])
if tecnica == "DBSCAN":
    eps = st.sidebar.slider("eps (radio del vecindario)", 0.5, 4.0, 1.5, 0.1)
    min_s = st.sidebar.slider("Mínimo de vecinos", 3, 30, 5)
    etiquetas = DBSCAN(eps=eps, min_samples=min_s).fit_predict(Xz)
    prefijo = "D"
else:
    k = st.sidebar.slider("Número de grupos (k)", 2, 8, 3)
    if tecnica == "k-means":
        etiquetas = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit_predict(Xz)
        prefijo = "C"
    else:
        etiquetas = fcluster(linkage(Xz, method="ward"), t=k, criterion="maxclust")
        prefijo = "H"

df = clientes.copy()
df["Cluster"] = ordenar_por_gasto(df, etiquetas, prefijo)
grupos = sorted(g for g in df["Cluster"].unique() if g != "Ruido")

# ---------- KPIs ----------
mask = etiquetas != -1
sil = silhouette_score(Xz[mask], etiquetas[mask]) if len(set(etiquetas[mask])) > 1 else np.nan
c1, c2, c3, c4 = st.columns(4)
c1.metric("Clientes", f"{len(df):,}")
c2.metric("Grupos", len(grupos))
c3.metric("Silueta", "—" if np.isnan(sil) else f"{sil:.3f}",
          help="De −1 a 1. >0.50 fuerte · 0.25–0.50 moderada · <0.25 débil")
c4.metric("Ruido", f"{(~mask).mean():.1%}" if tecnica == "DBSCAN" else "0%")

tab_valor, tab_perfil, tab_mapa, tab_k, tab_datos = st.tabs(
    ["💰 Valor por segmento", "🧭 Perfil", "🗺️ Mapa", "🔢 Elegir k", "📄 Datos"])

with tab_valor:
    valor = df.groupby("Cluster").agg(
        clientes=("gasto_total", "size"),
        gasto_promedio=("gasto_total", "mean"),
        gasto_total=("gasto_total", "sum"),
        tasa_respuesta=("respondio_ultima_campana", "mean"),
    )
    valor["% clientes"] = valor["clientes"] / valor["clientes"].sum()
    valor["% gasto"] = valor["gasto_total"] / valor["gasto_total"].sum()
    valor["índice de valor"] = valor["% gasto"] / valor["% clientes"]
    valor["valor esp. por contacto*"] = valor["tasa_respuesta"] * valor["gasto_promedio"]
    st.dataframe(valor.style.format({
        "gasto_promedio": "{:,.0f}", "gasto_total": "{:,.0f}", "tasa_respuesta": "{:.1%}",
        "% clientes": "{:.1%}", "% gasto": "{:.1%}", "índice de valor": "{:.2f}",
        "valor esp. por contacto*": "{:,.0f}"}), width="stretch")
    st.caption("*Proxy = tasa de respuesta × gasto promedio. Supuesto: el gasto histórico aproxima el valor "
               "de un respondedor; no es margen ni CLV.")
    largo = valor[["% clientes", "% gasto"]].reset_index().melt("Cluster", var_name="medida", value_name="pct")
    st.plotly_chart(px.bar(largo, x="Cluster", y="pct", color="medida", barmode="group",
                           labels={"pct": "%"}, title="% de clientes vs % del gasto")
                    .update_yaxes(tickformat=".0%"), width="stretch")

with tab_perfil:
    indice = (df[df["Cluster"] != "Ruido"].groupby("Cluster")[ATRIBUTOS].mean()
              / df[ATRIBUTOS].mean() * 100).round(0)
    st.markdown("**Índice vs. promedio general** (100 = promedio). Úselo para ponerle nombre a cada grupo.")
    st.plotly_chart(px.imshow(indice.T, text_auto=".0f", color_continuous_scale="RdYlGn",
                              color_continuous_midpoint=100, aspect="auto"), width="stretch")
    st.markdown("**Promedios reales**")
    st.dataframe(df.groupby("Cluster")[ATRIBUTOS].mean().T.style.format("{:,.1f}"), width="stretch")

with tab_mapa:
    a, b = st.columns(2)
    eje_x = a.selectbox("Eje X", ATRIBUTOS, index=0)
    eje_y = b.selectbox("Eje Y", ATRIBUTOS, index=1)
    st.plotly_chart(px.scatter(df.sort_values("Cluster"), x=eje_x, y=eje_y, color="Cluster", opacity=.65,
                               hover_data=["edad", "hijos_pequenos", "recencia_dias"]), width="stretch")

with tab_k:
    curva = curva_silueta(Xz)
    a, b = st.columns(2)
    a.plotly_chart(px.line(curva, x="k", y="silueta", markers=True, title="Silueta por k"), width="stretch")
    b.plotly_chart(px.line(curva, x="k", y="inercia", markers=True, title="Inercia (codo)"), width="stretch")
    st.caption("Calculado con k-means sobre las 9 variables estandarizadas.")

with tab_datos:
    st.dataframe(df, width="stretch", height=400)
    st.download_button("⬇️ Descargar clientes_con_cluster.csv", df.to_csv(index=False).encode("utf-8"),
                       "clientes_con_cluster.csv", "text/csv")
