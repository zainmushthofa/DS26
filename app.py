import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Konfigurasi Halaman
st.set_page_config(page_title="Wine Clustering", layout="wide")

FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol",
]


@st.cache_data
def load_data():
    df = pd.read_csv("winequality-red.csv")
    df = df.drop_duplicates().reset_index(drop=True)
    return df


df = load_data()

# Halaman Utama
st.title("🍷 Clustering Profil Wine")
st.write(
    """
    Aplikasi ini mengelompokkan sampel red wine ke dalam beberapa **profil**
    berdasarkan kandungan fisikokimianya menggunakan **K-Means Clustering**.

    **Tujuan project:** menemukan kelompok wine dengan karakteristik kimiawi
    yang mirip, dan melihat apakah kelompok tersebut berkaitan dengan skor
    kualitas wine.

    **Dataset:** Wine Quality Dataset (Red Wine) — UCI Machine Learning
    Repository (Cortez et al., 2009).
    Sumber: https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009
    """
)

st.divider()

# Data
st.header("📊 Data")

col1, col2 = st.columns(2)
with col1:
    st.write("Contoh data:")
    st.dataframe(df.head())
with col2:
    st.write("Informasi dataset:")
    st.write(f"- Jumlah baris: **{df.shape[0]}**")
    st.write(f"- Jumlah kolom: **{df.shape[1]}**")
    st.write(f"- Jumlah fitur numerik yang digunakan clustering: **{len(FEATURES)}**")
    st.write("- Missing value:", int(df.isna().sum().sum()))

st.divider()

# Clustering
st.header("🔗 Clustering")

n_clusters = st.slider("Jumlah Cluster", min_value=2, max_value=6, value=2)

X = df[FEATURES]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

st.write("Jumlah data pada setiap cluster:")
st.dataframe(df["cluster"].value_counts().rename("jumlah_data"))

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df["cluster"], palette="Set1", ax=ax, alpha=0.7)
ax.set_title("Visualisasi Cluster (PCA 2D)")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
st.pyplot(fig)

st.divider()

# Evaluation
st.header("✅ Evaluation")

sil_score = silhouette_score(X_scaled, df["cluster"])
st.metric("Silhouette Score", f"{sil_score:.4f}")

st.caption(
    "Silhouette Score berkisar antara -1 hingga 1. Semakin mendekati 1, "
    "semakin baik pemisahan antar cluster."
)

st.divider()

# Cluster Profile
st.header("🧬 Cluster Profile")

cluster_profile = df.groupby("cluster")[FEATURES + ["quality"]].mean().round(3)
st.write("Rata-rata fitur untuk setiap cluster:")
st.dataframe(cluster_profile)
