import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifetimes import BetaGeoFitter, GammaGammaFitter

# Load data
summary = pd.read_csv('summary.csv', index_col='customer_id')

# Fit BG/NBD model
bgf = BetaGeoFitter(penalizer_coef=0.01)
bgf.fit(summary['frequency'], summary['recency'], summary['T'])

# Fit Gamma-Gamma model
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(summary['frequency'], summary['monetary_value'])

# Tambahkan kolom prediksi jika belum ada
if 'predicted_purchases' not in summary.columns:
    summary['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(
        30, summary['frequency'], summary['recency'], summary['T'])

if 'expected_avg_value' not in summary.columns:
    summary['expected_avg_value'] = ggf.conditional_expected_average_profit(
        summary['frequency'], summary['monetary_value'])

# Sidebar
st.sidebar.title("\U0001F4CA Segmentasi Pelanggan")
tab = st.sidebar.radio("Pilih Jenis Segmentasi:", ["Manual (RFM/CLTV)", "KMeans", "Perbandingan", "Model Lifetime"])

st.title("\U0001F4BC Dashboard Segmentasi Pelanggan")

if tab == "Manual (RFM/CLTV)":
    st.subheader("Segmentasi Berdasarkan RFM/CLTV")

    option = st.selectbox("Pilih Segmentasi:", ['segment', 'cltv_segment'])

    count_data = summary[option].value_counts().reset_index()
    count_data.columns = [option, 'count']

    fig, ax = plt.subplots()
    sns.barplot(data=count_data, x=option, y='count', palette='Set2', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.dataframe(summary[[option, 'frequency', 'monetary_value', 'CLTV']].sort_values(by='CLTV', ascending=False))

elif tab == "KMeans":
    st.subheader("Segmentasi Berdasarkan KMeans")

    features = ['frequency', 'monetary_value', 'predicted_purchases']

    fig = sns.pairplot(summary, hue='kmeans_cluster', vars=features, palette='Set2')
    st.pyplot(fig)

    st.dataframe(summary[['kmeans_cluster', 'frequency', 'monetary_value', 'CLTV', 'prob_alive']].sort_values(by='CLTV', ascending=False))

    # ⬇️ Tambahan analisis dan visualisasi prob_alive & frequency
    st.markdown("### 📈 Rata-rata Probability Alive & Frekuensi per Cluster")

    cluster_stats = summary.groupby('kmeans_cluster').agg({
        'prob_alive': 'mean',
        'frequency': 'mean'
    }).reset_index()

    cluster_stats.columns = ['Cluster', 'Rata-rata Probabilitas Alive', 'Rata-rata Frekuensi']
    st.dataframe(cluster_stats)

    fig1, ax1 = plt.subplots()
    sns.barplot(data=cluster_stats, x='Cluster', y='Rata-rata Probabilitas Alive', palette='Set2', ax=ax1)
    plt.title("Rata-rata Probability Alive per Cluster")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    sns.barplot(data=cluster_stats, x='Cluster', y='Rata-rata Frekuensi', palette='Set3', ax=ax2)
    plt.title("Rata-rata Frekuensi Pembelian per Cluster")
    st.pyplot(fig2)

elif tab == "Perbandingan":
    st.subheader("Perbandingan Segmentasi Manual vs KMeans")

    compare_df = summary[['segment', 'cltv_segment', 'kmeans_cluster']]
    st.dataframe(compare_df)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(data=summary, x='segment', hue='kmeans_cluster', palette='Set2', ax=ax)
    plt.title("Distribusi KMeans dalam Segmentasi Manual")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.markdown("**Interpretasi:** Bandingkan bagaimana KMeans mengelompokkan pelanggan dalam kategori 'segment' tradisional.")

elif tab == "Model Lifetime":
    st.subheader("Prediksi Lifetime Customer")

    st.markdown("**Model BG/NBD dan Gamma-Gamma**")
    st.write("Prediksi jumlah pembelian dan nilai rata-rata per pelanggan selama 30 hari ke depan:")

    st.dataframe(summary[['frequency', 'recency', 'T', 'monetary_value',
                          'predicted_purchases', 'expected_avg_value']].sort_values(by='predicted_purchases', ascending=False))

    fig, ax = plt.subplots()
    sns.scatterplot(data=summary, x='predicted_purchases', y='expected_avg_value', hue='segment', palette='Set2', ax=ax)
    plt.title("Prediksi Pembelian vs Nilai Rata-rata")
    st.pyplot(fig)
