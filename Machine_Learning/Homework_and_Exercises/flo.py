################ Gözetimsiz Öğrenme ile Müşteri Segmentasyonu ################
# FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
# Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranışlardaki öbeklenmelere göre
# gruplar oluşturulacak.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import dendrogram, linkage

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


def load_dataset():
    dataframe = pd.read_csv("/Users/mehmetbuyukgumus/Desktop/Miuul Data Sciencetist Bootcamp/datasets/flo_data_20k.csv")
    return dataframe


def new_features(dataframe):
    date_cols = [col for col in dataframe.columns if "date" in col]
    for col in date_cols:
        dataframe[col] = pd.to_datetime(dataframe[col])

    analysis_date = dataframe["last_order_date"].max()
    analysis_date = analysis_date + pd.to_timedelta(5, "D")
    dataframe["tenure"] = analysis_date - dataframe["first_order_date"]
    dataframe["recancy"] = analysis_date - dataframe["last_order_date"]
    dataframe["tenure"] = dataframe["tenure"].dt.days
    dataframe["recancy"] = dataframe["recancy"].dt.days
    dataframe["monetary"] = dataframe["customer_value_total_ever_offline"] + dataframe[
        "customer_value_total_ever_online"]
    return dataframe


def pred_data_to_model(dataframe):
    dataframe = dataframe[["tenure", "recancy", "monetary"]]
    return dataframe


def standart_standardize(dataframe):
    scaler = StandardScaler()
    dataframe = scaler.fit_transform(dataframe)
    return dataframe


def min_max_standardize(dataframe):
    scaler = MinMaxScaler((0,1))
    dataframe = scaler.fit_transform(dataframe)
    return dataframe


def modeling_wtih_kmeans(standardized_dataframe, show=False):
    kmeans = KMeans()
    ssd = []
    K = range(1,30)
    for k in K:
        kmeans = KMeans(n_clusters=k).fit(standardized_dataframe)
        ssd.append(kmeans.inertia_)

    elbow = KElbowVisualizer(kmeans, k=range(1,30))
    elbow.fit(standardized_dataframe)
    if show:
        elbow.show()
    kmeans_model = KMeans(n_clusters=elbow.elbow_value_).fit(standardized_dataframe)
    return kmeans_model.labels_


def modeling_with_hierarchical_kmeans(standardized_dataframe, show=False):
    hc_average = linkage(standardized_dataframe, "average")

    plt.figure(figsize=(7, 5))
    plt.title("Hiyerarşik Kümeleme Dendogramı")
    plt.xlabel("Gözlem Birimleri")
    plt.ylabel("Uzaklıklar")
    dendrogram(hc_average,
               truncate_mode="lastp",
               p=20,
               show_contracted=True,
               leaf_font_size=10)
    if show:
        plt.show()

    plt.figure(figsize=(7, 5))
    plt.title("Dendrograms")
    dend = dendrogram(hc_average)
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.axhline(y=0.6, color='b', linestyle='--')
    if show:
        plt.show()
    cluster = AgglomerativeClustering(n_clusters=5, linkage="average").fit(standardized_dataframe)
    predictions_ = cluster.fit_predict(standardized_dataframe)
    return predictions_


def reporting(dataframe, cluster="h"):
    if cluster == "h":
        print("########## Hiyerarşik Kümeleme İstatistikleri ##########")
        print(dataframe.groupby("clusters_h").agg({"tenure": "mean",
                                                   "recancy": "mean",
                                                   "monetary": "mean"}))
    else:
        print("########## K-Means Kümeleme İstatistikleri ##########")
        print(dataframe.groupby("clusters_k").agg({"tenure": "mean",
                                                   "recancy": "mean",
                                                   "monetary": "mean"}))


def main(cluster="k"):
    df = load_dataset()
    df = new_features(df)
    new_df = pred_data_to_model(df)
    # standardized_df = standart_standardize(new_df)
    standardized_df = min_max_standardize(new_df)
    predict_results = modeling_wtih_kmeans(standardized_df)
    predict_results_h = modeling_with_hierarchical_kmeans(standardized_df)
    df["clusters_k"] = predict_results
    df["clusters_h"] = predict_results_h
    reporting(df, cluster=cluster)
    print("""
    ################### Kümelenmiş Veri Seti İlk Beş Gözlem ###################
    """)
    print(df.head().iloc[::,-5:])

if __name__ == "__main__":
    print("""
    Hiyerarşik kümeleme sonuçlarını görmek için main fonksiyonunun içini "h" değerini argüman olarak eklemeyi
    unutmayın. Aksi halde yalnızca K-Means kümeleme sonuçlarına ulaşabilirsiniz.
    .
    .
    İşlem birkaç dakika sürebilir...
    """)
    main("h")

