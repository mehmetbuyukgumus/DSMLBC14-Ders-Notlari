# Görev 1: Veri Setinin Hazırlanması

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

# Adım 1 : Movie,Rating Okutulması

movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')

# Adım 2 : Rating veri setine İd'lere ait film isimlerini ve türlerini movie veri setinden ekleyelim
df = movie.merge(rating, how="left", on="movieId")

# Adım 3: Toplam oy kullanılma sayısın 1000'ın altında olanları tutun ve veri setinden çıkarın
comment_counts = pd.DataFrame(df["title"].value_counts())
rare_movies = comment_counts[comment_counts["count"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]
common_movies["title"].value_counts()

# Adım 4: Index'te UserId Sutunlarda film isimler, ve değer olarak rating olacak şekilde pivot table
user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")


# Adım 5: Fonksiyonlaştırma
def create_user_movie_df():
    import pandas as pd
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 500)
    pd.set_option('display.expand_frame_repr', False)
    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["count"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df


user_movie_df = create_user_movie_df()

# GÖREV 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
# Adım 1: Rastgele bir kullanıcı id'si
random_user = int(pd.Series(user_movie_df.index).sample(1).values)

# Adım 2: Seçilen kullanıcıya ait gözlem birimlerinden oluşan random_user_dfadında yeni bir dataframe oluşturunuz
random_user_df = user_movie_df[user_movie_df.index == random_user]

# Adım 3: Seçilen kullanıcıların oy kullandığı filmleri movies_watched adında bir listeye atayınız
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

# Görev 3:  Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişilmesi
# Adım 1: Seçilen kullanıcının izlediği fimlere ait sutunları user_movie_df'ten seçiniz ve
# movies_watched_df adında yeni bir dataframe oluşturunuz
movies_watched_df = user_movie_df[movies_watched]

# Adım2: Random_user'in İzlediği filmleri diğer kullanıcılar kaçını izlediği bulunuz
user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]

# Adım 3: aynı filmlerden %60'na oy vermiş olanları users_same_movies adında listede getir
perc = len(movies_watched) * 60 / 100
users_same_movie = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

# GÖREV 4: Öneri Yapılacak Kullanıcı ile Benzer Kullanıcılar
# Adım 1 : Movies_watched_df
final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movie)],
                      random_user_df[movies_watched]])

# Adım 2 : Kolerasyonlar ile corr_df
corr_df = final_df.T.corr().stack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df)
corr_df.index.names = ["user_id_1", "user_id_2"]
corr_df.columns = ["corr"]
corr_df = corr_df.reset_index()

# Adım 3: random_user göre diğer ıdlerle korelasyonu 0.65'ten yüksekleri top_users
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)
# Rating ve top_users merge et
corr_df[corr_df["user_id_1"] == random_user].sort_values(by="corr", ascending=False).head()

# Adım 4 : Rating merge etmek
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how="inner")

# Gorev 5: Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
# Adım 1 : Corr ve Rating çarpımınn weighted_rating adıyla yeni bir değişken
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

# Adım 2:

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()

recommendation_df[recommendation_df["weighted_rating"] > 3.5]

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating",
                                                                                                   ascending=False)

movies_to_be_recommend.merge(movie[["movieId", "title"]])
movies_to_be_recommend.head()
