#######################
# IMDB Uygulaması
#######################

import pandas as pd
import math
import scipy.stats as st
import warnings
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

data = pd.read_csv("datasets/movies_metadata.csv", low_memory=False)
df = data.copy()
df = df[["title", "vote_average", "vote_count"]]

#######################
# Vote Average'a Göre Sıralama
#######################

df.sort_values(by="vote_average", ascending=False).head(20)

df["vote_count"].describe().T
df["vote_average"].describe([0.10, 0.25, 0.50, 0.70, 0.80, 0.90, 0.95]).T

df.loc[df["vote_count"] > 400].sort_values(by="vote_average", ascending=False)
df["vote_count_score"] = MinMaxScaler(feature_range=(1, 10)).fit(df[["vote_count"]]).transform(df[["vote_count"]])
df["average_count_score"] = df["vote_average"] * df["vote_count_score"]
df.sort_values(by="average_count_score", ascending=False).head(20)

#######################
# IMDB Weighted Rating
#######################

# weighted_rating = (v/(v+M) * r) + (M/(v+M) * C)

# r = vote average
# v = vote count
# M = minimum votes required to be listed in the Top 250
# C = the mean vote across the whole report (currently 7.0)

# Film 1:
# r = 8
# M = 500
# v = 1000

# (1000 / (1000+500))*8 = 5.33


# Film 2:
# r = 8
# M = 500
# v = 3000

# (3000 / (3000+500))*8 = 6.85

# (1000 / (1000+500))*9.5

# Film 1:
# r = 8
# M = 500
# v = 1000

# Birinci bölüm:
# (1000 / (1000+500))*8 = 5.33

# İkinci bölüm:
# 500/(1000+500) * 7 = 2.33

# Toplam = 5.33 + 2.33 = 7.66


# Film 2:
# r = 8
# M = 500
# v = 3000

# Birinci bölüm:
# (3000 / (3000+500))*8 = 6.85

# İkinci bölüm:
# 500/(3000+500) * 7 = 1

# Toplam = 7.85

M = 2500
C = df["vote_average"].mean()


def weighted_rating(r, v, M, C):
    return (v / (v + M) * r) + (M / (v + M) * C)


df.sort_values(by="average_count_score", ascending=False).head(20)

weighted_rating(7.4, 11444, M, C)
weighted_rating(8.1, 14075, M, C)
weighted_rating(8.5, 8358, M, C)

df["weighted_rating"] = weighted_rating(df["vote_average"], df["vote_count"], M, C)
df.sort_values(by="weighted_rating", ascending=False).head(10)


#######################
# Bayesian Average Rating Score
#######################

# 12481                                    The Dark Knight
# 314                             The Shawshank Redemption
# 2843                                          Fight Club
# 15480                                          Inception
# 292                                         Pulp Fiction

def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score


bayesian_average_rating([34733, 4355, 4704, 6561, 13515, 26183, 87368, 273082, 600260, 1295351])

bayesian_average_rating([37128, 5879, 6268, 8419, 16603, 30016, 78538, 199430, 402518, 837905])

df = pd.read_csv("datasets/imdb_ratings.csv")
df = df.iloc[0:, 1:]

df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["one", "two", "three", "four", "five",
                                                                "six", "seven", "eight", "nine", "ten"]]), axis=1)
df.sort_values(by="bar_score", ascending=False).head(20)
