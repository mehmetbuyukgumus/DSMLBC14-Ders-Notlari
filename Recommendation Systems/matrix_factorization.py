import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate

pd.set_option('display.max_columns', None)
pd.set_option("display.expand_frame_repr", False)

###################################
# 1. Veri Seti'nin Hazırlanması
###################################
movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
df = pd.merge(rating, movie, on='movieId', how='left')

movie_ids = [130219, 356, 4422, 541]
movies = ["The Dark Knight (2011)",
          "Cries and Whispers (Viskningar och rop) (1972)",
          "Forrest Gump (1994)",
          "Blade Runner (1982)"]

sample_df = df[df['movieId'].isin(movie_ids)]

user_movie_df = pd.pivot_table(sample_df, index="userId", columns="title", values="rating")


reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(sample_df[["userId", "movieId", "rating"]], reader)

##############################
# Adım 2: Modelleme
##############################
trainset, testset = train_test_split(data, test_size=.25)
svd_model = SVD()
svd_model.fit(trainset)
predictions = svd_model.test(testset)
accuracy.rmse(predictions)
svd_model.predict(uid=1.0, iid=541, verbose=True)
svd_model.predict(uid=1.0, iid=356, verbose=True)

sample_df[sample_df["userId"] == 1]

##############################
# Adım 3: Model Tuning
##############################
param_grid = {'n_epochs': [5, 10, 20],
              'lr_all': [0.002, 0.005, 0.007]}

gs = GridSearchCV(SVD,
                  param_grid,
                  measures=['rmse', 'mae'],
                  cv=3,
                  n_jobs=-1,
                  joblib_verbose=True)

gs.fit(data)
gs.best_score["rmse"]
gs.best_params["rmse"]

##############################
# Adım 4: Final Model ve Tahmin
##############################
dir(svd_model)
svd_model.n_epochs

svd_model = SVD(**gs.best_params['rmse'])

data = data.build_full_trainset()
svd_model.fit(data)

svd_model.predict(uid=1.0, iid=541, verbose=True)
