## İş Problemei
# Scout’lar tarafından izlenen futbolcuların özelliklerine verilen puanlara göre, oyuncuların hangi sınıf
# (average, highlighted) oyuncu olduğunu tahminleme.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from warnings import filterwarnings
from Machine_Learning.Homework_and_Exercises.config import *
filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)


def load_data():
    att_df = pd.read_csv(
        "/Users/mehmetbuyukgumus/Desktop/Miuul Data Sciencetist Bootcamp/datasets/scoutium_attributes.csv", sep=";")
    pot_df = pd.read_csv(
        "/Users/mehmetbuyukgumus/Desktop/Miuul Data Sciencetist Bootcamp/datasets/scoutium_potential_labels.csv",
        sep=";")
    df = pd.merge(att_df, pot_df, on=["task_response_id", 'match_id', 'evaluator_id', "player_id"], how="inner")
    df = df[df["position_id"] != 1]
    df = df[df["potential_label"] != "below_average"]
    df = pd.pivot_table(df, index=["player_id", "position_id", "potential_label"],
                        columns="attribute_id", values="attribute_value", aggfunc="mean")
    df.reset_index(inplace=True)
    df.columns = df.columns.map(str)
    return df


def pre_preccessing(dataframe, column):
    encoder = LabelEncoder()
    dataframe[column] = encoder.fit_transform(dataframe[column])
    num_cols = dataframe.select_dtypes("float")
    scaler = StandardScaler()
    for col in num_cols:
        dataframe[col] = scaler.fit_transform(dataframe[[col]])
    return dataframe


def base_models(X, y):
    print("Base Models....")
    scoring = ["roc_auc", "f1", "precision", "recall", "accuracy"]
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier(verbose=-1)),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]
    for scrore in scoring:
        for name, classifier in classifiers:
            cv_results = cross_validate(classifier, X, y, cv=3, scoring=scrore, verbose=0)
            print(f"{scrore}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")


## En iyi sonuç veren 4 model RF, LGB, GBM, XGBosst


def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in opt_classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results["test_score"].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results["test_score"].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models


def voting_classifier(best_models, X, y):
    print("Voting Classifier...")

    voting_clf = VotingClassifier(estimators=[('RF', best_models["RF"]),
                                              ('GBM', best_models["GBM"]),
                                              ('XGBoost', best_models["XGBoost"]),
                                              ('LightGBM', best_models["LightGBM"])],
                                  voting='soft').fit(X, y)

    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf


def main():
    df = load_data()
    df = pre_preccessing(df, "potential_label")
    y = df["potential_label"]
    X = df.drop("potential_label", axis=1)
    base_models(X, y)
    best_models = hyperparameter_optimization(X, y)
    voting_clf = voting_classifier(best_models, X, y)
    return voting_clf


if __name__ == "__main__":
    opt_classifiers = [("RF", RandomForestClassifier(), rf_params),
                       ('LightGBM', LGBMClassifier(verbose=-1), lightgbm_params),
                       ('GBM', GradientBoostingClassifier(), gbm_params),
                       ('XGBoost', XGBClassifier(), xgboost_params)]
    main()

Son Görev
df = load_data()
df = pre_preccessing(df, "potential_label")
y = df["potential_label"]
X = df.drop("potential_label", axis=1)
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title(f'Features {model}')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

knn_model = KNeighborsClassifier()
knn_model.fit(X, y)
xgbmodel = XGBClassifier()
xgbmodel.fit(X,y)
rf_model = RandomForestClassifier()
rf_model.fit(X,y)
importances = [xgbmodel, rf_model]

for imp in importances:
    plot_importance(imp, X)


