import polars as pl
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor


def split(fichier: str, boite: str):
    """
    Fonction qui permet de faire le découpages des données test et d'entraînement,
    selon une proportion de 20% pour les données test et 80% pour les données d'entraînement.

    """
    df = pl.read_json(fichier)

    data_df = df.filter((pl.col("Boite") == boite))
    cible_df = data_df.select("Prix")
    data_df = (
        data_df.select(pl.exclude("Référence"))
        .select(pl.exclude("Nom"))
        .select(pl.exclude("Modèle"))
        .select(pl.exclude("Marque"))
        .select(pl.exclude("Boite"))
        .select(pl.exclude("Energie"))
        .select(pl.exclude("Prix"))
        .select(pl.exclude("Localisation"))
    )
    data_df = data_df.with_columns([(1 / pl.col("Kilomètre")).alias("Kilomètre")])
    data = np.array(data_df)
    predict = np.array(cible_df)
    X = data
    y = predict
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=54, shuffle=True
    )
    return X, y, X_tr, X_te, y_tr, y_te


def meilleur_modele(fichier: str, boite: str) -> list:
    """Fonction qui permet de choisir le meilleur modèle de prédiction
    parmi les KNN, la RandomForest, la SVM et la Régression linéaire.
    Le choix du meilleur modèle repose sur la séléction du meilleur
    score d'entraînement et sur la non-présence de sur-apprentissage.

    Exemple:
    >>> meilleur_modele("annonces.json", boite = "Manuelle")
    [RandomForestRegressor(n_estimators=128)]
    MAE moyenne : 12670.198689956333
    """

    X, y, X_tr, X_te, y_tr, y_te = split(fichier, boite)

    meilleur_estimateur = []

    # KNeighborsRegressor
    knr = KNeighborsRegressor()
    knr_gs = GridSearchCV(
        knr,
        {
            "n_neighbors": range(1, 10),
            "weights": ("uniform", "distance"),
        },
        cv=KFold(5),
    )
    knr_gs.fit(X_tr, y_tr.ravel())
    meilleur_estimateur.append(knr_gs.best_estimator_)

    # RandomForestRegressor
    rfr = RandomForestRegressor()
    rfr_gs = GridSearchCV(
        rfr,
        {
            "n_estimators": (8, 16, 32, 64, 128, 256),
        },
        cv=KFold(5),
    )
    rfr_gs.fit(X_tr, y_tr.ravel())
    meilleur_estimateur.append(rfr_gs.best_estimator_)

    # SVR
    svr = SVR()
    svr_pip = Pipeline(
        [
            ("mise_echelle", MinMaxScaler()),
            ("standardisation", StandardScaler()),
            ("support_vecteurs", svr),
        ]
    )
    svr_gs = GridSearchCV(
        svr_pip,
        {
            "support_vecteurs__C": [0.1, 1.0, 10, 100, 1000],
            "support_vecteurs__epsilon": (0.1, 1.0, 10, 100, 1000),
        },
        cv=KFold(5),
    )
    svr_gs.fit(X_tr, y_tr.ravel())
    meilleur_estimateur.append(svr_gs.best_estimator_)

    # LinearRegression
    lr = LinearRegression()
    lr.fit(X_tr, y_tr.ravel())
    meilleur_estimateur.append(lr)

    score_train = []
    score_test = []
    for i in meilleur_estimateur:
        i.fit(X_tr, y_tr.ravel())
        score_train.append(i.score(X_tr, y_tr.ravel()))
        score_test.append(i.score(X_te, y_te.ravel()))

    df_estimateur = pd.DataFrame(
        {
            "estimateur": meilleur_estimateur,
            "score train": score_train,
            "score test": score_test,
        }
    )

    df_estimateur["sur_apprentissage"] = (
        abs(df_estimateur["score train"] - df_estimateur["score test"]) > 0.3
    )

    meilleur_modele_candidates = df_estimateur[~df_estimateur["sur_apprentissage"]]

    if meilleur_modele_candidates.empty:
        print("Attention, présence de surapprentissage")
        meilleur_modele = df_estimateur.loc[
            df_estimateur["score test"].idxmax(), "estimateur"
        ]
    else:
        meilleur_modele = meilleur_modele_candidates.loc[
            meilleur_modele_candidates["score test"].idxmax(), "estimateur"
        ]

    return [meilleur_modele]


def predict(fichier: str, boite: str) -> list:
    """Fonction qui permet de prédire le prix des voitures grâce
    à la fonction meilleur_modele() selon le type de boîte choisie,
    avec une information sur l'erreur absolue moyenne.

    Exemple:
    >>> predict("annonces.json", boite = "Manuelle")

    MAE moyenne : 5786.731404445548
    [                               Nom       y  y_pred  y_pred - y
     0                      PEUGEOT 208   17399   11787       -5612
     1                      PEUGEOT 208   16899   11786       -5113
     2     LAND-ROVER Range rover sport  149990   17952     -132038
     3             MINI Cooper 3 portes   34590   17579      -17011
     4                  RENAULT Austral   44199   17952       26247
     ...                            ...     ...     ...         ...
     7781           CITROEN C3 aircross   10499   11874        1375
     7782               RENAULT Austral   26999   13783      -13216
     7783               MINI Countryman   19390   12704       -6686
     7784               MINI Countryman   17999   12690       -5309
     7785                  PEUGEOT 3008   24499   13471      -11028

     [7786 rows x 4 columns]]

    """

    liste = meilleur_modele(fichier, boite)
    modele = liste[0]
    df = pl.read_json(fichier)

    data_df = df.select(["Kilomètre", "Année", "Puissance", "Mensualité", "IDF"])

    data_df = data_df.with_columns([(1 / pl.col("Kilomètre")).alias("Kilomètre")])

    target_df = df.select("Prix")

    X = data_df.to_numpy()
    y = target_df.to_numpy()

    y_pred = []
    nom = df["Nom"].to_list()

    for i in range(0, len(data_df)):
        y_pred.append(modele.predict(X[[i]]))

    liste_y_pred = [int(pred[0]) for pred in y_pred]
    liste_y = [int(val[0]) for val in y]

    ecart = [pred - real for pred, real in zip(liste_y_pred, liste_y)]

    mae_per_prediction = [abs(pred - real) for pred, real in zip(liste_y_pred, liste_y)]
    mae_moyenne = sum(mae_per_prediction) / len(mae_per_prediction)

    df_pred = pd.DataFrame(
        {"Nom": nom, "y": liste_y, "y_pred": liste_y_pred, "y_pred - y": ecart}
    )

    print(f"MAE moyenne : {mae_moyenne}")

    return [df_pred]


def meilleures_voitures(fichier: str, boite: str) -> list:
    """Fonction qui permet de choisir les 5 meilleures voitures pour lesquelles
    le prix réel est minimisé par rapport au prix prédit.

    Exemple :
    >>> choix_voitures("annonces.json", boite = "Manuelle")
    [['MERCEDES Amg gt', 'RENAULT Zoe', 'RENAULT Zoe', 'FORD Fiesta', 'RENAULT Captur'], [7508, 6949, 7424, 7344, 6031]]

    """
    liste = predict(fichier, boite)
    df = liste[0]
    nom = []
    indices = []

    df_sorted = df.sort_values(by="y_pred - y", ascending=False)

    top_5 = df_sorted.head(5)

    nom = top_5["Nom"].tolist()
    indices = top_5.index.tolist()

    return [nom, indices]
