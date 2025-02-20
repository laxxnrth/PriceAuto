# PriceAuto ✔️

PriceAuto✔️ est une application qui permet à tout utilisateur d'obtenir les cinq voitures sous-évaluées sur le marché, et donc les cinq voitures les plus intéressantes à acheter.

L'objectif est de prédire les prix des voitures d'occasion selon plusieurs caractéristiques, tout en prenant en compte le type de boite de vitesse, afin de connaître les principales voitures dont le prix de vente est inférieur à ce qu'elles valent réellement.

## Scraping (lib_scraping.py)

Pour récolter nos données sur les voitures, on utilise la méthode de Web Scraping, une technique d'extraction automatique des données issues de sites internet. On se base sur le site de l'[Autosphère](https://www.autosphere.fr/), premier distributeur d'automobiles de France.

Plus précisément, on va s'intéresser aux voitures d'occasion :
- Scraping des données contenues dans l'onglet *Occasion* à l'aide des packages `requests` et `bs4`.
- Génération d'une liste `voitures` pour chaque élément du scraping, itéré sur 300 pages.
- Création d'une fonction `nettoyage()` en utilisant le package `polars` qui permet la mise en forme des données.
- Création d'une fonction `fichier_json()` permettant d'enregistrer le dataframe dans un fichier json, qu'on applique à notre liste `voitures`. On obtient alors notre fichier `annonces.json`.

## Machine Learning (lib_predicteur.py)

Notre objectif principal est de prédire le prix des voitures d'occasion, à l'aide du package `scikit-learn`.

- Création d'une fonction `split()` permettant de diviser nos données en deux sous-ensembles (test et entraînement) à l'aide de `train_test_split()`. 
- Entraînement de 4 modèles sur nos données d'entraînement :
    - La régression linéaire,
    - Les KNN,
    - La Random Forest,
    - La SVM.
- Création d'une fonction `meilleur_modele()` permettant de choisir le meilleur modèle de prédiction selon deux critères de performance : le meilleur score d'entraînement et l'absence de sur-apprentissage.
- Création d'une fonction `predict()` permettant de renvoyer :
    - les prix prédits grâce à `meilleur_modele()`,
    - les prix réels et la différence entre les deux,
    - l'erreur absolue moyenne.
- Création d'une fonction `meilleures_voitures()` renvoyant les cinq voitures qui maximisent la différence entre le prix prédit et prix réel (prix réel < prix prédit) en utilisant les résultats de `predict()`.

## Application (application.py)

Notre application a été créée avec `streamlit`, elle contient 4 pages consultables à l'aide du menu latéral.

**Portabilité du projet**

La gestion des dépendances s'est effectué avec `uv`, elle doit être importée avec la commande suivante : 

```powershell
py -m pip install uv
py -m uv add "packages"
```

Le code a été formatté avec `black` et cette commande peut être lancée :

```powershell
py -m pip install black
py -m black ./lib_scraping.py ./lib_predicteur.py ./application.py
```

**Lancement de l'application**

Afin d'ouvrir l'application, il suffit de lancer :

```powershell
py -m streamlit run application.py
```

L'application se construit en 4 pages :
- Sur la page **Accueil**, on retrouve une brève introduction à destination des utilisateurs leur permettant une mise en contexte concernant le marché des voitures d'occasion. Cette page leur permet aussi de connaître l'objectif principal de ce projet, ainsi qu'une explication sur la distinction entre boîte automatique et boîte manuelle. Enfin pour finir, une présentation de l'application ainsi qu'une définition du contenu des différents onglets de celle-ci leur est proposée.
- Dans l'onglet **Données des voitures 📈**, l'utilisateur retrouve les différentes caractéristiques de toutes les données scrapées grâce à un tableau intéractif. La page lui permet également de voir des simples statistiques descriptives sur certaines catégories. 
- L'onglet **Filtrer les voitures 🔍** permet à l'utilisateur de filtrer les résultats selon une tranche de prix, avec des informations sur la référence afin de rediriger l'utilisateur pour un potentiel achat. Une indication sur le prix moyen et le prix médian des voitures est aussi donnée.
- Enfin, le dernier onglet, **Prédiction de prix 💸**, affiche les cinq voitures pour lesquelles le prix réel est minimisé par rapport au prix prédit, selon le choix de boîte de vitesse fait par l'utilisateur, grâce à la fonction `meilleures_voitures()`. Nous avons ainsi les informations sur les principales voitures sous-évaluées sur le marché.
