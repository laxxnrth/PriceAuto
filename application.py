import streamlit as st
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lib_predicteur import meilleures_voitures


@st.cache_data
def load_data():
    return pl.read_json("annonces.json")


@st.cache_resource
def load_meilleures_voitures(boite: str):
    best_voitures = meilleures_voitures("annonces.json", boite)
    return best_voitures


st.set_page_config(page_title="PriceAuto")
st.title("PriceAuto ✔️​")

df = load_data()

couleur = sns.color_palette("Blues_d")[1]

def Accueil():
    st.markdown(
        """
        Le marché des voitures d'occasion est extrêmement vaste et diversifié, avec des prix influencés par un grand nombre de facteurs, tels que le kilométrage, l'année de fabrication,
        le modèle ou même le type de boîte. Les prix des voitures d'occasion sont souvent difficiles à estimer de manière exacte, car ils dépendent de caractérisitiques internes (comme l'état 
        de la voiture) et externes (comme la demande locale).

        Dans ce contexte, il peut être difficile pour un acheteur ou un vendeur de savoir si le prix demandé pour une voiture est **correct** ou si la voiture est 
        **sous-évaluée** ou **sur-évaluée** par rapport à sa valeur réelle sur le marché.

        ### Objectif 🎯​

        L'objectif principal de ce projet est d'utiliser des modèles de Machine Learning pour prédire le prix d'une voiture en fonction de plusieurs caractéristiques mesurables,
        puis de comparer ce **prix prédit** avec le **prix réel** sur le marché.

        L'idée est d'identifier les voitures pour lesquelles le prix réel est significativement plus bas que le prix prédit, ce qui peut indiquer que ces voitures sont sous-évaluées  et 
        représentent une bonne affaire pour un acheteur potentiel.

        ### Spécificité de la boîte de vitesse ⚙️

        La boîte **manuelle** et la boîte **automatique** peuvent être perçues différemment par les consommateurs. \n
        Dans de nombreux marchés, les voitures à boîte automatique sont généralement considérées comme plus confortables et modernes, ce qui peut entraîner
        des prix plus plus élevés, en particulier sur des modèles récents. Les voitures à boîte manuelle, quant à elles, peuvent être moins populaires dans certaines régions
        et peuvent être source d'une demande plus faible dans d'autres marchés. Cela peut donc influencer le prix de manière différente.

        Afin de prendre en compte l'hétérogeneité des tranches de prix pour les voitures à boîte manuelle et celles des voitures à boîte automatique, nous vous permettons
        de prédire les prix des voitures selon ces deux types de boîte de vitesse.

        ### Présentation de l'application 📱

        Cette application *Streamlit* vous permet dans un premier temps de visualiser les données récoltées suite au Web Scraping dans l'onglet "*Données des voitures*".
        Vous pouvez jeter un coup d'oeil à l'intégralité des données à notre disposition, mais nous vous offrons également la possibilité de
        visualiser les graphiques de distribution des voitures selon différentes catégories. \n 
        
        L'onglet "*Flitrer les voitures*" vous permet d'effectuer des recherches selon la tranche de prix qui vous intéresse, vous offrant ainsi la possibilité d'aller 
        sur la page d'annonce de vente de la voiture (*Référence*) qui correspond le mieux à votre budget.

        L'onglet "*Prédiction du prix*" renvoie les 5 voitures dont le prix réel est minimisé par rapport au prix prédit selon le type de boîte de vitesse, qui 
        constituent donc les principales voitures sous-évaluées sur le marché des voitures d'occasion. 

        ##### ... 🚗 **Nous vous laissons découvrir cette application en toute sérénité !**

        """
    )


def Donnees():
    st.subheader("Données des voitures 📈")
    st.markdown(
        """ 
        Les données des voitures ont été récoltées grâce au *Web Scraping*, une technique d'extraction automatique des données de sites web.
        Nous nous sommes basés sur le site [**Autosphère**](https://www.autosphere.fr/), premier distributeur d'automobiles en France, 
        mettant en vente plus de 15 000 voitures neuves et d'occasion.

        Pour chaque voiture, nous avons les caractéristiques suivantes : 
        * Le **nom**,
        * La **marque**,
        * Le **modèle**,
        * La **puissance** réelle en chevaux,
        * L'**énergie**,
        * L'**année** de fabrication,
        * Le **kilomètrage**,
        * Le type de **boîte** de vitesse,
        * Le **prix** en euros,
        * La **mensualité** minimale proposée pour payer en plusieurs fois,
        * La **localisation** de vente grâce au code postal,
        * La **proximité** ou non avec la région d'Île-de-France.

        Nous vous laissons jeter un coup d'oeil aux données grâce au tableau intéractif ci-dessous. ⬇️​

        """
    )

    df_sans_ref = df.drop("Référence")
    st.write(df_sans_ref)

    st.markdown(
        """ 
        Afin de présenter au mieux les données, voici les distributions des voitures selon 5 catégories :

        """
    )

    option = st.radio(
        "",
        ("Marque", "Boite", "Energie", "Prix", "Kilométrage"),
    )

    if option == "Marque":

        marques_count = df.group_by("Marque").agg(pl.len().alias("Nombre de voitures"))
        marques_count_sorted = marques_count.sort("Nombre de voitures", descending=True)

        top_10_marques = marques_count_sorted.head(10)
        top_10_marques_pandas = top_10_marques.to_pandas()

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(
            x="Marque",
            y="Nombre de voitures",
            data=top_10_marques_pandas,
            ax=ax,
            palette="Blues_d",
        )
        for bar in ax.patches:
            bar.set_edgecolor("white")

        ax.set_ylabel(" ")
        ax.set_xlabel(" ")
        ax.tick_params(axis="x")
        ax.tick_params(axis="y")
        ax.set_title(
            "Top 10 des marques de voitures les plus présentes en nombre de voitures"
        )
        ax.tick_params(axis="y")

        st.pyplot(fig)

    elif option == "Boite":

        boite_count = df.group_by("Boite").agg(pl.len().alias("Nombre de voitures"))
        boite_count_sorted = boite_count.sort("Nombre de voitures", descending=True)

        type_boite = boite_count_sorted.head(10)
        type_boite_pandas = type_boite.to_pandas()

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(
            x="Boite",
            y="Nombre de voitures",
            data=type_boite_pandas,
            ax=ax,
            palette="Blues_d",
            width=0.3,
        )
        for bar in ax.patches:
            bar.set_edgecolor("white")

        ax.set_ylabel(" ")
        ax.set_xlabel(" ")
        ax.tick_params(axis="x")
        ax.tick_params(axis="y")
        ax.set_title(
            "Distribution des voitures selon le type de boîte de vitesse en nombre de voitures"
        )
        ax.tick_params(axis="y")

        st.pyplot(fig)

    elif option == "Energie":

        energie_count = df.group_by("Energie").agg(pl.len().alias("Nombre de voitures"))

        energie_count_sorted = energie_count.sort("Nombre de voitures", descending=True)

        energie_count_sorted_pd = energie_count_sorted.to_pandas()

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(
            x="Energie",
            y="Nombre de voitures",
            data=energie_count_sorted_pd,
            ax=ax,
            palette="Blues_d",
        )

        for bar in ax.patches:
            bar.set_edgecolor("white")

        ax.set_xlabel(" ")
        ax.set_ylabel(" ")
        ax.set_title("Distribution des Energies en nombre de voitures")
        ax.tick_params(axis="x")
        ax.tick_params(axis="y")

        st.pyplot(fig)

    if option == "Prix":
        fig, ax = plt.subplots(figsize=(12, 8))
        prix_df = df.select("Prix")
        prix_pandas = prix_df.to_pandas()

        sns.histplot(prix_pandas, ax=ax, color=couleur, bins=100)
        ax.set_xlim(0, 125000)
        ax.legend().set_visible(False)

        ax.set_xlabel("Prix (€)")
        ax.set_ylabel(" ")
        ax.tick_params(axis="x")
        ax.tick_params(axis="y")
        ax.set_title(
            "Distribution des Prix des Voitures en nombre de voitures"
        )

        for bar in ax.patches:
            bar.set_edgecolor("white")
            bar.set_facecolor(couleur)

        st.pyplot(fig)

    if option == "Kilométrage":
        fig, ax = plt.subplots(figsize=(12, 8))
        kilometre_df = df.select("Kilomètre")
        kilometre_pandas = kilometre_df.to_pandas()

        sns.histplot(kilometre_pandas, ax=ax, color=couleur, bins=75)
        ax.set_xlim(0, 125000)
        ax.legend().set_visible(False)

        ax.set_xlabel("Kilométrage (km)")
        ax.set_ylabel(" ")
        ax.tick_params(axis="x")
        ax.tick_params(axis="y")

        for bar in ax.patches:
            bar.set_edgecolor("white")
            bar.set_facecolor(couleur)

        ax.set_title(
            "Distribution des Kilométrages des Voitures en nombre de voitures"
        )

        st.pyplot(fig)

    st.markdown(
        """ 
        *Nous tenons à rappeler que ces données sont utilisées à des fins pédagogiques dans le cadre d'un projet universitaire. 
        Toute utilisation à des fins commerciales est strictement interdite.*
        """
    )

    st.sidebar.markdown("Données des voitures")


def Filtrer():
    st.subheader("Filtrer les voitures 🔍​")

    st.write(
        f""" 

        Nous savons à quel point l'achat d'une voiture peut être une décision importante et parfois difficile, 
        surtout lorsqu'on prend en compte la diversité des modèles, des marques, et des critères comme l'âge, le kilométrage, 
        ou encore le type de boîte de vitesse. Ajoutez à cela le facteur prix, et cela peut rapidement devenir un vrai casse-tête.

        C'est pourquoi cette page a été spécialement conçue pour vous permettre de **filtrer les voitures selon une tranche de prix adaptée à votre budget**. 
        Grâce à cet outil, vous pourrez explorer les options qui vous correspondent le mieux et éviter de perdre du temps avec des voitures qui dépassent votre budget.

        **Les informations clés sur les prix des voitures disponibles :**

        - Le prix **minimal** des voitures présentes dans notre base de données est de *{df["Prix"].min()} €*.
        - Le prix **maximum** s'élève à *{df["Prix"].max()} €*, offrant ainsi une large gamme de véhicules, du plus abordable au plus premium.
        - Le prix **moyen** des voitures disponibles est de *{round(df['Prix'].mean())} €*, ce qui vous donne une bonne idée de la gamme de prix générale.
        - Enfin, le prix **médian**, c'est-à-dire celui qui sépare la moitié des voitures moins chères de l'autre moitié, est de *{round(df['Prix'].median())} €*. 
        Cela peut être un bon indicateur du prix central, loin des extrêmes.

        Grâce à ces données, vous pourrez ajuster vos attentes en fonction du budget que vous souhaitez investir à votre achat et facilement 
        trouver une voiture qui correspond à vos critères financiers.

        Maintenant, vous pouvez utiliser le filtre de prix ci-dessous pour affiner votre recherche en fonction de votre budget et 
        découvrir les voitures qui vous conviennent le mieux !

        """
    )

    min_prix, max_prix = df["Prix"].min(), df["Prix"].max()
    prix = st.slider(
        "Choisissez une plage de prix",
        min_value=min_prix,
        max_value=max_prix,
        value=(min_prix, max_prix),
        step=10,
    )

    filtered_data = df.filter((df["Prix"] >= prix[0]) & (df["Prix"] <= prix[1]))

    columns = [col for col in filtered_data.columns if col != "Référence"] + [
        "Référence"
    ]
    filtered_data = filtered_data.select(columns)
    filtered_data_pandas = filtered_data.to_pandas()

    st.dataframe(
        filtered_data_pandas.style.hide(axis="index"),
        use_container_width=True,
    )

    st.sidebar.markdown("Filtrer les voitures")


def Predictions():
    st.subheader("Prédictions de prix 💸​")

    st.markdown(
        """
        Sur cette page, nous mettons à votre disposition une sélection de 5 voitures dont le prix réel est **minimisé par rapport au prix prédit** 
        par notre modèle. Cela vous permet de découvrir les voitures qui, selon notre algorithme, offrent un bon rapport qualité-prix en termes de prix réel et estimé.

        **Les variables clés utilisées pour l'entraînement de nos modèles de prédiction sont les suivantes :**
    
        - **Mensualité minimale** proposée pour un paiement en plusieurs fois, permettant de visualiser l'importance du prix total de la voiture.
        - **Kilométrage** de la voiture, un facteur essentiel pour estimer l'usure et la valeur restante du véhicule.
        - **Puissance** en chevaux, qui impacte non seulement la performance de la voiture mais aussi son prix de marché.
        - **Année de mise en circulation**, influençant la dépréciation du véhicule et sa valeur estimée.
        - **Appartenance à la région Île-de-France** ou non, un critère qui peut jouer sur la valeur des voitures en fonction de la demande et de l'offre locale.

        Ces variables sont combinées dans nos **modèles de Machine Learning** pour effectuer des prédictions aussi précises que possible sur le prix des voitures. 
        Nous avons utilisé plusieurs techniques pour entraîner nos modèles, et voici les principaux :

        - **Forêt Aléatoire (Random Forest)** : un modèle puissant qui apprend à partir de multiples arbres décisionnels pour fournir des prédictions robustes.
        - **K-Nearest Neighbors (KNN)** : un modèle qui se base sur la similarité des voitures pour prédire les prix en fonction des voisins les plus proches.
        - **Régression Linéaire** : un modèle plus simple, mais efficace, qui cherche à établir une relation linéaire entre les variables indépendantes et le prix.
        - **Support Vector Machines (SVM)** : un modèle qui maximise la marge entre les différentes classes de données pour améliorer la précision des prédictions.

        Le **choix du meilleur modèle de prédiction** repose sur deux critères fondamentaux : la *performance sur les données d'entraînement* et 
        l'*absence de surapprentissage* (ou overfitting). Nous avons ainsi sélectionné le modèle offrant la meilleure généralisation, c'est-à-dire 
        celui qui prédit le mieux sur des données non vues, sans être trop spécifique aux données d'entraînement.

        **Découvrez dès maintenant les résultats en choisissant le type de boîte de vitesse !**
        """
    )

    boite = st.selectbox("Choisissez le type de boîte", ["Manuelle", "Automatique"])

    if st.button("Afficher les meilleures voitures"):

        liste_voitures = load_meilleures_voitures(boite)

        indices_voitures = liste_voitures[1]

        data = pd.read_json("annonces.json")
        voitures_affichage = []

        for i in indices_voitures:
            voiture = data.loc[
                i,
                [
                    "Nom",
                    "Prix",
                    "Mensualité",
                    "Puissance",
                    "Energie",
                    "Kilomètre",
                    "Année",
                    "Localisation",
                    "Référence",
                ],
            ]
            voitures_affichage.append(voiture)

        df_voitures = pd.DataFrame(voitures_affichage)

        st.write(df_voitures.drop_duplicates())

    st.sidebar.markdown("Prédiction des prix")


page_names_to_funcs = {
    "Accueil": Accueil,
    "Données des voitures": Donnees,
    "Filtrer les voitures": Filtrer,
    "Prédiction du prix": Predictions,
}

selected_page = st.sidebar.selectbox("Choisis une page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
