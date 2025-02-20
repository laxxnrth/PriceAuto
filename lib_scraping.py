from requests import get
import polars as pl

import random
import time
import re
from bs4 import BeautifulSoup


def extract_puissance(text):
    powers = re.findall(r"(\d{1,3})(?:\s*(?:CH|CV|ch|cv|hp)?)", text)
    powers = [int(power) for power in powers]
    powers = [power for power in powers if 2 <= power <= 999]

    if powers:
        return max(powers)
    return None


def extraire_boite(text: str):
    return re.findall(r"(manuelle|automatique)", text, re.IGNORECASE)


def extraire_utilitaire(text: str) -> bool:
    return bool(re.search(r"UTILITAIRE", text, re.IGNORECASE))


motif_boite = r"(Manuelle|Automatique)"

voitures = []

for i in range(1, 300):
    url = f"https://www.autosphere.fr/recherche?market=VO&page={i}&ordre=proximite-asc&critaire_checked[]=year&critaire_checked[]=discount&critaire_checked[]=emission_co2"
    response = get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "lxml")

        time.sleep(random.uniform(1, 5))

        voiture_list = soup.find_all("div", class_="bloc_infos_veh_parent")

        for voiture in voiture_list:

            try:
                ref = voiture.find("div", class_="fiche_hover")

                if ref:
                    link = ref.find("a")
                    base_ref = link.get("href")
                    utilitaire = extraire_utilitaire(ref.get_text(strip=True))
                    marque = ref.find("span", class_="marque").get_text(strip=True)
                    modele = ref.find("span", class_="modele").get_text(strip=True)
                else:
                    ref = None
                    base_ref = None
                    utilitaire = None
                    marque = None
                    modele = None

            except AttributeError:
                ref = None
                base_ref = None
                utilitaire = None
                marque = None
                modele = None

            try:
                elements = voiture.find("span", class_="serie ellipsis").get_text(
                    strip=True
                )
                puissance = extract_puissance(elements)

            except AttributeError:
                elements = None
                puissance = None

            try:
                caract = voiture.find("div", class_="caract").get_text(strip=True)

                if caract:
                    elements = [e.strip() for e in caract.split("/")]

                    energie = elements[0] if len(elements) > 0 else None
                    kilometre = elements[1] if len(elements) > 1 else None
                    annee = elements[2] if len(elements) > 2 else None
                    boite = elements[3] if len(elements) > 3 else None
                else:
                    energie = None
                    kilometre = None
                    annee = None
                    boite = None

            except AttributeError:
                caract = None
                energie = None
                annee = None
                kilometre = None
                boite = None

            try:
                budget = voiture.find("div", class_="prix_wrapper")

                if budget:
                    prix = budget.find("span", class_="bloc_prix").get_text(strip=True)
                    mensualite = budget.find(
                        "span", class_="mensualite_montant"
                    ).get_text(strip=True)
                else:
                    prix = None
                    mensualite = None

            except AttributeError:
                prix = None
                mensualite = None

            try:
                footer = voiture.find("div", class_="span12 thumbnail_footer")

                if footer:
                    localisation = footer.find(
                        "span", class_="localisation regular"
                    ).get_text(strip=True)
                else:
                    localisation = None

            except AttributeError:
                localisation = None

            voitures.append(
                {
                    "Référence": base_ref,
                    "Nom": (marque or "") + " " + (modele or ""),
                    "Marque": marque,
                    "Modèle": modele,
                    "Puissance": puissance,
                    "Energie": energie,
                    "Année": annee,
                    "Kilomètre": kilometre,
                    "Boite": boite,
                    "Prix": prix,
                    "Mensualité": mensualite,
                    "Localisation": localisation,
                    "Utilitaire": utilitaire,
                }
            )


def nettoyage(liste: list) -> pl.DataFrame:
    """Fonction qui permet de nettoyer les données collectées à la suite
    du scraping : conversion de type, ajout d'une colonne, supression des doublons,
    des valeurs nulles et des colonnes non pertinentes.
    """
    df = pl.DataFrame(liste)

    df = df.select(
        pl.col("Référence"),
        pl.col("Nom"),
        pl.col("Marque"),
        pl.col("Modèle"),
        pl.col("Puissance"),
        pl.col("Energie"),
        pl.col("Utilitaire"),
        pl.col("Année").cast(pl.Int64).alias("Année"),
        pl.col("Kilomètre")
        .str.replace(" km", "")
        .str.replace(" ", "")
        .cast(pl.Int64)
        .alias("Kilomètre"),
        pl.col("Boite").str.extract(motif_boite).alias("Boite"),
        pl.col("Prix")
        .str.replace("\xa0€", "")
        .str.replace(" ", "")
        .cast(pl.Int64)
        .alias("Prix"),
        pl.col("Mensualité")
        .str.replace(" €", "")
        .str.replace(" ", "")
        .cast(pl.Int64)
        .alias("Mensualité"),
        pl.col("Localisation").alias("Localisation"),
    )

    df = df.filter(
        ~(pl.col("Référence") == "")
        & ~(pl.col("Nom") == "")
        & ~(pl.col("Modèle") == "")
        & ~(pl.col("Marque") == "")
        & ~(pl.col("Année").is_null())
        & ~(pl.col("Puissance").is_null())
        & ~(pl.col("Kilomètre").is_null())
        & ~(pl.col("Boite") == "")
        & ~(pl.col("Prix").is_null())
        & ~(pl.col("Mensualité").is_null())
        & ~(pl.col("Energie") == "")
        & ~(pl.col("Localisation") == "")
        & ~(pl.col("Référence").is_duplicated())
        & ~(pl.col("Utilitaire") == True)
    )

    df = df.with_columns(
        (
            pl.col("Localisation").str.starts_with("78")
            | pl.col("Localisation").str.starts_with("95")
            | pl.col("Localisation").str.starts_with("77")
            | pl.col("Localisation").str.starts_with("91")
            | pl.col("Localisation").str.starts_with("94")
            | pl.col("Localisation").str.starts_with("75")
            | pl.col("Localisation").str.starts_with("92")
            | pl.col("Localisation").str.starts_with("93")
        ).alias("IDF")
    )

    df = df.drop(["Utilitaire"])

    return df


def fichier_json(liste: list):
    """Fonction qui permet de convertir le DataFrame en un
    fichier json, afin de faciliter sa manipulation par la suite.
    """
    df = nettoyage(liste)
    df.write_json("annonces.json")


fichier_json(voitures)
