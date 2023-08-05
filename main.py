import os
import re
import os.path
import seaborn as sns
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DataProcessor:
    def __init__(self):
        self.df = None

    def load_data(self, file_path, sheet_name, year):
        print(f"Traitement de la feuille '{sheet_name}' pour l'année {year}")

        # Charger le DataFrame en ignorant les deux premières lignes
        base1PN = pd.read_excel(file_path, sheet_name=sheet_name)
        base1PN = base1PN.drop([0, 1]).reset_index(drop=True)
        base1PN_b = base1PN.iloc[:, 1:].reset_index(drop=True) # Supprimer la première colonne

        # Transposer les données
        base1PN_transposed = base1PN_b.transpose()

        # Définir la deuxième ligne comme en-tête
        base1PN_transposed.columns = base1PN_transposed.iloc[0]
        df = base1PN_transposed[1:]
        df['DepartementPrincipal'] = df.index.str.split('.').str[0]

        # Grouper les lignes par départements principaux et effectuer la somme
        df = df.groupby('DepartementPrincipal').sum()

        # Ajout colonne année
        df = df.assign(Année=year)

        # Créer une nouvelle colonne 'Departement' avec les valeurs de l'index
        df['Departement'] = df.index

        # reset index
        df = df.reset_index()

        # Convertir la colonne 'Département' en facteur
        df['Departement'] = df['Departement'].astype('category')

        # Supprimer les colonnes contenant le mot "Index"
        df = df.drop(df.filter(like='Index', axis=1), axis=1)

        self.df = df

    def get_dataframe(self):
        return self.df

if __name__ == "__main__":
    # Création d'une liste des années à analyser de 2012 à 2021
    years = list(range(2012, 2021))

    # Création d'une liste pour stocker les DataFrames résultants
    dataframes = []

    for year in years:
        # Création d'une instance de DataProcessor pour chaque année
        processor = DataProcessor()

        # Remplacez le chemin du fichier Excel par votre chemin réel
        file_path = "data/Base de données 1 crimes et delits enregistres depuis-2012.xlsx"

        # Ajuster le nom de la feuille pour chaque année en supprimant l'espace à la fin
        sheet_name = f'Services PN {year}'

        processor.load_data(file_path, sheet_name=sheet_name, year=year)

        # Récupérer le DataFrame final dans une variable et l'ajouter à la liste
        df = processor.get_dataframe()
        dataframes.append(df)

        # Exporter le DataFrame pour l'année courante dans le répertoire "output/clean"
        output_dir = "output/clean"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file_name = f"{output_dir}/services_pn_{year}.csv"
        df.to_csv(file_name, index=False)
        print(f"DataFrame for year {year} exported to {file_name}")

    # Concaténer les DataFrames de chaque année en un seul DataFrame
    final_dataframe = pd.concat(dataframes)

    # Exporter le DataFrame final dans le répertoire "output/clean"
    output_dir = "output/clean"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    final_file_name = f"{output_dir}/services_pn_all_years.csv"
    final_dataframe.to_csv(final_file_name, index=False)
    print(f"Final DataFrame exported to {final_file_name}")

    print(final_dataframe)
    final_dataframe.describe(include="all").to_csv(f"{output_dir}/describe.txt", index=False)

    # Ouvrir le fichier en mode écriture
    final_file_name = f"{output_dir}/services_pn_all_years_cols.txt"
    with open(final_file_name, 'w') as fichier:
        # Écrire chaque colonne dans une nouvelle ligne du fichier
        for colonne in final_dataframe.columns:
            fichier.write(colonne + '\n')



class BarPlotGenerator:
    def __init__(self, df):
        self.df = df.drop(columns=['Departement', 'DepartementPrincipal'])

    def generate_plots(self):
        # Vérifier si le répertoire "output/plots" existe, sinon le créer
        output_dir = "output/plots"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Générer un plot pour chaque colonne et le sauvegarder dans "output/plots"
        for column in self.df.columns:
            # Grouper les données par année et calculer la somme pour la colonne
            df_colonne = self.df.groupby('Année')[column].sum()

            plt.figure(figsize=(10, 6))  # Ajuster la taille du graphique
            plt.bar(df_colonne.index, df_colonne.values)
            plt.xlabel('Année')
            plt.ylabel(column)
            plt.title(f'Répartition des {column} par année')
            plt.xticks(df_colonne.index, rotation=90)  # Faire pivoter les étiquettes des années sur l'axe des x
            plt.tight_layout()  # Ajuster la disposition pour éviter que les étiquettes se chevauchent
            plt.savefig(f'{output_dir}/{column}_barplot.png')
            plt.close()

if __name__ == "__main__":
    # Créer une instance de BarPlotGenerator en utilisant le DataFrame final_dataframe
    plot_generator = BarPlotGenerator(final_dataframe)

    # Générer les barplots et les stocker dans le répertoire "output/plots"
    plot_generator.generate_plots()
