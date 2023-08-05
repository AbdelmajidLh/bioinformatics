import os
import re
import os.path
import seaborn as sns
#import sklearn as skl
import datetime as dt
import numpy as np
import pandas as pd
#import scipy
import matplotlib.pyplot as plt
#import statsmodels.api as sm

class DataProcessor:
    def __init__(self):
        self.df = None

    def load_data(self, file_path, sheet_name):
        # Charger le DataFrame en ignorant les deux premières lignes
        base1PN2012 = pd.read_excel(file_path, sheet_name=sheet_name)
        base1PN2012 = base1PN2012.drop([0, 1]).reset_index(drop=True)
        base1PN2012_b = base1PN2012.iloc[:, 1:].reset_index(drop=True) # Supprimer la première colonne

        # Transposer les données
        base1PN2012_transposed = base1PN2012_b.transpose()

        # Définir la deuxième ligne comme en-tête
        base1PN2012_transposed.columns = base1PN2012_transposed.iloc[0]
        df1 = base1PN2012_transposed[1:]
        df1['DepartementPrincipal'] = df1.index.str.split('.').str[0]

        # Grouper les lignes par départements principaux et effectuer la somme
        df1 = df1.groupby('DepartementPrincipal').sum()

        # Ajout colonne année
        df1 = df1.assign(Année=2012)

        # Créer une nouvelle colonne 'Departement' avec les valeurs de l'index
        df1['Departement'] = df1.index

        # Convertir la colonne 'Département' en facteur
        df1['Departement'] = df1['Departement'].astype('category')

        self.df = df1

    def get_dataframe(self):
        return self.df

if __name__ == "__main__":
    # Création d'une instance de DataProcessor
    processor = DataProcessor()

    # Remplacez 'chemin/vers/votre/fichier.xlsx' par le chemin réel de votre fichier Excel
    processor.load_data("data/Base de données 1 crimes et delits enregistres depuis-2012.xlsx", sheet_name='Services PN 2012')

    # Récupérer le DataFrame final dans une variable
    dataframe_final = processor.get_dataframe()
    print(dataframe_final)
