# Supprimer toutes les variables globales
#for variable in list(globals().keys()):
#    del globals()[variable]
from src.functions import *
log("[INFO] Début du process :")

import os
import re
import os.path
import seaborn as sns
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

log("[INFO] Début du data processing :")
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

log("[INFO] Données base 1 traitées : voir output :")

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


# print(final_dataframe.columns)
print("------------------------------------------------------")


import matplotlib.colors as mcolors

log("[INFO] Ploting Bar plots par département :")
class BarPlotByDepartementTop10Generator:
    def __init__(self, df):
        self.df = df.drop(columns=['Année', 'DepartementPrincipal'])

    def generate_plots(self):
        # Vérifier si le répertoire "output/plots" existe, sinon le créer
        output_dir = "output/plots"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Grouper les données par département et calculer la somme pour chaque colonne
        grouped_by_departement = self.df.groupby('Departement').sum()

        # Trier les départements par la somme de chaque colonne et sélectionner le top 10
        top_10_departements = grouped_by_departement.sum(axis=1).nlargest(10).index
        df_top_10 = grouped_by_departement.loc[top_10_departements]

        # Créer une palette de couleurs personnalisée allant du plus foncé au plus clair
        colors = mcolors.LinearSegmentedColormap.from_list('OrangeGrad', ['darkorange', 'orange'], N=len(self.df))

        # Générer un plot par colonne pour le top 10 des départements
        for column in self.df.columns:
            plt.figure(figsize=(10, 6))  # Ajuster la taille du graphique
            sorted_values = df_top_10[column].sort_values(ascending=False)  # Tri des valeurs en ordre décroissant
            plt.bar(sorted_values.index, sorted_values, color=colors(sorted_values.values))
            plt.xlabel('Département')
            plt.ylabel('Somme')
            plt.title(f'Somme des {column} par département (Top 10)')
            plt.xticks(rotation=90)  # Faire pivoter les étiquettes des départements sur l'axe des x
            plt.tight_layout()  # Ajuster la disposition pour éviter que les étiquettes se chevauchent
            plt.savefig(f'{output_dir}/{column}_barplot_top10_departements.png')
            plt.close()

if __name__ == "__main__":
     try:
         # Créer une instance de BarPlotByDepartementTop10Generator en utilisant le DataFrame final_dataframe
         plot_generator_top10_departements = BarPlotByDepartementTop10Generator(final_dataframe)

         # Générer les barplots pour le top 10 des départements et les stocker dans le répertoire "output/plots"
         plot_generator_top10_departements.generate_plots()
     except Exception as e:
         print("An error occurred while generating barplots by department:", e)



log("[INFO] Début du plots time series :")
class TimeSeriesGenerator:
    def __init__(self, df, columns):
        self.df = df
        self.columns = columns
        self.position = 'left'  # Initialize the starting position

    def generate_time_series(self):
        # Vérifier si le répertoire "output/time_series" existe, sinon le créer
        output_dir = "output/time_series"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Grouper les données par département
        grouped_df = self.df.groupby('Departement')

        # Création des séries temporelles pour chaque colonne spécifiée
        for column in self.columns:
            plt.figure(figsize=(12, 8))
            for departement, data in grouped_df:
                plt.plot(data['Année'], data[column])
                last_value = data[column].iloc[-1]

                # Alternate the position between left and right
                if self.position == 'left':
                    xy = (data['Année'].iloc[-1], last_value)
                    xytext = (-10, 10)
                    self.position = 'right'
                else:
                    xy = (data['Année'].iloc[-1], last_value)
                    xytext = (10, 10)
                    self.position = 'left'

                alpha_value = 0.5
                plt.annotate(departement, xy=xy, xytext=xytext,
                             textcoords='offset points', arrowprops=dict(arrowstyle="->", color='black'),
                             alpha=alpha_value)

            plt.xlabel('Année')
            plt.ylabel(column)
            plt.title(f'Série temporelle de {column} par départements')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/{column}_time_series.png')
            plt.close()

if __name__ == "__main__":
    # Liste des colonnes à générer en séries temporelles
    columns_to_plot = [
        "Autres coups et blessures volontaires criminels ou correctionnels",
        "Menaces ou chantages dans un autre but",
        "Vols violents sans arme contre des femmes sur voie publique ou autre lieu public",
        "Vols d'automobiles"
    ]

    # Créer une instance de TimeSeriesGenerator en utilisant le DataFrame final_dataframe
    time_series_generator = TimeSeriesGenerator(final_dataframe, columns_to_plot)

    # Générer les séries temporelles et les stocker dans le répertoire "output/time_series"
    time_series_generator.generate_time_series()

log("[INFO] Début du process Taux de chomage :")
class UnemploymentRatePlotter:
    def __init__(self, data_path, sheet_name, output_dir):
        self.data = pd.read_excel(data_path, sheet_name=sheet_name, header=3)
        print(self.data.columns)
        self.output_dir = output_dir
    
    def preprocess_data(self):
        numeric_columns = self.data.select_dtypes(include=[np.number])
        self.grouped_data = numeric_columns.groupby('Code').mean()
    
    def generate_plot(self):
        plt.figure(figsize=(12, 8))
        self.grouped_data.T.plot(ax=plt.gca())
        plt.title('Evolution du taux de chômage par région')
        plt.xlabel('Année')
        plt.ylabel('Valeur')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.02))
        plt.tight_layout()

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        output_path = os.path.join(self.output_dir, 'unemployment_rate_plot.png')
        plt.savefig(output_path)
        plt.close()

if __name__ == "__main__":
    data_path = "data/Base de donnée 4 Taux de de chomage par région en France.xls"
    sheet_name = 'Région2'
    output_dir = "output/plots/chaumage"
    
    plotter = UnemploymentRatePlotter(data_path, sheet_name, output_dir)
    plotter.preprocess_data()
    plotter.generate_plot()

log("[INFO] Début du process Pauvereté par département  :")
class PovertyRateBarPlotter:
    def __init__(self, data_path, sheet_name, output_dir, title, selected_departments):
        self.data = pd.read_excel(data_path, sheet_name=sheet_name, skiprows=4)  # Ignorer les 4 premières lignes
        self.output_dir = output_dir
        self.title = title
        self.selected_departments = selected_departments
    
    def preprocess_data(self):
        self.filtered_data = self.data[self.data['Code géographique'].isin(self.selected_departments)]
    
    def generate_plot(self):
        plt.figure(figsize=(12, 8))
        plt.bar(self.filtered_data['Code géographique'], self.filtered_data['Taux de pauvreté-Ensemble (%)'])
        plt.title(self.title)
        plt.xlabel('Départements')
        plt.ylabel('Taux de pauvreté')
        plt.tight_layout()

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        output_path = os.path.join(self.output_dir, f'poverty_rate_{self.title.lower().replace(" ", "_")}_barplot.png')
        plt.savefig(output_path)
        plt.close()

if __name__ == "__main__":
    data_path = "data/Base de données 3 Taux de pauvreté annuel.xlsx"
    output_dir = "output/plots/pauvrete"
    
    list_of_departments = [
        ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '21'],
        ['22', '23', '24', '25', '26', '27', '28', '29', '2A', '2B', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39'],
        ['40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59'],
        ['60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79'],
        ['80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '971', '972', '973', '974', '976', '987', '988'],
        # ... Add more lists of departments here
    ]
    
    for idx, departments in enumerate(list_of_departments):
        title = f"List {idx + 1}"
        plotter = PovertyRateBarPlotter(data_path, 'DEP', output_dir, title, departments)
        plotter.preprocess_data()
        plotter.generate_plot()

log("[INFO] Début du process Pauvreté all départements  :")
class PovertyRateBarPlotter:
    def __init__(self, data_path, sheet_name, output_dir, title, selected_departments):
        self.data = pd.read_excel(data_path, sheet_name=sheet_name, skiprows=4)  # Ignorer les 4 premières lignes
        self.output_dir = output_dir
        self.title = title
        self.selected_departments = selected_departments
    
    def preprocess_data(self):
        self.filtered_data = self.data[self.data['Code géographique'].isin(self.selected_departments)]
        self.filtered_data = self.filtered_data.sort_values(by='Taux de pauvreté-Ensemble (%)', ascending=False)
    
    def generate_plot(self):
        plt.figure(figsize=(12, 8))
        plt.bar(self.filtered_data['Code géographique'], self.filtered_data['Taux de pauvreté-Ensemble (%)'])
        plt.title(self.title)
        plt.xlabel('Départements')
        plt.ylabel('Taux de pauvreté')
        plt.xticks(rotation=90)  # Ajuster l'angle des étiquettes sur l'axe des x
        plt.tight_layout()

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        output_path = os.path.join(self.output_dir, f'poverty_rate_{self.title.lower().replace(" ", "_")}_barplot.png')
        plt.savefig(output_path)
        plt.close()

if __name__ == "__main__":
    data_path = "data/Base de données 3 Taux de pauvreté annuel.xlsx"
    output_dir = "output/plots/pauvrete"
    
    title = "Ensemble des départements"
    selected_departments = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '21', '22', '23', '24', '25', '26', '27', '28', '29', '2A', '2B', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '971', '972', '973', '974', '976', '987', '988']
    
    plotter = PovertyRateBarPlotter(data_path, 'DEP', output_dir, title, selected_departments)
    plotter.preprocess_data()
    plotter.generate_plot()



log("[INFO] Début du process Merge bases  :")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

class PovertyDataProcessor:
    def __init__(self, file_path, sheet_name, year_column):
        self.data = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=4)
        self.year_column = year_column
    
    def preprocess_data(self):
        self.data.rename(columns={'Code géographique': 'Département'}, inplace=True)
        self.data.drop(self.data.columns[[1, -1]], axis=1, inplace=True)
    
    def merge_with_year_data(self, year_data):
        merged_data = pd.merge(self.data, year_data, left_on='Département', right_on='Departement', how='outer')
        merged_data.drop(merged_data.columns[-1], axis=1, inplace=True)
        merged_data.dropna(inplace=True)
        merged_data["Département"] = merged_data["Département"].astype("category")
        return merged_data

class MissingValuesAnalyzer:
    def __init__(self, data):
        self.data = data
    
    def analyze_missing_values(self):
        missing_percentages = (self.data.isnull().sum() / len(self.data)) * 100
        return missing_percentages
    
    def generate_heatmap(self, output_dir):
        plt.figure(figsize=(15, 10))
        sns.heatmap(self.data.isnull(), cbar=False, cmap='viridis')
        plt.title('Heatmap of Missing Values')
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        plt.savefig(os.path.join(output_dir, 'missing_values_heatmap.png'))
        plt.close()

if __name__ == "__main__":
    file_path = "data/Base de données 3 Taux de pauvreté annuel.xlsx"
    sheet_name = "DEP"
    year_column = "Année"
    
    processor = PovertyDataProcessor(file_path, sheet_name, year_column)
    processor.preprocess_data()
    
    merged_data = processor.merge_with_year_data(final_dataframe[final_dataframe["Année"]==2019])
    print(merged_data)
    
    # Analyse des valeurs manquantes
    analyzer = MissingValuesAnalyzer(merged_data)
    missing_percentages = analyzer.analyze_missing_values()
    print("Percentage of Missing Values:\n", missing_percentages)
    
    # Génération du heatmap des valeurs manquantes
    output_dir = "output/plots/"
    analyzer.generate_heatmap(output_dir)



log("[INFO] Début du process plot correlations  :")

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from scipy import stats

# class DataAnalyzer:
#     def __init__(self, output_dir):
#         self.output_dir = output_dir
    
#     def analyze_and_plot(self, x_column, y_column, title, df_model):
#         # Convertir les colonnes en données numériques en utilisant .loc
#         df_model.loc[:, x_column] = pd.to_numeric(df_model[x_column], errors='coerce')
#         df_model.loc[:, y_column] = pd.to_numeric(df_model[y_column], errors='coerce')

#         # Filtrer les données non nulles en utilisant .loc
#         df_model.dropna(subset=[x_column, y_column], inplace=True)
        
#         # Supprimer les outliers en utilisant le test z-score
#         z_scores = np.abs(stats.zscore(df_model[[x_column, y_column]]))
#         threshold = 3
#         df_model = df_model[(z_scores < threshold).all(axis=1)]

#         # Extraire les données numériques
#         x = df_model[x_column].to_numpy()
#         y = df_model[y_column].to_numpy()

#         # Calculer la courbe de tendance
#         coefficients = np.polyfit(x, y, 1)
#         polynomial = np.poly1d(coefficients)
#         trendline = polynomial(x)

#         # Créer le nuage de points avec la courbe de tendance
#         plt.figure(figsize=(10, 6))
#         plt.scatter(x, y, label=f"{y_column} - {x_column}")
#         plt.plot(x, trendline, color="red", label="Courbe de tendance")
#         plt.xlabel(x_column)
#         plt.ylabel(y_column)
#         plt.title(title)
#         plt.legend()

#         # Sauvegarder le graphique
#         if not os.path.exists(os.path.join(self.output_dir, "stats")):
#             os.makedirs(os.path.join(self.output_dir, "stats"))
        
#         plot_filename = f"{x_column}_vs_{y_column}.png"
#         plot_path = os.path.join(self.output_dir, "stats", plot_filename)
#         plt.savefig(plot_path)
#         plt.close()

# if __name__ == "__main__":
#     output_dir = "output/plots"
#     variables = ["Taux de pauvreté-Ensemble (%)", "Médiane du niveau vie (€)", "Nombre de ménages fiscaux", 
#                  "Vols violents sans arme contre des établissements financiers,commerciaux ou industriels", 
#                  "Sequestrations", 'Menaces ou chantages pour extorsion de fonds', 'Menaces ou chantages dans un autre but', 
#                  'Atteintes à la dignité et à la  personnalité', 'Violations de domicile', 'Vols à main armée contre des établissements financiers', 
#                  'Vols à main armée contre des éts industriels ou commerciaux', 'Vols à main armée contre des entreprises de transports de fonds']
#     df_model = merged_data[variables]
    
#     analyzer = DataAnalyzer(output_dir)
#     analyzer.analyze_and_plot("Taux de pauvreté-Ensemble (%)", "Vols à main armée contre des éts industriels ou commerciaux", "Corrélation entre les vols à main armée et le taux de pauvreté", df_model)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

class DataAnalyzer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
  
    def analyze_and_plot(self, x_column, y_column, title, df_model):
        # Convertir les colonnes en données numériques en utilisant .loc
        df_model.loc[:, x_column] = pd.to_numeric(df_model[x_column], errors='coerce')
        df_model.loc[:, y_column] = pd.to_numeric(df_model[y_column], errors='coerce')

        # Filtrer les données non nulles en utilisant .loc
        df_model.dropna(subset=[x_column, y_column], inplace=True)
      
        # Supprimer les outliers en utilisant le test z-score
        z_scores = np.abs(stats.zscore(df_model[[x_column, y_column]]))
        threshold = 3
        df_model = df_model[(z_scores < threshold).all(axis=1)]

        # Extraire les données numériques
        x = df_model[x_column].to_numpy()
        y = df_model[y_column].to_numpy()

        # Calculer la courbe de tendance
        coefficients = np.polyfit(x, y, 1)
        polynomial = np.poly1d(coefficients)
        trendline = polynomial(x)

        # Créer le nuage de points avec la courbe de tendance
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, label=f"{y_column} - {x_column}")
        plt.plot(x, trendline, color="red", label="Courbe de tendance")
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title(title)
        plt.legend()

        # Sauvegarder le graphique
        if not os.path.exists(os.path.join(self.output_dir, "stats")):
            os.makedirs(os.path.join(self.output_dir, "stats"))
        
        plot_filename = f"{x_column}_vs_{y_column}.png"
        plot_path = os.path.join(self.output_dir, "stats", plot_filename)
        plt.savefig(plot_path)
        plt.close()

import itertools

if __name__ == "__main__":
    output_dir = "output/plots/stats"
    variables = ["Taux de pauvreté-Ensemble (%)", "Médiane du niveau vie (€)", "Nombre de ménages fiscaux", 
                 "Vols violents sans arme contre des établissements financiers,commerciaux ou industriels", 
                 "Sequestrations", 'Menaces ou chantages pour extorsion de fonds', 'Menaces ou chantages dans un autre but', 
                 'Atteintes à la dignité et à la  personnalité', 'Violations de domicile', 'Vols à main armée contre des établissements financiers', 
                 'Vols à main armée contre des éts industriels ou commerciaux', 'Vols à main armée contre des entreprises de transports de fonds']
    df_model = merged_data[variables]
  
    analyzer = DataAnalyzer(output_dir)
    
    for x_column, y_column in itertools.combinations(df_model.columns, 2):
        title = f"Corrélation entre {y_column} et {x_column}"
        analyzer.analyze_and_plot(x_column, y_column, title, df_model)


# partie ECONOMETRIE
merged_data.to_csv("output/clean/merged_data.csv")
print("------------------- END --------------------")