import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from linearmodels.panel import PanelOLS
import statsmodels.api as sm
import seaborn as sns
from sklearn.preprocessing import StandardScaler


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


# fonctions
def calculate_missing_percentages(dataframe):
    missing_percentages = (dataframe.isnull().mean() * 100).round(2)
    return missing_percentages

def filter_columns_with_high_missing(df, threshold):
    missing_percentages = calculate_missing_percentages(df)
    high_missing_columns = missing_percentages[missing_percentages > threshold].index
    return df.drop(high_missing_columns, axis=1)

# ## 1 - Préparation des données :
# 
# Supprimez les colonnes qui ne sont pas pertinentes pour notre analyse, par exemple, les colonnes "Libellé" et vérifiez et traitez les valeurs manquantes si nécessaire.


# service PN : 2012-2020
df_servicePN = pd.read_csv("output/clean/services_pn_all_years.csv")


# taux de criminalité
df_tauxCrim = pd.read_excel('data/Base de données (annexe) taux de criminalité annuel en France.xls', skiprows=3)
df_tauxCrim.columns = ["Année", "Taux de criminalité"]
df_tauxCrim = df_tauxCrim.drop(df_tauxCrim.tail(3).index)
df_tauxCrim['Année'] = df_tauxCrim['Année'].astype(int)

df_tauxCrim.tail(5)


# Merger df_servicePN et df_tauxCrim ==> df1
df1 = pd.merge(df_servicePN, df_tauxCrim, on="Année", how='left')
df1.shape

# taux de chaumage par departement
df_chomage = pd.read_excel("data/Base de donnée 4 Taux de de chomage par région en France.xls", sheet_name='Département2', skiprows=3)

df = df_chomage
# Sélectionnez les colonnes qui vous intéressent (Code, Libellé et les colonnes de trimestres)
selected_columns = ['Code', 'Libellé'] + [col for col in df.columns if col.startswith('T')]

# Créez un DataFrame temporaire avec les colonnes sélectionnées
temp_df = df[selected_columns]

# Utilisez la fonction melt pour transformer les colonnes de trimestres en lignes
melted_df = pd.melt(temp_df, id_vars=['Code', 'Libellé'], var_name='Trimestre', value_name='Taux_chômage')

# Enlevez le préfixe 'T' des valeurs de trimestre
melted_df['Trimestre'] = melted_df['Trimestre'].str[3:]

# Convertissez la colonne 'Trimestre' en datetime pour faciliter le groupby par année
#melted_df['Trimestre'] = pd.to_datetime(melted_df['Trimestre'])

# Groupby par 'Trimestre == Année', en calculant la moyenne pour chaque groupe
result = melted_df.groupby(['Trimestre', 'Code', 'Libellé'])['Taux_chômage'].mean().reset_index()

# Renommez les colonnes
result.rename(columns={'Trimestre': 'Année', 'Code': 'Departement'}, inplace=True)

result['Année'] = result['Année'].astype('int64')


# Affichez le nouveau DataFrame au format long
print(result.head(3))

df1.query("Année >= 2012 and Année <= 2015")["Departement"].unique()

# Définir le seuil de pourcentage de valeurs manquantes
seuil = 80

# Filtrer les colonnes avec un pourcentage de valeurs manquantes > seuil
filtered_df = filter_columns_with_high_missing(df1, seuil)

filtered_df.head(3)


# supprimer les colonnes inutils
colonnes_a_supprimer = ['DepartementPrincipal']
df1 = df1.drop(columns=colonnes_a_supprimer)

# Convert the "Departement" column to string
df1["Departement"] = df1["Departement"].astype(str)
result["Departement"] = result["Departement"].astype(str)

# Perform the inner merge
df2 = df1.merge(result, on=['Departement', 'Année'], how='inner')


# Merger df_servicePN et df_tauxCrim et taux de chaumage par departement ==> df2
df2 = df1.merge(result, on=['Departement', 'Année'], how='inner')


# Base - taux de pauvereté
chemin_fichier = 'data/Base de données 3 Taux de pauvreté annuel.xlsx'
nom_feuille = 'DEP_2'
df_pauvrete = pd.read_excel(chemin_fichier, sheet_name=nom_feuille)

# renommer
df_pauvrete.rename(columns={'Libellé géographique': 'Libellé', 'Code géographique': 'Departement'}, inplace=True)

# supprimer la colonne Libellé
df_pauvrete = df_pauvrete.drop('Libellé', axis=1)

# merge pauvreté avec les autres ==> df3
df3 = df2.merge(df_pauvrete, on=['Departement', 'Année'], how='left')
df3.shape


# afficher les colonnes
column_names = df3.columns.tolist()
print(column_names)


# ### Traiter les valeurs manquantes

# Vérifier les valeurs manquantes par colonne
missing_values = df3.isnull().sum()
print(missing_values)

# # Imputer les valeurs manquantes pour la colonne Taux de chomage

# ## Calculer la médiane de la colonne "Taux de chômage"
# median_value = df3["Taux_chômage"].median()

# ## Imputer les valeurs manquantes avec la médiane
# df3["Taux_chômage"].fillna(median_value, inplace=True)


# Supprimer les colonnes avec plus de 50% de valeurs manquantes
threshold = 0.5
data_cleaned = df3.dropna(thresh=threshold*len(df3), axis=1)

# Imputer les valeurs manquantes par la médiane

median_pauvrete = data_cleaned["Taux de pauvreté-Ensemble (%)"].median()
median_chomage = data_cleaned["Taux_chômage"].median()

data_cleaned["Taux de pauvreté-Ensemble (%)"].fillna(median_pauvrete, inplace=True)
data_cleaned["Taux_chômage"].fillna(median_chomage, inplace=True)

# verification
remaining_missing = data_cleaned.isnull().sum()
print(remaining_missing)

# ## 2 - Exploration des données :
# 
# Visualisez la distribution des différents types de crimes pour avoir une idée générale de leur répartition.
# Visualisez la corrélation entre les différents types de crimes et les facteurs économiques (taux de chômage, médiane du niveau de vie, etc.). On va utiliser des graphiques de dispersion, des matrices de corrélation, etc.

# On va se focaliser sur une liste de variables de criminalité pour l'ensemble des analyses.

# Le critère de sélection est basé sur les armes, la mort et le danger pour les autres, 
# voici une liste de 10 crimes et délits à analyser parmi l'ensemble des crimes enregistrés par la PN :
variables_criminalite = ["Règlements de compte entre malfaireurs", "Homicides pour voler et à l'occasion de vols",
                         "Tentatives d'homicides pour voler et à l'occasion de vols", "Tentatives homicides pour d'autres motifs",
                         'Coups et blessures volontaires suivis de mort', 'Vols à main armée contre des établissements financiers',
                         'Vols à main armée contre des entreprises de transports de fonds', 'Vols à main armée contre des particuliers à leur domicile',
                         "Vols avec armes blanches contre des établissements financiers,commerciaux ou industriels",
                         'Vols avec armes blanches contre des particuliers à leur domicile']



autres_variables = ['Année', 'Departement', 'Taux_chômage', 'Nombre de ménages fiscaux', 'Nombre de personnes dans les ménages fiscaux',
                    'Médiane du niveau vie (€)','Part des ménages fiscaux imposés (%)','Taux de pauvreté-Ensemble (%)',
                    'Taux de pauvreté-moins de 30 ans (%)', 'Taux de pauvreté-30 à 39 ans  (%)', 'Taux de pauvreté-40 à 49 ans (%)',
                    'Taux de pauvreté-50 à 59 ans (%)', 'Taux de pauvreté-60 à 74 ans (%)', 'Taux de pauvreté-75 ans ou plus (%)',
                    'Taux de pauvreté-propriétaires (%)', 'Taux de pauvreté-locataires (%)', "Part des revenus d'activité (%)",
                    'Part des revenus du patrimoine et autres revenus (%)', "Part de l'ensemble des prestations sociales (%)",
                    'dont part des prestations familiales (%)', 'dont part des minima sociaux (%)', 'Part des impôts (%)', '1er décile du niveau de vie (€)',
                    '9e décile du niveau de vie (€)', 'Rapport interdécile 9e décile/1er décile']
df5 = data_cleaned[variables_criminalite + autres_variables]

# Calculer les statistiques descriptives
print(df5.describe())


# Calculer la table de corrélation
correlation_table = df5.corr()

# seuil de correlation
threshold = 0.0
strong_correlations = correlation_table.abs() > threshold

# Créer la heatmap des corrélations les plus fortes
plt.figure(figsize=(30, 30))
sns.heatmap(correlation_table[strong_correlations], cmap='coolwarm', annot=True)
plt.title('Corrélations les plus fortes')

# Assurez-vous que le répertoire de sortie existe
output_dir = 'output/plots/econometrie'
os.makedirs(output_dir, exist_ok=True)

# Sauvegarder la figure au format PNG
output_path = os.path.join(output_dir, 'heatmap.png')
plt.savefig(output_path)

# Afficher la figure
plt.show()



# exporter la table des correlations
output_dir = 'output/econometrie'
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, 'correlations_selected_vars.xlsx')
correlation_table.to_excel(output_path, index=True)


# Effectuer l'encodage one-hot pour la colonne "Departement"
df_encoded = pd.get_dummies(df5, columns=['Departement'], drop_first=True)

df_encoded.head(3)

target_variables = ["Règlements de compte entre malfaireurs", "Homicides pour voler et à l'occasion de vols",
                         "Tentatives d'homicides pour voler et à l'occasion de vols", "Tentatives homicides pour d'autres motifs",
                         'Coups et blessures volontaires suivis de mort', 'Vols à main armée contre des établissements financiers',
                         'Vols à main armée contre des entreprises de transports de fonds', 'Vols à main armée contre des particuliers à leur domicile',
                         "Vols avec armes blanches contre des établissements financiers,commerciaux ou industriels",
                         'Vols avec armes blanches contre des particuliers à leur domicile']



feature_variables = ['Année', 'Taux_chômage', 'Nombre de ménages fiscaux', 'Nombre de personnes dans les ménages fiscaux',
                    'Médiane du niveau vie (€)','Part des ménages fiscaux imposés (%)','Taux de pauvreté-Ensemble (%)',
                    'Taux de pauvreté-moins de 30 ans (%)', 'Taux de pauvreté-30 à 39 ans  (%)', 'Taux de pauvreté-40 à 49 ans (%)',
                    'Taux de pauvreté-50 à 59 ans (%)', 'Taux de pauvreté-60 à 74 ans (%)', 'Taux de pauvreté-75 ans ou plus (%)',
                    'Taux de pauvreté-propriétaires (%)', 'Taux de pauvreté-locataires (%)', "Part des revenus d'activité (%)",
                    'Part des revenus du patrimoine et autres revenus (%)', "Part de l'ensemble des prestations sociales (%)",
                    'dont part des prestations familiales (%)', 'dont part des minima sociaux (%)', 'Part des impôts (%)', '1er décile du niveau de vie (€)',
                    '9e décile du niveau de vie (€)', 'Rapport interdécile 9e décile/1er décile']

# %%
# Import necessary libraries
import pandas as pd
from sklearn.impute import SimpleImputer

# Create a copy of the df_encoded DataFrame
df_imputed = df_encoded.copy()

# List of columns with missing values
columns_with_missing = df_imputed.columns[df_imputed.isnull().any()]

# Initialize a SimpleImputer with median strategy
imputer = SimpleImputer(strategy='median')

# Impute missing values with the median for selected columns
df_imputed[columns_with_missing] = imputer.fit_transform(df_imputed[columns_with_missing])

# Now you can proceed with your regression analysis on the df_imputed DataFrame


# %%
print(df_imputed.isnull().sum())


# %% [markdown]
# Le code présent met en œuvre une analyse de régression linéaire avec la technique de sélection de variables Lasso (Least Absolute Shrinkage and Selection Operator) pour quantifier l'effet des facteurs économiques sur différentes variables cibles. Dans un premier temps, le code normalise les caractéristiques économiques pour préparer les données à l'analyse. Ensuite, il applique le modèle Lasso pour chaque variable cible, en sélectionnant les variables explicatives les plus pertinentes pour chaque relation. Les coefficients sélectionnés par le modèle Lasso, qui représentent l'importance des variables explicatives, sont affichés. Le code construit ensuite un modèle de régression linéaire basé sur les caractéristiques choisies par Lasso, et évalue ses performances en utilisant le coefficient de détermination R². En parallèle, le résumé statistique de la régression est affiché, fournissant des informations détaillées sur les coefficients de régression, leurs significativités et d'autres statistiques pertinentes. Enfin, les résultats de chaque analyse sont enregistrés dans des fichiers texte, facilitant la documentation et la présentation des conclusions. Cette approche permet de cerner les facteurs économiques les plus influents sur chaque variable cible, fournissant ainsi des informations cruciales pour comprendre les relations économiques sous-jacentes et prendre des décisions éclairées.

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LinearRegression
import statsmodels.api as sm

# Définir le chemin du répertoire de sortie
output_dir = "output/econometrie/regression_lineaire/"
os.makedirs(output_dir, exist_ok=True)

# Pour chaque variable cible, effectuez le processus d'analyse de regression
for target_variable in target_variables:
    # Étape 1: Sélection des variables avec Lasso
    X = df_imputed[feature_variables]  # Features
    y = df_imputed[target_variable]  # Target

    # Normaliser les fonctionnalités
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Ajuster le modèle Lasso
    lasso = Lasso(alpha=0.01, max_iter=10000)
    lasso.fit(X_scaled, y)

    # Afficher les coefficients sélectionnés par le modèle Lasso
    selected_features = X.columns[lasso.coef_ != 0]
    print("Fonctionnalités sélectionnées par Lasso pour", target_variable, ":", selected_features)

    # Étape 2: Construction du modèle de régression linéaire avec Lasso
    # Utiliser uniquement les fonctionnalités sélectionnées par Lasso
    X_selected = X[selected_features]

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # Construire un modèle de régression linéaire
    model_with_lasso = LinearRegression()
    model_with_lasso.fit(X_train, y_train)

    # Évaluer le modèle sur l'ensemble de test
    r2_score = model_with_lasso.score(X_test, y_test)
    print("Score R² du modèle de régression avec Lasso pour", target_variable, ":", r2_score)

    # Afficher le summary de la regression
    X_selected = sm.add_constant(X_selected)
    model_with_lasso = sm.OLS(y, X_selected)
    results = model_with_lasso.fit()
    print("Résumé de la régression pour", target_variable, ":\n", results.summary())

    # Écrire les résultats dans un fichier texte
    output_file = os.path.join(output_dir, f"results_{target_variable}.txt")
    with open(output_file, "w") as f:
        f.write("Fonctionnalités sélectionnées par Lasso : " + str(selected_features) + "\n")
        f.write("Score R² du modèle de régression avec Lasso : " + str(r2_score) + "\n")
        f.write("Résumé de la régression :\n" + str(results.summary()))

    print(f"Résultats pour {target_variable} exportés dans {output_file}")



# 
# ## 5 - Interprétation des résultats :
# 
# Analysez les coefficients de régression pour déterminer comment chaque facteur économique affecte la criminalité. Les coefficients positifs indiquent une augmentation de la criminalité avec une augmentation du facteur, tandis que les coefficients négatifs indiquent une diminution de la criminalité.
# Vérifiez la significativité statistique des coefficients et l'ajustement global du modèle.
# 
# 
# 

# ## 8 -  Modele à effet fixe
# 
# La régression avec effets fixes (Fixed Effects Regression) est une méthode d'analyse statistique utilisée pour modéliser la relation entre une variable dépendante (cible) et plusieurs variables indépendantes (explicatives) en tenant compte des effets spécifiques aux groupes ou aux unités dans les données. Cette méthode est couramment utilisée dans le contexte des données de panel, où vous avez des observations sur plusieurs unités (par exemple, des individus, des entreprises ou des régions) à plusieurs moments dans le temps.
# 
# L'objectif principal de la régression avec effets fixes est de contrôler les effets inobservés et constants spécifiques à chaque unité, qui pourraient biaiser les résultats de la régression. Ces effets inobservés peuvent être des caractéristiques propres à chaque unité qui ne varient pas dans le temps mais qui pourraient influencer la variable dépendante.
# 
# L'approche clé de la régression avec effets fixes est d'introduire des variables indicatrices (dummies) pour chaque unité dans le modèle de régression. Ces variables indicatrices capturent les effets spécifiques à chaque unité qui restent constants dans le temps. En incluant ces variables indicatrices dans le modèle, vous éliminez l'effet des différences inobservées entre les unités.
# 
# L'intérêt de la régression avec effets fixes réside dans le fait qu'elle permet de contrôler les effets inobservés fixes qui pourraient être confondus avec les relations que vous essayez de modéliser. Cela peut être particulièrement utile lorsque vous analysez des données de panel où les unités individuelles diffèrent de manière systématique et que vous voulez isoler les effets des variables explicatives sur la variable dépendante.
# 
# En résumé, les principaux avantages de la régression avec effets fixes sont :
# 
# 1. Contrôle des effets inobservés spécifiques à chaque unité.
# 2. Possibilité de modéliser les relations entre les variables explicatives et la variable dépendante tout en tenant compte des variations entre les unités.
# 3. Réduction du risque de biais de sélection ou de variables omises.
# 
# Cependant, la régression avec effets fixes peut être sensible à la collinéarité entre les variables explicatives et les effets fixes, et elle ne permet pas de capturer les effets qui varient au fil du temps. Si vous avez des effets variables dans le temps ou si vous souhaitez capturer des relations globales plutôt que spécifiques aux unités, d'autres méthodes comme la régression à effets aléatoires pourraient être plus appropriées.
# 
# 
# -------------------------
# Une analyse de régression à effets fixes, également connue sous le nom de modèle à effets fixes, est utilisée lorsque l'on souhaite contrôler les effets constants non observés entre les unités individuelles dans une analyse longitudinale ou en panel. Dans ce type de modèle, on introduit des variables indicatrices pour chaque unité individuelle (comme les individus, les départements, etc.) afin de capturer les effets spécifiques à ces unités.
# 
# Pour une analyse de régression à effets fixes, vous pouvez introduire les variables suivantes :
# 
# 1. **Variables temporelles :** Si vos données sont longitudinales, c'est-à-dire collectées sur plusieurs périodes, vous pouvez inclure des variables indicatrices pour chaque période. Cela permettra de contrôler les effets temporels communs à toutes les unités.
# 
# 2. **Variables indicatrices d'unité :** Introduisez des variables indicatrices pour chaque unité individuelle (comme les départements, les individus, etc.). Par exemple, si vous avez des données pour plusieurs départements, vous pourriez créer une série de variables indicatrices, une pour chaque département, qui prend la valeur 1 si l'observation appartient à ce département et 0 sinon. Cela vous permettra de capturer les effets fixes spécifiques à chaque département.
# 
# 3. **Variables explicatives :** En plus des variables indicatrices d'unité et temporelles, vous pouvez également inclure d'autres variables explicatives qui sont susceptibles d'influencer la variable cible. Assurez-vous que ces variables sont mesurées au niveau de l'unité et ne varient pas dans le temps, car le modèle à effets fixes ne capture que les variations inter-unités, pas les variations dans le temps.
# 
# 4. **Variables de contrôle :** Comme dans une régression classique, vous pouvez inclure des variables de contrôle qui peuvent influencer la relation entre les variables explicatives et la variable cible.
# 
# Lorsque vous spécifiez un modèle à effets fixes, il est important de prendre en compte que les variables indicatrices d'unité et temporelles augmentent la dimension de votre modèle, ce qui peut avoir un impact sur l'estimation des coefficients et la gestion des degrés de liberté. De plus, les modèles à effets fixes supposent que les effets spécifiques à chaque unité sont constants dans le temps et ne sont pas corrélés avec les variables explicatives.

df8 = df5

# supprimer les Departement texte (2A ...)
df8 = df8[~df8['Departement'].isin(['2A', '2B'])]

# convertir la colonne en numeric
#df8['Departement'] = pd.to_numeric(df8['Departement'], errors='coerce')
df8.loc[:, 'Departement'] = pd.to_numeric(df8['Departement'], errors='coerce')

# renommer les colonnes 
new_columns = ["Reglements_de_compte_entre_malfaireurs", "Homicides_pour_voler_et_a_loccasion_de_vols",
"Tentatives_dhomicides_pour_voler_et_a_loccasion_de_vols", "Tentatives_homicides_pour_dautres_motifs",
"Coups_et_blessures_volontaires_suivis_de_mort", "Vols_a_main_armee_contre_des_etablissements_financiers",
"Vols_a_main_armee_contre_des_entreprises_de_transports_de_fonds", "Vols_a_main_armee_contre_des_particuliers_a_leur_domicile",
"Vols_avec_armes_blanches_contre_des_etablissements_financiers_commerciaux_ou_industriels", "Vols_avec_armes_blanches_contre_des_particuliers_a_leur_domicile",
"Annee", "Departement", "Taux_chomage", "Nombre_de_menages_fiscaux", "Nombre_de_personnes_dans_les_menages_fiscaux",
"Mediane_du_niveau_vie", "Part_des_menages_fiscaux_imposes", "Taux_de_pauvrete_Ensemble", "Taux_de_pauvrete_moins_de_30_ans",
"Taux_de_pauvrete_30_a_39_ans", "Taux_de_pauvrete_40_a_49_ans", "Taux_de_pauvrete_50_a_59_ans", "Taux_de_pauvrete_60_a_74_ans",
"Taux_de_pauvrete_75_ans_ou_plus", "Taux_de_pauvrete_proprietaires", "Taux_de_pauvrete_locataires", "Part_des_revenus_dactivite", "Part_des_revenus_du_patrimoine_et_autres_revenus",
"Part_de_lensemble_des_prestations_sociales", "dont_part_des_prestations_familiales", "dont_part_des_minima_sociaux", "Part_des_impots", "premier_decile_du_niveau_de_vie",
"neuvieme_decile_du_niveau_de_vie", "Rapport_interdecile_9e_decile_premier_decile"]
df8.columns = new_columns

#df8["Departement"] = df8["Departement"].astype(str)
df8['Departement'] = df8['Departement'].astype('category')


# Liste des variables explicatives
vars_explicatives = ["Annee", "Departement", "Taux_chomage", "Nombre_de_menages_fiscaux", "Nombre_de_personnes_dans_les_menages_fiscaux",
"Mediane_du_niveau_vie", "Part_des_menages_fiscaux_imposes", "Taux_de_pauvrete_Ensemble", "Taux_de_pauvrete_moins_de_30_ans",
"Taux_de_pauvrete_30_a_39_ans", "Taux_de_pauvrete_40_a_49_ans", "Taux_de_pauvrete_50_a_59_ans", "Taux_de_pauvrete_60_a_74_ans",
"Taux_de_pauvrete_75_ans_ou_plus", "Taux_de_pauvrete_proprietaires", "Taux_de_pauvrete_locataires", "Part_des_revenus_dactivite", "Part_des_revenus_du_patrimoine_et_autres_revenus",
"Part_de_lensemble_des_prestations_sociales", "dont_part_des_prestations_familiales", "dont_part_des_minima_sociaux", "Part_des_impots", "premier_decile_du_niveau_de_vie",
"neuvieme_decile_du_niveau_de_vie", "Rapport_interdecile_9e_decile_premier_decile"] 

# Liste des variables cibles
targets = ["Reglements_de_compte_entre_malfaireurs", "Homicides_pour_voler_et_a_loccasion_de_vols",
"Tentatives_dhomicides_pour_voler_et_a_loccasion_de_vols", "Tentatives_homicides_pour_dautres_motifs",
"Coups_et_blessures_volontaires_suivis_de_mort", "Vols_a_main_armee_contre_des_etablissements_financiers",
"Vols_a_main_armee_contre_des_entreprises_de_transports_de_fonds", "Vols_a_main_armee_contre_des_particuliers_a_leur_domicile",
"Vols_avec_armes_blanches_contre_des_etablissements_financiers_commerciaux_ou_industriels", "Vols_avec_armes_blanches_contre_des_particuliers_a_leur_domicile"]


# Regression pour chaque binome de variables avec effet fixe : sans tests de robustesse
import statsmodels.formula.api as smf
import os

targets = targets

# Création du dossier output/regression si il n'existe pas
if not os.path.exists('output/econometrie/regression_effet_fixe_1to1'):
    os.makedirs('output/econometrie/regression_effet_fixe_1to1')

for target in targets:
    for variable in vars_explicatives:
        # Créer la formule pour la régression à effets fixes
        # Ici, je suppose que 'Annee' est votre variable de temps et 'Departement' votre variable cross-section
        # 'C(Departement)' crée des dummies pour chaque département (effet fixe)
        formula = f"{target} ~ {variable} + C(Departement) + Annee  + Annee :C(Departement)"
        model = smf.ols(formula, data=df8)
        result = model.fit()

        # Écrire les résultats de la régression dans un fichier texte
        with open(f'output/econometrie/regression_effet_fixe_1to1/{target}_{variable}_results.txt', 'w') as file:
            file.write(str(result.summary()))

print("les resultats sont stockée dans : output/econometrie/regression_effet_fixe_1to1")



# # voir la partie en bas avec test de robustesse
# """
# Avec ce code, une double boucle parcourt chaque variable cible et chaque variable explicative. 
# Pour chaque paire cible-explicative, un modèle de régression à effets fixes est ajusté en tenant compte des effets fixes des départements et des 
# interactions entre les années et les départements. Les résultats de chaque régression sont enregistrés dans des fichiers texte distincts. Vous obtiendrez 
# ainsi une série de résultats qui vous montrera comment chaque variable explicative influence chaque variable cible dans le contexte d'un modèle de régression à effets fixes.

# """
# import statsmodels.formula.api as smf
# import os

# targets = ["Reglements_de_compte_entre_malfaireurs", "Homicides_pour_voler_et_a_loccasion_de_vols",
#            "Tentatives_dhomicides_pour_voler_et_a_loccasion_de_vols", "Tentatives_homicides_pour_dautres_motifs",
#            "Coups_et_blessures_volontaires_suivis_de_mort", "Vols_a_main_armee_contre_des_etablissements_financiers",
#            "Vols_a_main_armee_contre_des_entreprises_de_transports_de_fonds", "Vols_a_main_armee_contre_des_particuliers_a_leur_domicile",
#            "Vols_avec_armes_blanches_contre_des_etablissements_financiers_commerciaux_ou_industriels", "Vols_avec_armes_blanches_contre_des_particuliers_a_leur_domicile"]

# # Création du dossier output/regression si il n'existe pas
# if not os.path.exists('output/econometrie/regression_effet_fixe'):
#     os.makedirs('output/econometrie/regression_effet_fixe')

# # Itérer sur les variables cibles
# for target in targets:
#     # Créer la formule pour la régression à effets fixes
#     # Ici, je suppose que 'Annee' est votre variable de temps et 'Departement' votre variable cross-section
#     # 'C(Departement)' crée des dummies pour chaque département (effet fixe)
#     formula = f"{target} ~ {' + '.join(vars_explicatives)} + C(Departement) + Annee  + Annee :C(Departement)"
    
#     # Ajuster le modèle de régression
#     model = smf.ols(formula, data=df8)
#     result = model.fit()

#     # Écrire les résultats de la régression dans un fichier texte
#     with open(f'output/econometrie/regression_effet_fixe/{target}_results.txt', 'w') as file:
#         file.write(str(result.summary()))


# ### Test de robuste - Test de normalité des résidus


# transformation log
import statsmodels.formula.api as smf
import os
from scipy import stats
import numpy as np

# Liste des variables cibles
targets = ["Reglements_de_compte_entre_malfaireurs", "Homicides_pour_voler_et_a_loccasion_de_vols",
           "Tentatives_dhomicides_pour_voler_et_a_loccasion_de_vols", "Tentatives_homicides_pour_dautres_motifs",
           "Coups_et_blessures_volontaires_suivis_de_mort", "Vols_a_main_armee_contre_des_etablissements_financiers",
           "Vols_a_main_armee_contre_des_entreprises_de_transports_de_fonds", "Vols_a_main_armee_contre_des_particuliers_a_leur_domicile",
           "Vols_avec_armes_blanches_contre_des_etablissements_financiers_commerciaux_ou_industriels", "Vols_avec_armes_blanches_contre_des_particuliers_a_leur_domicile"]

# Création du dossier output/regression si il n'existe pas
if not os.path.exists('output/econometrie/regression_effet_fixe'):
    os.makedirs('output/econometrie/regression_effet_fixe')

# Charger et préparer les données dans le DataFrame df8

# Itérer sur les variables cibles
for target in targets:
    # Créer la formule pour la régression à effets fixes
    formula = f"{target} ~ {' + '.join(vars_explicatives)} + C(Departement) + Annee  + Annee:C(Departement)"
    
    # Ajuster le modèle de régression
    model = smf.ols(formula, data=df8)
    result = model.fit()

    # Effectuer la transformation logarithmique sur les résidus
    residuals = result.resid
    residuals_log = np.log(residuals)

    # Test de normalité des résidus
    p_value = stats.shapiro(residuals_log)[1]
    alpha = 0.05  # Niveau de signification

    if p_value < alpha:
        normality_result = "Le test de Shapiro-Wilk rejette l'hypothèse nulle (résidus non normaux)"
    else:
        normality_result = "Le test de Shapiro-Wilk ne peut pas rejeter l'hypothèse nulle (résidus normaux)"

    # Écrire les résultats de la régression et le test de normalité dans un fichier texte
    results_summary = str(result.summary())
    results_with_normality = f"{results_summary}\n\nTest de normalité des résidus:\n{normality_result}"

    with open(f'output/econometrie/regression_effet_fixe/{target}_results.txt', 'w') as file:
        file.write(results_with_normality)


# Le test de normalité des résidus que vous avez effectué indique que les résidus ne suivent pas une distribution normale. Cela peut avoir des implications sur l'interprétation des résultats de votre modèle de régression à effets fixes. Lorsque les résidus ne sont pas normalement distribués, certaines des hypothèses sous-jacentes des méthodes de régression peuvent être violées.
# 
# Dans de tels cas, vous pourriez envisager certaines approches pour traiter ce problème :
# 
# 1. **Transformation des données** : Vous pouvez essayer d'appliquer des transformations aux variables cibles ou explicatives pour rendre les résidus plus proches d'une distribution normale. Les transformations courantes incluent la transformation logarithmique, la racine carrée, etc.
# 
# 2. **Utilisation de méthodes robustes** : Si vos résidus ne sont pas normaux en raison de valeurs aberrantes ou extrêmes, vous pourriez envisager d'utiliser des méthodes de régression robustes qui sont moins sensibles aux valeurs aberrantes.
# 
# 3. **Revoir les hypothèses du modèle** : Si la violation de la normalité des résidus est due à des raisons structurelles ou conceptuelles, il se peut que votre modèle ne soit pas adapté pour capturer la relation entre les variables. Dans ce cas, il pourrait être nécessaire de revoir votre modèle ou vos hypothèses.
# 
# 4. **Utilisation d'autres méthodes statistiques** : Si la normalité des résidus est un problème persistant, vous pourriez envisager d'utiliser des méthodes statistiques non paramétriques ou d'autres techniques qui ne supposent pas la normalité des résidus.
# 
# 5. **Ensemble de données plus large** : Si votre ensemble de données est relativement petit, la distribution des résidus peut sembler non normale même si elle suit une distribution normale dans la population. Avoir un ensemble de données plus large pourrait aider à mieux estimer la distribution des résidus.
# 
# 6. **Consultation d'un expert** : Si la violation de la normalité des résidus est un problème crucial pour votre analyse, il peut être judicieux de consulter un statisticien ou un expert en domaine pour obtenir des conseils spécifiques à votre situation.
# 
# En résumé, il est important de comprendre l'impact de la non-normalité des résidus sur vos résultats et de prendre des mesures appropriées pour traiter ce problème en fonction de la nature de vos données et de vos objectifs d'analyse.


# Si le test de normalité des résidus indique que l'hypothèse nulle ne peut pas être rejetée (c'est-à-dire que les résidus sont normaux), cela est généralement une bonne nouvelle. Cela signifie que les résidus de votre modèle de régression à effets fixes sont conformes à l'assomption de normalité, ce qui renforce la validité de vos résultats.
# afficher les colonnes
column_names = df8.columns.tolist()
print(column_names)
