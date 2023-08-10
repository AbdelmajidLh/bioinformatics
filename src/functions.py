import datetime

def log(texte):
    maintenant = datetime.datetime.now()
    format_date_heure = maintenant.strftime("%Y-%m-%d %H:%M:%S")
    print(texte, format_date_heure)

# Appel de la fonction avec le texte en argument
#log("Date, heure et secondes actuelles :")