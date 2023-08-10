import pandas as pd

class PovertyDataProcessor:
    def __init__(self, file_path, sheet_name):
        self.data = pd.read_excel(file_path, sheet_name=sheet_name)
    
    def preprocess_data(self):
        self.data = self.data.drop([0, 1]).reset_index(drop=True)
        self.data.rename(columns={'Code géographique': 'Département'}, inplace=True)
        self.data = self.data.drop(self.data.columns[[1, -1]], axis=1)
    
    def show_modified_data(self):
        print(self.data)

if __name__ == "__main__":
    file_path = "data/Base de données 3 Taux de pauvreté annuel.xlsx"
    sheet_name = "DEP"
    
    processor = PovertyDataProcessor(file_path, sheet_name)
    processor.preprocess_data()
    processor.show_modified_data()
