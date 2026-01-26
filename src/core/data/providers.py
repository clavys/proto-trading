"""
Data Providers Module
Regroupe les différentes sources d'accès aux données historiques
(API REST, Fichiers locaux .csv, DB lecture/écriture).
"""


class BaseDataProvider:
    """
    Classe de base pour tous les fournisseurs de données.
    """
    
    def __init__(self):
        pass
    
    def fetch_data(self, params):
        """
        Récupère les données selon les paramètres fournis.
        """
        pass


class APIDataProvider(BaseDataProvider):
    """
    Fournisseur de données via API REST.
    """
    
    def __init__(self, api_url, api_key=None):
        super().__init__()
        pass
    
    def fetch_data(self, params):
        """
        Récupère les données depuis l'API.
        """
        pass


class CSVDataProvider(BaseDataProvider):
    """
    Fournisseur de données depuis des fichiers CSV locaux.
    """
    
    def __init__(self, data_directory):
        super().__init__()
        pass
    
    def fetch_data(self, params):
        """
        Récupère les données depuis des fichiers CSV.
        """
        pass


class DatabaseDataProvider(BaseDataProvider):
    """
    Fournisseur de données depuis une base de données.
    """
    
    def __init__(self, db_connection):
        super().__init__()
        pass
    
    def fetch_data(self, params):
        """
        Récupère les données depuis la base de données.
        """
        pass
    
    def write_data(self, data):
        """
        Écrit des données dans la base de données.
        """
        pass
