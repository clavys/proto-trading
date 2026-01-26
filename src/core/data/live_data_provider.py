"""
Live Data Provider Module
Maintient un canal de communication permanent avec les exchanges
(WebSocket - Réception uniquement).
"""


class LiveDataProvider:
    """
    Fournit des données en temps réel via WebSocket.
    """
    
    def __init__(self, websocket_url):
        pass
    
    def connect(self):
        """
        Établit une connexion WebSocket avec l'exchange.
        """
        pass
    
    def disconnect(self):
        """
        Ferme la connexion WebSocket.
        """
        pass
    
    def subscribe(self, channels):
        """
        S'abonne à des canaux de données spécifiques.
        """
        pass
    
    def unsubscribe(self, channels):
        """
        Se désabonne de canaux de données.
        """
        pass
    
    def on_message(self, message):
        """
        Callback appelé lors de la réception d'un message.
        """
        pass
    
    def get_latest_data(self):
        """
        Retourne les dernières données reçues.
        """
        pass
