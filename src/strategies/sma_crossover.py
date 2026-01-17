# src/strategies/sma_cross.py
import pandas as pd
import pandas_ta as ta
from src.core.signal import TradeSignal, SignalAction, OrderType

class SMACrossStrategy:
    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        """
        Stratégie générique de croisement de moyennes mobiles.
        :param fast_period: Période de la moyenne mobile rapide (ex: 20)
        :param slow_period: Période de la moyenne mobile lente (ex: 50)
        """
        self.name = f"SMA_Cross_{fast_period}_{slow_period}"
        self.fast_period = fast_period
        self.slow_period = slow_period
        
        # Le besoin minimum en données est égal à la période la plus longue
        self.min_data_required = slow_period + 1 

    def generate_signal(self, df: pd.DataFrame, metadata: dict) -> TradeSignal:
        """
        Analyse les données pour générer un signal LONG, SHORT ou HOLD.
        """
        symbol = metadata.get('symbol', 'UNKNOWN')
        
        # --- 1. SÉCURITÉ : WARM-UP ---
        # Si on n'a pas assez de bougies pour calculer la SMA lente, on attend.
        if len(df) < self.min_data_required:
            return TradeSignal(action=SignalAction.HOLD, symbol=symbol)

        # --- 2. CALCUL DES INDICATEURS ---
        # On calcule les deux SMA sur la colonne 'close'
        fast_sma = ta.sma(df['close'], length=self.fast_period)
        slow_sma = ta.sma(df['close'], length=self.slow_period)

        # --- 3. RÉCUPÉRATION DES VALEURS (Bougies de clôture) ---
        # On utilise -2 (dernière bougie fermée) et -3 (celle d'avant) 
        # pour confirmer un croisement réel et éviter les faux signaux du prix en direct.
        
        f_current, f_prev = fast_sma.iloc[-2], fast_sma.iloc[-3]
        s_current, s_prev = slow_sma.iloc[-2], slow_sma.iloc[-3]

        # Sécurité supplémentaire si le calcul renvoie NaN
        if pd.isna(f_current) or pd.isna(s_current):
            return TradeSignal(action=SignalAction.HOLD, symbol=symbol)

        # --- 4. LOGIQUE DE CROISEMENT (CROSSOVER) ---
        action = SignalAction.HOLD
        
        # Croisement haussier : La rapide passe au-dessus de la lente
        if f_prev <= s_prev and f_current > s_current:
            action = SignalAction.LONG
            
        # Croisement baissier : La rapide passe en-dessous de la lente
        elif f_prev >= s_prev and f_current < s_current:
            action = SignalAction.SHORT

        # --- 5. RETOUR DU SIGNAL ---
        return TradeSignal(
            action=action,
            symbol=symbol,
            order_type=OrderType.MARKET, # On entre au marché pour valider le croisement
            price=df['close'].iloc[-1],  # Prix actuel pour info
            leverage=1
        )
