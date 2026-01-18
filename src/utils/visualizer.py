import matplotlib.pyplot as plt
import pandas as pd
from src.core.signal import SignalAction

def plot_backtest_results(df, trades, fast_period, slow_period):
    plt.figure(figsize=(15, 8))
    
    # 1. Tracer le prix de clôture
    plt.plot(df['timestamp'], df['close'], label='Prix BTC', color='black', alpha=0.3)
    
    # 2. Recalculer et tracer les SMA
    fast_sma = df['close'].rolling(window=fast_period).mean()
    slow_sma = df['close'].rolling(window=slow_period).mean()
    plt.plot(df['timestamp'], fast_sma, label=f'SMA Rapide ({fast_period})', color='blue', lw=1)
    plt.plot(df['timestamp'], slow_sma, label=f'SMA Lente ({slow_period})', color='orange', lw=1.5)

    # 3. Marquer les trades et afficher les bénéfices
    for trade in trades:
        # Correction : On compare avec l'Enum SignalAction
        is_long = trade['type'] == SignalAction.LONG
        color = 'green' if is_long else 'red'
        marker = '^' if is_long else 'v'
        
        # Décalage pour la visibilité des flèches
        offset = trade['entry_price'] * 0.001
        entry_y = trade['entry_price'] - offset if is_long else trade['entry_price'] + offset

        # Dessiner l'entrée
        plt.scatter(trade['entry_time'], entry_y, 
                    color=color, marker=marker, s=100, edgecolors='black', zorder=5)
        
        # Dessiner la sortie et le texte du bénéfice
        if trade.get('exit_time'):
            plt.scatter(trade['exit_time'], trade['exit_price'], 
                        color='black', marker='x', s=60, zorder=6)
            
            # Affichage du bénéfice (PnL Cash) à côté de la sortie
            pnl = trade['pnl_cash']
            pnl_text = f"{pnl:+.2f} USDT"
            text_color = 'green' if pnl > 0 else 'red'
            
            # On place le texte légèrement au-dessus du point de sortie
            plt.text(trade['exit_time'], trade['exit_price'] + offset, 
                     pnl_text, fontsize=9, color=text_color, fontweight='bold',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.title(f"Backtest SMA {fast_period}/{slow_period} - {len(trades)} trades")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()