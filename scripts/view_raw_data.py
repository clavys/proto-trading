import pandas as pd
import os
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse

# Ajouter le répertoire parent au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.data.handler import DataHandler

def preview_raw_data(file_path):
    if not os.path.exists(file_path):
        print(f"Erreur : Le fichier {file_path} est introuvable.")
        return

    print(f"--- Lecture des données brutes : {file_path} ---")
    
    # Chargement sans headers (format Binance/Hyperliquid brut)
    df_raw = pd.read_csv(file_path, header=None)

    # Normalisation via ton DataHandler
    df = DataHandler.normalize_binance_klines(df_raw)
    
    total_candles = len(df)
    duration_days = (df['timestamp'].max() - df['timestamp'].min()).days
    print(f"\n[Données chargées] : {total_candles:,} bougies (1 minute)")
    print(f"[Période] : {df['timestamp'].min()} à {df['timestamp'].max()}")
    print(f"[Durée] : {duration_days} jours")
    print(f"[Prix] : Min={df['low'].min():.2f}, Max={df['high'].max():.2f}")
    
    # Créer le graphique interactif avec Plotly
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.75, 0.25],
        subplot_titles=('Prix BTC/USDT (1min)', 'Volume')
    )
    
    # Chandelier (Candlestick)
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='BTCUSDT',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
            increasing_fillcolor='#26a69a',
            decreasing_fillcolor='#ef5350',
            line=dict(width=1),
            increasing_line_width=1,
            decreasing_line_width=1
        ),
        row=1, col=1
    )
    
    # Volume avec couleurs et largeur optimisée
    colors = ['#26a69a' if df['close'].iloc[i] >= df['open'].iloc[i] else '#ef5350' 
              for i in range(len(df))]
    
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume',
            marker=dict(
                color=colors,
                line=dict(width=0)  # Pas de bordure pour coller les barres
            ),
            showlegend=False,
            opacity=0.8
        ),
        row=2, col=1
    )
    
    # Style TradingView Dark Mode
    fig.update_layout(
        title=dict(
            text=f'BTC/USDT (1min) - {os.path.basename(file_path)}',
            font=dict(size=16, color='#D1D4DC')
        ),
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        height=900,
        hovermode='x unified',
        plot_bgcolor='#131722',
        paper_bgcolor='#131722',
        font=dict(color='#D1D4DC', size=11),
        margin=dict(l=50, r=50, t=80, b=50),
        xaxis=dict(
            gridcolor='#2A2E39',
            showgrid=True,
            type='date',
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1h", step="hour", stepmode="backward"),
                    dict(count=4, label="4h", step="hour", stepmode="backward"),
                    dict(count=1, label="1j", step="day", stepmode="backward"),
                    dict(count=7, label="1s", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(step="all", label="Tout")
                ]),
                bgcolor='#1e222d',
                activecolor='#2962FF',
                font=dict(color='#D1D4DC'),
                x=0,
                y=1.05
            ),
            rangeslider=dict(visible=False)
        ),
        xaxis2=dict(
            gridcolor='#2A2E39',
            showgrid=True,
            type='date'
        ),
        yaxis=dict(
            gridcolor='#2A2E39',
            showgrid=True,
            side='right',
            title='Prix (USDT)'
        ),
        yaxis2=dict(
            gridcolor='#2A2E39',
            showgrid=True,
            side='right',
            title='Volume'
        ),
        bargap=0,  # Pas d'espace entre les barres de volume
        bargroupgap=0
    )
    
    # Configuration de l'interactivité
    fig.update_xaxes(
        gridcolor='#2A2E39',
        showgrid=True,
        fixedrange=False,
        showspikes=True,
        spikecolor='#D1D4DC',
        spikethickness=1,
        spikedash='dot'
    )
    
    fig.update_yaxes(
        gridcolor='#2A2E39',
        showgrid=True,
        fixedrange=False,
        showspikes=True,
        spikecolor='#D1D4DC',
        spikethickness=1,
        spikedash='dot'
    )
    
    print("\n[Graphique interactif généré]")
    print("\n💡 NAVIGATION:")
    print("  - Zoom: Sélectionnez une zone avec la souris")
    print("  - Pan: Cliquez et glissez pour déplacer")
    print("  - Boutons: 1h, 4h, 1j, 1s (semaine), 1m (mois), Tout")
    print("  - Reset: Double-clic sur le graphique")
    print("  - Sauvegarde: Cliquez sur l'icône caméra en haut à droite")
    
    # Afficher le graphique dans le navigateur
    fig.show()
    
    print("\n[Graphique ouvert dans le navigateur]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualiser les données de trading avec navigation interactive')
    parser.add_argument('--file', '-f', type=str, default="data/raw/BTCUSDT-1m-2025-12.csv",
                       help='Fichier CSV à visualiser')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("📊 VISUALISEUR INTERACTIF DE DONNÉES DE TRADING (1 MINUTE)")
    print("="*80)
    print(f"Fichier: {args.file}")
    print("Mode: Navigation interactive avec zoom/pan")
    print("="*80)
    
    preview_raw_data(args.file)
    
    print("\n💡 CONSEILS:")
    print("  - Le graphique s'ouvre dans votre navigateur web")
    print("  - Toutes les données sont chargées, zoomez/naviguez librement")
    print("  - Performance optimale grâce au rendu dynamique de Plotly")
