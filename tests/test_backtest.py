# tests/test_backtest.py
import pandas as pd
from src.core.data import DataHandler
from src.strategies.sma_crossover import SMACrossStrategyReverse
from src.strategies.sma_crossover_opti import SMACrossEnhanced
from src.utils.backtest import Backtester
from src.utils.visualizer import plot_backtest_results

def run_simulation():
    # 1. Charger le fichier CSV (Binance n'a pas de titres de colonnes, donc header=None)
    path = "data/raw/BTCUSDT-1m-2025-11.csv"
    #path = "data/raw/BTCUSDT-1m-2026-01-14.csv"
    raw_data = pd.read_csv(path, header=None)
    
    # 2. Transformer les données au format standard
    data = DataHandler.normalize_binance_klines(raw_data)
    
    # 3. Choisir la stratégie (ex: SMA 9 et 21)
    # Note : avec seulement 1440 lignes, évite des SMA trop longues (ex: 200)
    strategy = SMACrossStrategyReverse(fast_period=24, slow_period=88, stop_loss_pct=0.0, cooldown=16, verbose=False)
    '''
    strategy = SMACrossEnhanced(
        fast_period=30, 
        slow_period=90, 
        trend_period=200, 
        atr_period=14, 
        tp_mult=2.0, 
        sl_mult=1.5, 
        use_trend_filter=True, 
        cooldown=20
    )
    '''
    
    # 4. Lancer le backtester
    backtester = Backtester(strategy=strategy, initial_balance=1000, fee=0.0001)
    results = backtester.run(data, metadata={"symbol": "BTCUSDT"})
    
    # 5. Afficher les résultats
    print(f"--- Rapport de Simulation ---")
    print(f"Période : {data['timestamp'].min()} à {data['timestamp'].max()}")
    print(f"Solde final : {results['final_balance']:.2f} USDT")
    print(f"Nombre de trades : {results['num_trades']}")
    print(f"Win Rate : {results['win_rate_pct']:.2f}%")

    # 6. Visualiser les résultats

    plot_backtest_results(
        data, 
        results['trades'], 
        strategy.fast_period, 
        strategy.slow_period
    )


if __name__ == "__main__":
    run_simulation()

