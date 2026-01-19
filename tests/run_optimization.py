# tests/run_optimization.py
import pandas as pd
from src.core.data import DataHandler
from src.strategies.sma_crossover import SMACrossStrategy
from src.optimization.grid_search import GridSearch

def run_optimization():
    # 1. Préparation des données
    path = "data/raw/BTCUSDT-1m-2026-01-14.csv"
    raw_data = pd.read_csv(path, header=None)
    data = DataHandler.normalize_binance_klines(raw_data)

    # 2. Définition de la grille de paramètres (Modulable à souhait)
    # Ici vous contrôlez la finesse de la recherche
    param_grid = {
        "fast_period": [5, 10, 15, 20, 30],
        "slow_period": [20, 40, 60, 80, 100, 150, 200],
        "min_delta_pct": [0.0, 0.0005, 0.001],
        "cooldown": [5, 15]
    }

    # 3. Lancement de la recherche (mode Performance activé automatiquement)
    optimizer = GridSearch(SMACrossStrategy, data, fee=0.00035)
    best_configs = optimizer.optimize(param_grid)

    # 4. Affichage des 10 meilleures configurations
    print("\n" + "="*120)
    print("--- TOP 10 MEILLEURES CONFIGURATIONS ---")
    print("="*120)
    
    # Sélection des colonnes pertinentes pour l'affichage
    display_cols = ["fast_period", "slow_period", "min_delta_pct", "cooldown", 
                    "final_balance", "pnl_cash", "roi_pct", "num_trades", 
                    "win_rate_pct", "profit_factor", "max_drawdown_pct"]
    
    print(best_configs[display_cols].head(10).to_string(index=False))

    # 5. Détails de la meilleure configuration
    print("\n" + "="*120)
    print("--- MEILLEURE CONFIGURATION ---")
    print("="*120)
    best = best_configs.iloc[0]
    print(f"Paramètres:")
    print(f"  - Fast Period: {int(best['fast_period'])}")
    print(f"  - Slow Period: {int(best['slow_period'])}")
    print(f"  - Min Delta %: {best['min_delta_pct']}")
    print(f"  - Cooldown: {int(best['cooldown'])}")
    print(f"\nPerformances:")
    print(f"  - Profit: ${best['pnl_cash']:.2f}")
    print(f"  - ROI: {best['roi_pct']:.2f}%")
    print(f"  - Trades: {int(best['num_trades'])}")
    print(f"  - Win Rate: {best['win_rate_pct']:.2f}%")
    print(f"  - Profit Factor: {best['profit_factor']:.2f}")
    print(f"  - Max Drawdown: {best['max_drawdown_pct']:.2f}%")
    print()

    # 6. Sauvegarde optionnelle
    best_configs.to_csv("optimization_results.csv", index=False)
    print(f"✅ Résultats complets sauvegardés dans 'optimization_results.csv' ({len(best_configs)} configurations)")

if __name__ == "__main__":
    run_optimization()