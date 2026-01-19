# src/optimization/grid_search.py
import itertools
import pandas as pd
from typing import Type, Dict, List
from src.utils.backtest import Backtester

class GridSearch:
    def __init__(self, strategy_class: Type, dataset: pd.DataFrame, initial_balance=1000, fee=0.00035):
        self.strategy_class = strategy_class
        self.dataset = dataset
        self.initial_balance = initial_balance
        self.fee = fee

    def optimize(self, param_grid: Dict[str, List]) -> pd.DataFrame:
        """
        Teste toutes les combinaisons de paramètres fournies en mode léger (detailed=False).
        Retourne les résultats triés par profit.
        """
        # Création de toutes les combinaisons possibles
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        results = []
        total = len(combinations)
        print(f"Démarrage de l'optimisation : {total} configurations à tester.")
        print(f"Mode Performance activé (detailed=False)\n")

        for i, params in enumerate(combinations):
            # Filtre logique pour SMA : la rapide doit être plus petite que la lente
            if "fast_period" in params and "slow_period" in params:
                if params["fast_period"] >= params["slow_period"]:
                    continue

            # 1. Instanciation dynamique de la stratégie
            strategy = self.strategy_class(**params)
            
            # 2. Backtest en mode "light" (detailed=False) pour performance
            bt = Backtester(strategy, initial_balance=self.initial_balance, fee=self.fee)
            summary = bt.run(self.dataset, detailed=False)

            # 3. Extraction des métriques clés
            result_entry = {
                **params,
                "final_balance": round(summary["final_balance"], 2),
                "pnl_cash": round(summary["total_pnl_cash"], 2),
                "roi_pct": round(summary["roi_pct"], 2),
                "num_trades": summary["num_trades"],
                "win_rate_pct": round(summary["win_rate_pct"], 2),
                "avg_win": round(summary["avg_win"], 2),
                "avg_loss": round(summary["avg_loss"], 2),
                "profit_factor": round(summary["profit_factor"], 2),
                "max_drawdown_pct": round(summary["max_drawdown_pct"], 2)
            }
            results.append(result_entry)

            if (i + 1) % 10 == 0:
                print(f"Progression : {i+1}/{total} configurations testées")

        # Retourne un tableau trié par profit
        df_results = pd.DataFrame(results).sort_values(by="pnl_cash", ascending=False)
        print(f"\n✅ Optimisation terminée ! {len(df_results)} configurations valides trouvées.")
        return df_results
    
    def get_best_config(self, results: pd.DataFrame, metric: str = "pnl_cash") -> dict:
        """
        Retourne la meilleure configuration en fonction d'une métrique.
        :param results: DataFrame des résultats de l'optimisation
        :param metric: Métrique à optimiser (pnl_cash, roi_pct, profit_factor, etc.)
        """
        best_row = results.iloc[0]  # Déjà trié par défaut
        return best_row.to_dict()