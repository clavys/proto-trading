import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

class GridSearchAnalyzer:
    # Liste des métriques de performance standard produites par le Backtester
    KNOWN_METRICS = [
        'final_balance', 'pnl_cash', 'roi_pct', 'num_trades', 
        'win_rate_pct', 'avg_win', 'avg_loss', 'profit_factor', 
        'max_drawdown_pct'
    ]

    def __init__(self, csv_path='optimization_results.csv', target_metric='roi_pct'):
        self.csv_path = csv_path
        self.target_metric = target_metric
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Le fichier {csv_path} est introuvable.")
        
        self.df = pd.read_csv(csv_path)
        # Détection automatique des paramètres : colonnes qui ne sont pas des métriques
        self.params = [col for col in self.df.columns if col not in self.KNOWN_METRICS]
        print(f"Analyseur initialisé. Paramètres détectés : {self.params}")
        print(f"Métrique cible : {target_metric}\n")

    def plot_parameter_importance(self):
        """Affiche l'impact RELATIF de chaque paramètre (influence sur la métrique cible)."""
        print("\n📊 Calcul de l'importance de chaque paramètre...")
        
        importance = []
        for param in self.params:
            grouped = self.df.groupby(param)[self.target_metric].mean()
            spread = grouped.max() - grouped.min()  # Impact du paramètre
            importance.append({'param': param, 'impact': spread})
        
        df_importance = pd.DataFrame(importance).sort_values('impact', ascending=True)
        
        plt.figure(figsize=(10, 6))
        colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in df_importance['impact']]
        plt.barh(df_importance['param'], df_importance['impact'], color=colors)
        plt.xlabel(f"Impact sur {self.target_metric} (%)")
        plt.title("Importance Relative des Paramètres")
        plt.grid(axis='x', alpha=0.3)
        
        for i, v in enumerate(df_importance['impact']):
            plt.text(v, i, f" {v:.2f}%", va='center')
        
        plt.tight_layout()
        plt.savefig("analysis_parameter_importance.png")
        print("✓ Graphique d'importance des paramètres sauvegardé.")

    def plot_parameter_distribution(self):
        """Visualise chaque paramètre: valeurs testées, tendance, meilleure valeur en évidence."""
        n_params = len(self.params)
        fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 5), squeeze=False)

        for i, param in enumerate(self.params):
            values = sorted(self.df[param].unique())
            metrics = [self.df[self.df[param] == v][self.target_metric].mean() for v in values]
            
            # Trouver le meilleur
            best_idx = metrics.index(max(metrics))
            best_value = values[best_idx]
            
            # Courbe principale
            axes[0, i].plot(values, metrics, 'o-', linewidth=2, markersize=8, color='steelblue', label='Mean')
            
            # Mettre en évidence le meilleur
            axes[0, i].plot(best_value, metrics[best_idx], 'o', markersize=15, color='#2ecc71', label='Best', zorder=5)
            
            # Lissage pour voir la tendance
            if len(values) > 2:
                z = np.polyfit(range(len(values)), metrics, 2)
                p = np.poly1d(z)
                x_smooth = np.linspace(0, len(values)-1, 100)
                axes[0, i].plot(x_smooth, p(x_smooth), '--', alpha=0.5, color='gray', label='Trend')
            
            axes[0, i].set_title(f"{param}\n(Best: {best_value})")
            axes[0, i].set_xlabel(param)
            axes[0, i].set_ylabel(f"{self.target_metric}")
            axes[0, i].grid(True, alpha=0.3)
            axes[0, i].legend(loc='best')
        
        plt.tight_layout()
        plt.savefig("analysis_parameter_distribution.png")
        print("✓ Graphique de distribution des paramètres sauvegardé.")

    def plot_heatmap(self, param_x=None, param_y=None):
        """Génère une heatmap pour un duo de paramètres."""
        if len(self.params) < 2:
            print("⚠ Pas assez de paramètres pour une heatmap.")
            return

        # Si non spécifiés, on prend les deux premiers
        px = param_x if param_x else self.params[0]
        py = param_y if param_y else self.params[1]

        plt.figure(figsize=(10, 8))
        pivot = self.df.pivot_table(index=px, columns=py, values=self.target_metric, aggfunc='mean')
        sns.heatmap(pivot, annot=True, cmap='RdYlGn', center=0, fmt=".2f")
        plt.title(f"Heatmap {self.target_metric} : {px} vs {py}")
        plt.tight_layout()
        plt.savefig(f"analysis_heatmap_{px}_{py}.png")
        print(f"✓ Heatmap {px} vs {py} sauvegardée.")

    def plot_metric_ranges(self):
        """Visualise les ranges de valeurs testées pour chaque paramètre."""
        n_params = len(self.params)
        fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 5), squeeze=False)
        
        for i, param in enumerate(self.params):
            values = sorted(self.df[param].unique())
            metrics = [self.df[self.df[param] == v][self.target_metric].mean() for v in values]
            
            axes[0, i].plot(values, metrics, 'o-', linewidth=2, markersize=8, color='steelblue')
            axes[0, i].set_title(f"Performance vs {param}")
            axes[0, i].set_xlabel(param)
            axes[0, i].set_ylabel(f"Moyenne {self.target_metric}")
            axes[0, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("analysis_metric_ranges.png")
        print("✓ Graphique des ranges sauvegardé.")

    def get_parameter_insights(self):
        """Analyse chaque paramètre pour trouver les valeurs/ranges optimaux."""
        print("\n" + "="*80)
        print("📊 INSIGHTS PAR PARAMÈTRE")
        print("="*80)
        
        for param in self.params:
            grouped = self.df.groupby(param)[self.target_metric].agg(['mean', 'std', 'count'])
            grouped = grouped.sort_values('mean', ascending=False)
            
            best_value = grouped['mean'].idxmax()
            worst_value = grouped['mean'].idxmin()
            
            print(f"\n📌 {param}:")
            print(f"   ✓ Meilleure valeur: {best_value} (moy: {grouped.loc[best_value, 'mean']:.2f}%, tests: {int(grouped.loc[best_value, 'count'])})")
            print(f"   ✗ Pire valeur: {worst_value} (moy: {grouped.loc[worst_value, 'mean']:.2f}%)")
            print(f"   📈 Spread: {grouped['mean'].max() - grouped['mean'].min():.2f}%")
            print(f"   Valeurs testées: {sorted(self.df[param].unique())}")

    def get_optimization_recommendations(self):
        """Recommande où affiner les recherches pour le prochain grid search."""
        print("\n" + "="*80)
        print("🎯 RECOMMANDATIONS POUR LE PROCHAIN GRID SEARCH")
        print("="*80)
        
        recommendations = {}
        
        for param in self.params:
            grouped = self.df.groupby(param)[self.target_metric].mean()
            best_value = grouped.idxmax()
            values = sorted(self.df[param].unique())
            
            # Skip si seulement 1 valeur testée
            if len(values) < 2:
                print(f"\n⚠️ {param}: Seulement 1 valeur testée ({best_value}). Augmentez la plage pour la prochaine optimisation.")
                recommendations[param] = [best_value]
                continue
            
            # Déterminer le range optimal
            best_idx = values.index(best_value)
            step = values[1] - values[0]
            lower_bound = values[best_idx - 1] if best_idx > 0 else best_value - step
            upper_bound = values[best_idx + 1] if best_idx < len(values) - 1 else best_value + step
            
            # Créer une grille plus fine autour du meilleur
            if isinstance(best_value, (int, np.integer)):
                new_grid = list(range(int(lower_bound), int(upper_bound) + 1))
            else:
                step = (values[1] - values[0]) if len(values) > 1 else best_value / 10
                new_grid = list(np.arange(lower_bound, upper_bound + step, step / 2))
            
            recommendations[param] = new_grid
            print(f"\n🔧 {param}:")
            print(f"   Ancien range: {values}")
            print(f"   Recommandé:   {[round(v, 3) for v in new_grid[:10]]}{'...' if len(new_grid) > 10 else ''}")
        
        return recommendations

    def export_recommendations_to_config(self):
        """Exporte les recommandations comme code Python prêt à copier/coller."""
        recommendations = self.get_optimization_recommendations()
        
        config_code = "# Recommendation generated by GridSearchAnalyzer\n"
        config_code += "# Use for the next grid search\n\n"
        config_code += "param_grid = {\n"
        
        for param, grid in recommendations.items():
            # Limiter à 15 valeurs max pour pas surcharger
            if len(grid) > 15:
                step = len(grid) // 15
                grid = grid[::step][:15]
            
            grid_str = str([round(v, 3) if isinstance(v, float) else int(v) for v in grid])
            config_code += f"    \"{param}\": {grid_str},\n"
        
        config_code += "}\n"
        
        with open("recommended_param_grid.py", "w", encoding='utf-8') as f:
            f.write(config_code)
        
        print(f"\n✓ Configuration recommandée exportée dans 'recommended_param_grid.py'")
        print(f"   → Copier/coller dans run_optimization.py pour le prochain test\n")

    def run_all_analysis(self):
        """Exécute la suite complète d'analyses."""
        print("\n" + "="*80)
        print("🔍 LANCEMENT DE L'ANALYSE COMPLÈTE")
        print("="*80 + "\n")
        
        self.plot_parameter_importance()
        self.plot_parameter_distribution()
        self.plot_metric_ranges()
        
        if len(self.params) >= 2:
            self.plot_heatmap()
        
        # Analyses intelligentes
        self.get_parameter_insights()
        self.export_recommendations_to_config()
        
        print("\n" + "="*80)
        print("✅ Analyse terminée ! Fichiers générés :")
        print("  - analysis_parameter_importance.png")
        print("  - analysis_parameter_distribution.png")
        print("  - analysis_metric_ranges.png")
        if len(self.params) >= 2:
            print("  - analysis_heatmap_*.png")
        print("  - recommended_param_grid.py")
        print("="*80 + "\n")

if __name__ == "__main__":
    # Permet de lancer l'analyse seul depuis le terminal
    analyzer = GridSearchAnalyzer()
    analyzer.run_all_analysis()
