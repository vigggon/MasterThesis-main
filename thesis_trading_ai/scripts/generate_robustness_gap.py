import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Create plots directory if it doesn't exist
os.makedirs('results/plots', exist_ok=True)

# Data from THESIS_RESULTS.md and metrics
data = {
    'Model': ['Transformer (Backtest)', 'BiLSTM (Backtest)', 'Transformer (Live)', 'BiLSTM (Live)'],
    'Environment': ['Backtest', 'Backtest', 'Live', 'Live'],
    'Architecture': ['Transformer', 'BiLSTM', 'Transformer', 'BiLSTM'],
    'Total_Trades': [161, 152, 31, 19],
    'Net_Return_Pct': [23.1, 13.1, -1.07, 0.10]
}

df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))
colors = {'Transformer': '#2E86C1', 'BiLSTM': '#E67E22'}
markers = {'Backtest': 'o', 'Live': 'X'}

for i, row in df.iterrows():
    plt.scatter(row['Total_Trades'], row['Net_Return_Pct'], 
                color=colors[row['Architecture']], 
                marker=markers[row['Environment']], 
                s=200, label=row['Model'], alpha=0.8, edgecolors='black')

# Annotations
for i, txt in enumerate(df['Model']):
    plt.annotate(txt, (df['Total_Trades'][i]+2, df['Net_Return_Pct'][i]), fontsize=10)

plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.xlabel('Total Trade Count (Sample Size)', fontsize=12)
plt.ylabel('Net Return (%)', fontsize=12)
plt.title('Trade Frequency vs. Net Return: The Robustness Gap', fontsize=14)
plt.grid(True, alpha=0.3)

# Arrows showing the shift from backtest to live
plt.annotate('', xy=(31, -1.07), xytext=(161, 23.1),
             arrowprops=dict(arrowstyle="->", color=colors['Transformer'], lw=2, alpha=0.4, linestyle=':'))
plt.annotate('', xy=(19, 0.10), xytext=(152, 13.1),
             arrowprops=dict(arrowstyle="->", color=colors['BiLSTM'], lw=2, alpha=0.4, linestyle=':'))

plt.tight_layout()
plt.savefig('results/plots/robustness_gap_analysis.pdf')
print("Plot saved to results/plots/robustness_gap_analysis.pdf")
