import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
import os

path_csv = '../data/benchmark_CPU_GPU.csv'

if(not os.path.exists(path_csv)):
    print(f"Fichier {path_csv} introuvable")
    exit(1)

df = pd.read_csv(path_csv)
df['Gain'] = df['CPU'] / df['GPU']
dims = df['Dimension'].tolist()

fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(10, 6))
fig.suptitle("Benchmark de la multiplication matricielle sur CPU et GPU",fontsize=16,fontweight='bold')
ax1.plot(df['Dimension'], df['CPU'], label='Temps d\'execution moyen avec CPU', color='blue', marker='o')
ax1.plot(df['Dimension'], df['GPU'], label='Temps d\'execution moyen avec GPU', color='red', marker='o')
ax1.set_xscale('log', base=2)
ax1.set_yscale('log')
ax1.set_xticks(dims)
ax1.xaxis.set_major_formatter(ScalarFormatter())
ax1.xaxis.set_minor_formatter(NullFormatter())
ax1.set_title('Temps d\'execution')
ax1.set_xlabel('Dimension n (matrices n x n)')
ax1.set_ylabel('Secondes')
ax1.legend()
ax1.grid(True)

ax2.plot(df['Dimension'], df['Gain'], color='green', marker='o')
ax2.axhline(1, color='black', linestyle='--', linewidth=1)
ax2.set_xscale('log', base=2)
ax2.set_yscale('log')
ax2.set_xticks(dims)
ax2.xaxis.set_major_formatter(ScalarFormatter())
ax2.xaxis.set_minor_formatter(NullFormatter())
ax2.set_title('Combien de fois le GPU est plus rapide')
ax2.set_xlabel('Dimension n (matrices n x n)')
ax2.set_ylabel('Gain = temps CPU / temps GPU')
ax2.grid(True)


bas, haut = ax2.get_ylim()
bas = min(bas, 0.5)
haut = max(haut, 2) * 3
ax2.set_ylim(bas, haut)
ax2.axhspan(bas, 1, color='blue', alpha=0.08)
ax2.axhspan(1, haut, color='green', alpha=0.08)
ax2.text(dims[-1], bas * 1.15, 'CPU plus rapide', color='blue', va='bottom', ha='right')
ax2.text(dims[-1], 1.15, 'GPU plus rapide', color='green', va='bottom', ha='right')

# Un chiffre au-dessus de chaque point
for n, g in zip(dims, df['Gain']):
    if g >= 10:
        texte = f"x{g:.0f}"
    elif g >= 1:
        texte = f"x{g:.1f}"
    else:
        texte = f"CPU x{1 / g:.0f}"
    ax2.annotate(texte, (n, g), textcoords='offset points', xytext=(0, 9), ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()

os.makedirs('../data/plot', exist_ok=True)


plt.savefig('../data/plot/benchmark_CPU_GPU.png')
print("Graphique sauvegardé dans le dossier plot")


