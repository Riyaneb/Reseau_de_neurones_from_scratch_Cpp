import pandas as pd
import matplotlib.pyplot as plt
import os

path_csv = '../data/training_gpt_log.csv'

if(not os.path.exists(path_csv)):
    print(f"Fichier {path_csv} introuvable")
    exit(1)

df = pd.read_csv(path_csv)
df_fin = df[df['tour'] >= 5000]

fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(10, 9))

fig.suptitle("Évolution de l'entrainement du LLM (25,6 M de paramètres)",fontsize=16,fontweight='bold')

ax1.plot(df['tour'], df['perte_train'], label='Perte d\'entrainement (moyenne sur 1000 tours)', color='orange')
ax1.plot(df['tour'], df['perte_val'], label='Perte de validation (50 lots fixes)', color='blue')
ax1.set_xlabel('Tour')
ax1.set_ylabel('Perte')
ax1.set_title('Évolution de la perte sur tout l\'entrainement')
ax1.legend()
ax1.grid(True)

ax2.plot(df_fin['tour'], df_fin['perte_train'], label='Perte d\'entrainement (moyenne sur 1000 tours)', color='orange')
ax2.plot(df_fin['tour'], df_fin['perte_val'], label='Perte de validation (50 lots fixes)', color='blue')
ax2.set_xlabel('Tour')
ax2.set_ylabel('Perte')
ax2.set_title('Zoom à partir du tour 5000 : écart entre entrainement et validation')
ax2.legend()
ax2.grid(True)

plt.tight_layout()

plt.savefig('../data/plot/training_gpt_log.png')
print("Graphique sauvegardé dans le dossier plot")