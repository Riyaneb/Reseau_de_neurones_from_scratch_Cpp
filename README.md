# Réseau de neurones & Agent DQN / CNN from scratch en C++

Un réseau de neurones, un agent qui apprend à jouer à Snake, un réseau CNN ainsi qu'un LLM codés en C++17, sans utiliser aucune bibliothèque externe (PyTorch, TensorFlow).

Le projet part du calcul le plus simple et va jusqu'à un agent qui apprend à jouer à Snake, à reconnaitre des images et à écrire une histoire, tout seul, en passant par toutes les étapes intermédiaires. Chaque étape est vérifiée par des tests automatiques : plus de **711 tests** au total.

Projet personnel fait pour comprendre comment fonctionne un réseau de neurones, plutôt que d'utiliser un outil qui fait tout à ma place.

---

## Sommaire

- [Comment ça marche : les maths derrière](#comment-ça-marche--les-maths-derrière)
  - [Une couche de neurones](#une-couche-de-neurones)
  - [Les fonctions d'activation](#les-fonctions-dactivation)
  - [Fonction de perte](#fonction-de-perte)
  - [Corriger les poids : la descente de gradient](#corriger-les-poids--la-descente-de-gradient)
  - [La rétropropagation](#la-rétropropagation)
  - [Optimiseur Adam](#optimiseur-adam)
  - [Apprendre à jouer : l'équation de Bellman](#apprendre-à-jouer--léquation-de-bellman)
  - [La convolution](#la-convolution)
  - [Les filtres et l'astuce im2col](#les-filtres-et-lastuce-im2col)
  - [Le max pooling](#le-max-pooling)
  - [Le texte en nombres](#le-texte-en-nombres)
  - [Comment entrainer le LLM](#comment-entrainer-le-llm)
  - [L'embedding](#lembedding)
  - [Couche de normalisation](#couche-de-normalisation)
  - [La couche d'attention](#la-couche-dattention)
  - [Les connexions résiduelles](#les-connexions-résiduelles)
  - [Un Transformer](#un-transformer)
  - [Génération du texte](#génération-du-texte)
- [Architecture](#architecture)
- [Agent DQN (Snake)](#agent-dqn-snake)
  - [Résultats](#résultats)
  - [Démonstration](#démonstration)
- [Réseau convolutifs (CNN)](#réseau-convolutifs-cnn)
  - [Chiffres manuscrits (MNIST)](#chiffres-manuscrits-mnist)
  - [Radiographies pulmonaires (pneumonie)](#radiographies-pulmonaires-pneumonie)
  - [Les données et leur chargement](#les-données-et-leur-chargement)
  - [Entraîner, tester, visualiser](#entraîner-tester-visualiser)
  - [Interface graphique](#interface-graphique)
  - [Limites et pistes du CNN](#limites-et-pistes-du-cnn)
- [Petit LLM from scratch](#petit-llm-from-scratch)
  - [Résultats du LLM](#résultats-du-llm)
  - [Les données : LeCarnet](#les-données--lecarnet)
  - [Le modèle](#le-modèle)
  - [Les nouvelles classes](#les-nouvelles-classes)
  - [Entraînement](#entraînement)
  - [Limites et pistes du LLM](#limites-et-pistes-du-llm)
- [Compilation et exécution](#compilation-et-exécution)
- [Choix de conception](#choix-de-conception)
- [Tests](#tests)
- [Limites connues](#limites-connues)
- [Pistes d'amélioration](#pistes-damélioration)
- [Structure du dépôt](#structure-du-dépôt)

---

## Comment ça marche : les maths derrière

Cette section explique l'aspect mathématique des réseaux de neurones.

### Une couche de neurones

Un réseau de neurones est fait de couches. Chaque couche prend des nombres en entrée, les combine, et produit d'autres nombres en sortie.

![Schéma d'un réseau de neurones simple](docs/images/Shema_reseau_neuronal_simple.png)

Concrètement, chaque neurone de sortie fait une somme pondérée de toutes les entrées, puis ajoute une constante appelée **biais**. Si on a $i$ entrées et $j$ sorties :

```math
y_1 = x_1 w_{11} + x_2 w_{21} + \dots + x_i w_{i1} + b_1
```

```math
y_2 = x_1 w_{12} + x_2 w_{22} + \dots + x_i w_{i2} + b_2
```

```math
y_j = x_1 w_{1j} + x_2 w_{2j} + \dots + x_i w_{ij} + b_j
```

On les regroupe donc sous forme matricielle, ce qui donne une seule opération :

```math
Y = XW + b
```

Où :
- $X$ est la matrice d'entrée, de taille $N \times i$ (chaque ligne est un exemple),
- $W$ est la matrice des poids, de taille $i \times j$,
- $b$ est le vecteur de biais, de taille $1 \times j$, ajouté à chaque ligne,
- $Y$ est la sortie, de taille $N \times j$.

C'est d'ailleurs ce que fait la méthode `DenseLayer::forward` dans le code.

### Les fonctions d'activation

Si on empile deux couches l'une derrière l'autre sans rien entre les deux, on obtient :

```math
H = X W_1 + b_1
```

```math
Y = H W_2 + b_2 = (X W_1 + b_1) W_2 + b_2 = X (W_1 W_2) + (b_1 W_2 + b_2)
```

En posant $W' = W_1 W_2$ et $b' = b_1 W_2 + b_2$, on retombe sur $Y = X W' + b'$, c'est-à-dire une seule couche. Empiler des couches ne sert donc à rien tant qu'on ne casse pas cette linéarité. C'est le rôle des fonctions d'activation. On en glisse une entre chaque couche, et elle est appliquée case par case.

![Courbes des fonctions d'activation](docs/images/courbe_fonction_activation.png)

Les quatre fonctions implémentées dans ce projet :

**ReLU** : Elle garde les valeurs positives et met tout le reste à zéro.

```math
\mathrm{ReLU}(x) = \max(0, x)
```

Sa dérivée vaut 1 si $x > 0$ et 0 sinon.

**GeLU** : Une variante de la fonction ReLU. Au lieu de couper net à zéro, elle adoucit la courbe de sorte à laisser passer quelques valeurs négatives. Sa courbe adoucie permet de rendre l'apprentissage plus régulier sur les réseaux très profonds comme les LLM.

```math
\mathrm{GELU}(x) = \frac{x}{2} \left( 1 + \mathrm{erf}\left( \frac{x}{\sqrt{2}} \right) \right)
```

```math
\mathrm{GELU}'(x) = \frac{1}{2} \left( 1 + \mathrm{erf}\left( \frac{x}{\sqrt{2}} \right) \right) + \frac{x}{\sqrt{2\pi}} \, e^{-x^2/2}
```


**Sigmoïde** : Elle écrase toute valeur entre 0 et 1. A utiliser quand la sortie attendue est une probabilité ou un oui/non.

```math
\sigma(x) = \frac{1}{1 + e^{-x}}
```

```math
\sigma'(x) = \sigma(x) \, (1 - \sigma(x))
```

**Tanh** : même forme que la sigmoïde mais centrée sur zéro (sortie entre -1 et 1).

```math
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
```

```math
\tanh'(x) = 1 - \tanh^2(x)
```

**Softmax** : Semblable à la fonction sigmoïde où chaque valeur est comprise entre 0 et 1 et la somme de toutes les valeurs est égale à 1. Chaque sortie dépend des autres.

```math
s_i = \frac{e^{z_i}}{\sum_{j=1}^{k} e^{z_j}}
```

```math
\frac{\partial s_i}{\partial z_j} = s_i \, (\delta_{ij} - s_j)
```
où $\delta_{ij}$ vaut 1 si $i = j$, et 0 sinon.

**Identité** : Ne change rien à la valeur. Sa dérivée vaut toujours 1. Elle sert quand la valeur attendue n'est pas bornée, ce qui est le cas de l'agent DQN réalisé dans ce projet.

### Fonction de perte

Pour que le réseau apprenne, il faut qu'il sache à quel point il se trompe. C'est le rôle de la fonction de perte. Dans ce projet, j'utilise l'erreur quadratique : on prend l'écart entre la prédiction et la bonne réponse, on le met au carré, et on fait la moyenne.

```math
L = \frac{1}{2N} \sum_{n=1}^{N} \sum_{k} ( \hat{y}_{nk} - y_{nk} )^2
```

Où $\hat{y}$ est la prédiction du réseau, $y$ la bonne réponse, et $N$ le nombre d'exemples.

Le facteur $\frac{1}{2}$ est là pour simplifier la dérivée. Le carré fait apparaître un facteur 2 en dérivant, et les deux s'annulent. Ainsi le gradient s'écrit :

```math
\frac{\partial L}{\partial \hat{Y}} = \frac{\hat{Y} - Y}{N}
```

Voici les autres fonctions de perte disponible dans le projet :

**BCE Loss (Binary Cross-Entropy)** : Elle s'utilise lorsque le modèle doit choisir entre 2 choix. Elle est souvent utilisée avec la fonction d'activation Sigmoïde.

```math
L = -\frac{1}{N} \sum_{n=1}^{N} \sum_{k} \left[ y_{nk} \ln(p_{nk}) + (1 - y_{nk}) \ln(1 - p_{nk}) \right]
```

```math
\frac{\partial L}{\partial p_{nk}} = \frac{1}{N} \cdot \frac{p_{nk} - y_{nk}}{p_{nk} \, (1 - p_{nk})}
```


**CCE Loss (Categorical Cross-Entropy)** : Elle s'utilise quand il y a 3 classes ou plus (Par exemple pour prédire si une image est un chat, un chien ou un oiseau).


```math
L = -\frac{1}{N} \sum_{n=1}^{N} \sum_{k} y_{nk} \ln(p_{nk})
```

```math
\frac{\partial L}{\partial p_{nk}} = -\frac{1}{N} \cdot \frac{y_{nk}}{p_{nk}}
```

### Corriger les poids : la descente de gradient

Maintenant qu'on sait mesurer l'erreur, on cherche maintenant à la diminuer grâce à la descente de gradient.

![Illustration de la descente de gradient](docs/images/gradient_illustration.png)

La pente en un point donné, c'est le **gradient** qui correspond la dérivée de l'erreur par rapport à chaque poids. Il indique dans quelle direction l'erreur *augmente* le plus vite. On va donc dans le sens inverse :

```math
w \leftarrow w - \eta \cdot \frac{\partial L}{\partial w}
```

Où $\eta$ est le **taux d'apprentissage**, c'est-à-dire la taille du pas. Trop grand, on saute par-dessus le minimum et ça diverge. Trop petit, ça prend beaucoup de temps. C'est le réglage le plus important à ajuster dans un réseau de neurones.

Le signe moins fait qu'on descend au lieu de monter.

### La rétropropagation

On sait calculer l'erreur à la sortie du réseau, mais comment savoir de combien chaque poids de la première couche est responsable de cette erreur, alors qu'il est séparé de la sortie par plusieurs autres couches ?

On applique la règle de dérivation en chaîne. La dérivée d'une composition de fonctions est le produit des dérivées de chaque étape. On part donc de l'erreur à la sortie et on la fait remonter couche par couche, en la multipliant à chaque fois par la dérivée locale de la couche traversée.

![Propagation avant et rétropropagation](docs/images/Retropropagation_illustration.png)

Pour une couche dense, il faut calculer **trois** dérivées. En notant $\frac{\partial L}{\partial Y}$ le gradient qui arrive depuis la couche suivante :

**Le gradient des poids** : De combien faut-il corriger chaque poids :

```math
\frac{\partial L}{\partial W} = X^{T} \cdot \frac{\partial L}{\partial Y}
```

**Le gradient du biais** : On additionne les contributions de tous les exemples du groupe :

```math
\frac{\partial L}{\partial b} = \sum_{n=1}^{N} \frac{\partial L}{\partial Y_{n}}
```

**Le gradient de l'entrée** : C'est lui qu'on transmet à la couche précédente, pour que la chaîne continue :

```math
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \cdot W^{T}
```

Pour une couche d'activation, il suffit de multiplier case par case le gradient reçu par la dérivée de la fonction, évaluée sur l'entrée mémorisée :

```math
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \odot f'(X)
```

Le symbole $\odot$ désigne la multiplication case par case (produit de Hadamard).

```math
\frac{\partial L}{\partial w} \approx \frac{L(w + \varepsilon) - L(w - \varepsilon)}{2\varepsilon}
```

### Optimiseur Adam

La descente de gradient simple avance toujours du même pas, dans la direction indiquée par le gradient du moment. Ça fonctionne, mais ça oscille beaucoup et ça peut être lent.

L'optimiseur Adam permet d'améliorer ça. D'abord il garde une **moyenne des gradients passés**, ce qui lisse les oscillations. Ensuite il garde une moyenne des carrés des gradients, et s'en sert pour adapter la taille du pas à chaque poids individuellement.

Moyenne des gradients passés (le premier moment) :

```math
m_t = \beta_1 \, m_{t-1} + (1 - \beta_1) \, g_t
```

Moyenne des carrés des gradients (le second moment) :

```math
v_t = \beta_2 \, v_{t-1} + (1 - \beta_2) \, g_t^2
```

Correction du démarrage :

```math
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
```

```math
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
```

Mise à jour du poids :

```math
w \leftarrow w - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
```

Avec les valeurs usuelles $\beta_1 = 0.9$, $\beta_2 = 0.999$ et $\epsilon = 10^{-8}$. $g_t$ est le gradient à l'étape $t$, et $t$ compte le nombre de mises à jour effectuées depuis le début.

Il peut être utile de faire varier le taux d'apprentissage plutôt que de garder le même durant tout l'entrainement.

***Dans un premier temps***, on le fait varier de manière linéaire. Au départ, les poids d'un réseau sont aléatoires et les moyenne d'Adam ne reposent encore que sur quelques gradients. On evite d'utiliser des pas qui pourrait être trop grand. Pour se faire, on fait monter le taux d'apprentissage de 0 à la valeur maximale que peut prendre le taux pendant les K premiers pas.

***Dans un second temps***, on veut de grands pas pour avancer vite, puis de petits pas à la fin pour affiner sans sauter par-dessus le minimum. On le fait descendre du taux max au taux min avec la fonction cosinus.

Au pas $s$, sur un entraînement de $N$ pas :

```math
\eta_s = \begin{cases} \eta_{max} \, \dfrac{s+1}{K} & \text{si } s < K \\[2mm] \eta_{min} + \dfrac{1}{2} (\eta_{max} - \eta_{min}) \left( 1 + \cos\left( \pi \, \dfrac{s - K}{N - K} \right) \right) & \text{sinon} \end{cases}
```


### Apprendre à jouer : l'équation de Bellman

Tout ce qui précède suppose qu'on connaît la bonne réponse pour chaque exemple. Pour XOR, c'est le cas, on sait que $0 \oplus 1 = 1$.

Mais pour Snake, personne ne dit à l'agent quelle était la bonne direction. Il joue, il meurt au bout de 40 coups, et il faut déterminer lequel de ces 40 coups était mauvais. C'est le problème que résout l'apprentissage par renforcement.

L'idée est de faire apprendre au réseau une fonction $Q(s, a)$ pour une situation $s$ et une action $a$, combien de points je peux espérer gagner au total si je fais cette action maintenant et que je joue bien ensuite. Si on connaissait parfaitement $Q$, jouer serait trivial : dans chaque situation, on prend l'action de plus grande valeur.

L'équation de Bellman donne une façon de calculer une valeur proche :

```math
Q(s, a) = r + \gamma \cdot \max_{a'} Q(s', a')
```

La valeur d'une action, c'est la récompense immédiate $r$, plus la meilleure valeur possible depuis la situation suivante $s'$, réduite par un facteur $\gamma$ (gamma entre 0 et 1). Ce facteur dit à quel point l'agent est patient.

Quand la partie se termine, la valeur est simplement $Q(s,a) = r$.

![Boucle agent environnement](docs/images/bellman_illustration.png)

Cette équation transforme le problème en un apprentissage classique : elle fournit un target, et on retombe sur exactement le même mécanisme que pour XOR (erreur, gradient, correction des poids).

Brique de focntionnement :

**Le réseau cible** : La valeur à viser fait intervenir $Q$ du coup suivant, calculée par le réseau lui-même. On entraîne donc le réseau vers une cible qu'il produit, et qui bouge à chaque correction ce qui fausse les valeurs. La solution est d'utiliser une copie figée du réseau pour calculer les cibles, et de ne la remettre à jour que toutes les quelques milliers d'étapes.

**La mémoire des situations passées** : Deux coups consécutifs se ressemblent énormément, ce qui pose problème pour l'apprentissage (le réseau ne voit que la situation du moment et oublie le reste). On stocke donc des dizaines de milliers de situations vécues, et on pioche dedans au hasard pour s'entraîner.

### La convolution

Les couches denses prennent une ligne de nombre en entrée. Une image est une grille de pixel, on crée alors une matrice telle que son nombre de lignes correspond au nombre d'images (Une image = une ligne de la matrice). Ainsi dans ce projet, un pixel de C canaux de couleurs, de hauteur H et de largeur L, le pixel du canal c, de la hauteur h et de la largeur l sera mis dans la colonne :

```math
c \cdot H \cdot L + h \cdot L + l
```

![Rangement d'une image en une ligne d'une matrice](docs/images/cnn_image_en_ligne.png)

On normalise le niveau des couleurs variant de 0 à 255 de sorte à ce que ça tienne entre 0 et 1.

Passer une image directement dans les couches denses d'un MLP posent 3 problèmes :

**Le nombre de paramètres :** Pour une simple image de 200x200 pixels en couleurs, cela nécessite 120 000 poids pour chaque neurone de la première couche. Cela conpléxifie les calculs en plus de saturer la mémoire.

**La structure :** Un réseau de neurones classique traite les pixels comme une suite de valeur. Il ne sait pas que 2 pixels côte à côte. Il ignore que deux pixels côte à côte forment une forme, un contour ou une texture.

**Manque de généralisation :** Dans un réseau de neurones classique, si un élèment se trouve en haut à gauche de l'image puis se déplace en bas à droite, les pixels changent de place. Le réseau le considérera comme un objet totalement différent.

On utilise alors la convolution pour régler ces problèmes. Ce principe consiste à extraire les caractéristiques d'une image en balayant plusieurs filtres de taille $k \times k$ sur celle-ci. À chaque position, on calcule la somme pondérée des pixels qui se trouvent sous le filtre, et on ajoute un biais. On obtient alors carte de caractéristiques. Ainsi, au lieu d'avoir un poid par pixel, on a 1 poid par case de filtre.

Pour le filtre $f$, à la position $(h, l)$ :

```math
Y_{f,h,l} = b_f + \sum_{c=1}^{C} \sum_{i=0}^{k-1} \sum_{j=0}^{k-1} X_{c,\,h+i,\,l+j} \; K_{f,c,i,j}
```

Sans bord ajouté et avec un pas de 1, chaque filtre produit une carte de caractéristique de taille $(H - k + 1) \times (L - k + 1)$. Avec un filtre de 3×3, une image de 28×28 devient une image de 26×26, et une image de 64×64 devient une image de 62×62.

![Comparaison MLP et CNN](docs/images/cnn_convolution_filtre.png)

### Les filtres et l'astuce im2col

Écrire six boucles imbriquées afin de calculer la carte de caractéristiques d'une image peut vite devenir lent et ne permet pas d'exploiter la puissance de calcul d'un GPU. Dans ce projet, j'utilise la multiplication matricielle pour réaliser ce calcul. Pour cela, il faut ranger les poids et l'image d'une façon précise.

**La matrice de poids :** Un filtre voit les C canaux de l'entrée, avec $k \times k$ valeurs par canal. On l'aplatit en une ligne de $C \cdot k \cdot k$ valeurs. On place d'abord les k² valeurs du canal 0 les uns après les autres (lignes par lignes), puis ceux du canal 1 etc... Chaque ligne de la matrice représente un filtre. On forme ainsi la matrice de poids W et la matrice de biais b tel que :

```math
W \in \mathbb{R}^{F \times (C \cdot k \cdot k)} \qquad b \in \mathbb{R}^{1 \times F}
```

Par exemple, la deuxième convolution du deuxieme réseau CNN du projet sur les radiographies a C = 8 canaux en entrée, des filtres de dimension k = 3, et F = 16 filtres. Ainsi, sa matrice de poids et des dimensions $16 \times 72$.

**La matrice im2col :** Chaque position du filtre sur l'image correspond à un petit bloc de l'image. On crée une matrice im2col tel que son nombre de lignes correspond aux positions du filtre sur l'image. Pour chaque position, on prend le bloc formé sur les canaux, puis on les allignent lignes par lignes en commençant par le canal 0, puis le canal 1 etc... Pour un lot de N images, $N \times H' \times L'$ lignes, où $H'$ et $L'$ sont la hauteur et la largeur de la sortie. La convolution devient alors le calcul d'une couche dense :

```math
Y_{col} = \mathrm{im2col}(X) \cdot W^{T} + b
```

Chaque ligne de $Y_{col}$ correspond à une position du filtre et chaque colonne à un filtre (Elle a comme dimension $position \times F$). Puis, on remet le resultat dans l'ordre respectant la forme (nbFiltres, hauteur, largeur) attendu par la couche suivante.

![La matrice im2col et le produit avec les poids](docs/images/cnn_im2col.jpg)

Comme le calcul le même qu'un MLP classique, les gradients ont la même forme que pour une couche dense (on a $Y = X' \cdot W' + b $ où $X' = im2col(X)$ et $W' = W^{T}$).

```math
\frac{\partial L}{\partial W} = \left( \frac{\partial L}{\partial Y_{col}} \right)^{T} \cdot \mathrm{im2col}(X)
\qquad
\frac{\partial L}{\partial b} = \sum_{\text{lignes}} \frac{\partial L}{\partial Y_{col}}
```

```math
\frac{\partial L}{\partial\, \mathrm{im2col}(X)} = \frac{\partial L}{\partial Y_{col}} \cdot W
```

Cette dernière matrice est ensuite recollée dans la forme de l'image pour la rendre à la couche précédente dans le backward. Un même pixel appartient à plusieurs blocs qui se chevauchent. Ses contributions sont donc additionnées. On ne garde que la plus grande valeur de chaque carré.

### Le max pooling

Après une convolution, on cherche à réduire la taille de l'image. On la découpe en carré de $p \times p$ pixels sans que ces carrés se chevauchent.

![Le max pooling 2×2](docs/images/cnn_max_pooling.png)

Avec $p = 2$, la hauteur et la largeur sont divisées par 2 (26×26 devient 13×13) et le nombre de canaux ne change pas. Quand la taille est impaire (29×29 devient 14×14), la dernière ligne et la dernière colonne sont ignorées (C'est négligeable sur le resultat).

Pour la rétropropagation, seule la plus grande valeur de chaque carré a influencé la sortie. Le gradient est donc renvoyé uniquement à sa position, que la couche a mémorisée pendant le calcul avant, et vaut 0 pour les autres pixels du carré.

La couche **Flatten** ne calcule rien vue que les images sont déjà rangées à plat, une par ligne. Elle marque le passage entre la partie qui travaille sur des images et la partie dense, et elle vérifie les dimensions.

**Exemple d'un réseau CNN complet :**

![Réseau CNN exemple](docs/images/CNN-Pipeline.jpg)

### Le texte en nombres

Un LLM apprend une seule chose : deviner la suite d'un texte. On lui montre le début et il donne une probabilité pour chaque suite possible. Pour qu'un réseau puisse lire du texte, il faut d'abord le transformer en nombres.

On découpe le texte en morceaux appelés ***tokens***. Chaque token différent reçoit un numéro associé. L'ensemble des tokens possibles forme ce qu'on appelle le vocabulaire de taille V. Il existe plusieurs façon de les découper avec leurs avantages et leurs inconvéniants :

| Découpage | Exemple pour « chaton » | Taille du vocabulaire | Avantages et défauts |
|---|---|---|---|
| Par octet | c, h, a, t, o, n | 256 | Aucun mot inconnu possible, mais les textes deviennent très longs |
| Par mot | chaton | des centaines de milliers | Textes courts, mais un mot jamais vu ne peut pas être écrit |
| Par morceaux de mots (BPE) | chat, on | quelques dizaines de milliers | Le compromis utilisé par les grands modèles |

### Comment entrainer le LLM

On prend un morceau de texte aevc T tokens. La cible est tout simplement le même texte décalé d'un cran.

| position | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| ce que le modèle a vu | c | c h | c h a | c h a t |
| ce qu'il doit deviner | h | a | t | o |


### L'embedding

Chaque token est un numéro, mais ce numéro seul ne nous permet de faire des liens entre les tokens. Par exemple les token "chien" et "chat" ont un lien proche entre eux contrairement aux tokens "argent" et "bleu". On remplace donc chaque token par un vecteur de d nombres appris pendant l'entrainement. Les tokens qui s'utilisent de la même façon comme "chien" et "chat" finissent par avoir des vecteurs proches. Chaque valeurs de token à un vecteur associé, auquel addition un autre vecteur de sorte à préserver l'ordre dans une phrase. Sans ça, "le chat mange la souris" et "la souris mange le chat" seraient vus pareil.

Soit $E_t$, le vecteur du token t :

```math
E_t = \mathrm{Tok}[x_t] + \mathrm{Pos}[t]
```
Dans la rétropropagation, le gradient de la ligne $t$ s'ajoute à la ligne $x_t$ de $\mathrm{Tok}$ et à la ligne $t$ de $\mathrm{Pos}$. Si le même token apparaît plusieurs fois, sa ligne reçoit la somme de tous ses gradients.

### Couche de normalisation

En empilant beaucoup de couches, les valeurs peuvent grossir ou rétrécir de couche en couche, ce qui rend l'apprentissage instable. La LayerNorm recentre chaque vecteur, ligne par ligne, pour que ses d colonnes aient une moyenne de 0 et un écart-type de 1 :

```math
\mu = \frac{1}{d} \sum_{j=1}^{d} x_j
\qquad
\sigma^2 = \frac{1}{d} \sum_{j=1}^{d} (x_j - \mu)^2
\qquad
\hat{x}_j = \frac{x_j - \mu}{\sqrt{\sigma^2 + \varepsilon}}
```

Le petit $\varepsilon$ évite de diviser par zéro. Comme imposer exactement une de moyenne 0 et un écart-type 1 limiterait le réseau, on ajoute une échelle $\gamma$ et un décalage $\beta$ appris :

```math
y_j = \gamma_j \, \hat{x}_j + \beta_j
```

Chaque ligne est normalisée seule, donc le résultat ne dépend pas des autres exemples du lot.

Pour la rétropropagation, on note g le gradient reçu et $\hat{g} = g \odot \gamma$ :

```math
\frac{\partial L}{\partial \gamma} = \sum_{\text{lignes}} g \odot \hat{x}
\qquad
\frac{\partial L}{\partial \beta} = \sum_{\text{lignes}} g
```

```math
\frac{\partial L}{\partial x} = \frac{1}{\sqrt{\sigma^2 + \varepsilon}} \left( \hat{g} - \mathrm{moy}(\hat{g}) - \hat{x} \odot \mathrm{moy}(\hat{g} \odot \hat{x}) \right)
```

### La couche d'attention

Le problème d'une couche dense est qu'elle traite chaque token individuellement. Elle ne sait pas ce qu'il y a avant. Or pour deviner la suite "Le chat que j'ai vu hier soir dans le jardin est…", il faut retrouver chat qui est loin dans la phrase et ignorer les autres mots. L'attention permet à chaque token de choisir quels tokens regarder.

Chaque token fabrique 3 vecteurs avec 3 couche denses :

- une **question** Q : « qu'est-ce que je cherche ? » ;
- une **clé** K : « qu'est-ce que je contiens ? » ;
- une **valeur** V : « qu'est-ce que je transmets si on me choisit ? ».

Le score S entre le token a et b est le produit scalaire de la question de a avec la clé de b. Plus il est grand, plus les vecteurs vont dans la même direction.

```math
S = \frac{Q K^{T}}{\sqrt{d_k}}
```

Des scores trop grands rendraient le softmax presque binaire (un 1 et des 0) et ses gradients presque nuls. La division garde les scores à une bonne taille.

Attention, un modèle qui écrit du texte ne doit pas lire la suite pour deviner, sinon il n'apprendrait pas. Le token t n'a le droit de regarder que les tokens de 0 à t. On applique donc le softmax seulement sur les colonnes autorisées :

```math
A_{tj} = \begin{cases} \dfrac{e^{S_{tj}}}{\sum_{k \le t} e^{S_{tk}}} & \text{si } j \le t \\[2mm] 0 & \text{si } j > t \end{cases}
```

Chaque token reçoit enfin la moyenne des valeurs des tokens qu'il regarde, pondérée par ces poids :

```math
O = A V
```

O est ensuite transmis aux autres couches.

Pour la rétropropagation. On utilise la règle du produit de matrices suivante : $C = AB$ donne $\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} B^{T}$ et $\frac{\partial L}{\partial B} = A^{T} \frac{\partial L}{\partial C}$

```math
\frac{\partial L}{\partial A} = \frac{\partial L}{\partial O} V^{T}
\qquad
\frac{\partial L}{\partial V} = A^{T} \frac{\partial L}{\partial O}
```

Une seule attention donne, pour chaque token, un seul mélange. Mais un token a souvent besoin de plusieurs informations à la fois : le mot en cours, le sujet de la phrase, la dernière ponctuation etc…

On découpe les d colonnes de Q, K et V en H groupes de g = d/H. On les appelle des têtes. Chaque tête fait son propre calcul d'attention et peut donc se spécialiser. Les sorties des têtes sont recollées côte à côte, puis mélangées par une dernière couche dense.

Le calcul total est le même qu'avec une seule grande tête. On ne fait que découper les colonnes.

### Les connexions résiduelles

Quand on empile beaucoup de couches, le gradient est multiplié à chaque couche par une dérivée locale. S'il traverse des dizaines de couches, il peut devenir minuscule, et les premières couches n'apprennent presque plus rien. La connexion résiduelle ajoute l'entrée d'une couche à sa sortie.

```math
Y = X + R(X)
```

La couche n'apprend donc plus une transformation complète, seulement une **correction** $R(X)$ à ajouter. Et dans la rétropropagation :

```math
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} + \frac{\partial L}{\partial Y} \cdot \frac{\partial R}{\partial X}
```

### Un Transformer

Un Transformer assemble tout ce qui précède en deux moitiés, chacune avec sa connexion résiduelle :

```math
X_1 = X + \mathrm{MultiTête}(\mathrm{LN}_1(X))
```

```math
Y = X_1 + W_2 \, \mathrm{GELU}(W_1 \, \mathrm{LN}_2(X_1) + b_1) + b_2
```

- La première moitié fait communiquer les tokens entre eux.
- La deuxième moitié liasse chaque token réfléchir seul avec ce qu'il vient d'apprendre.

La LayerNorm est appliquée avant chaque moitié, et c'est l'entrée non normalisée qui est ajoutée par le résiduel.

L'entrée et la sortie d'un Transformer ont la même taille, ce qui permet d'en empiler autant qu'on veut.

Ainsi un LLM se construit de la manière suivante :

```
tokens → Embedding → Transformer 1 → Transformer 2 → … → Transformer N → LayerNorm → Dense (d → V) → Softmax
```

### Génération du texte

Une fois entrainé, le modèle peut écrire. On lui donne un début de texte, il calcule la probabilités du token suivant, il tire au hasard selon les probabilités et il l'ajoute au texte puis recommence. Si le texte est plus long que le contexte maximal C, on ne garde que les C premiers tokens.

On peut régler le hasard en divisant les scores s par une temperature y avant le softmax :

```math
p_i = \frac{e^{s_i / y}}{\sum_j e^{s_j / y}}
```

- $y < 1$ creuse les écarts : le modèle choisit presque toujours le token le plus probable. Le texte est prudent, mais il a tendance à se répéter.
- $y = 1$ garde les probabilités apprises.
- $y > 1$ aplatit les écarts : le texte devient plus original, puis incohérent si y est trop grand.

Prendre toujours le token le plus probable donne souvent des boucles du type "et il joua et il joua et il joua"

## Architecture



```
Matrix (Un tableau de nombres, stocké de façon optimisé, sur le processeur ou sur la carte graphique)
   │
   ├── Layer (une couche du réseau)
   │     ├── DenseLayer          : La couche qui fait le calcul : y = x·W + b
   │     ├── ActivationLayer     : Ajoute de la non-linéarité au réseau
   │     │     ├── ReLU
   │     │     ├── GeLU
   │     │     ├── Sigmoid
   │     │     ├── Tanh
   │     │     └── Identity
   │     ├── SoftmaxLayer        : Transforme les sorties en probabilités (somme égale à 1)
   │     ├── ConvLayer           : Convolution : fait glisser des filtres sur l'image (via im2col)
   │     ├── MaxPoolingLayer     : Réduit l'image en gardant la plus grande valeur de chaque carré
   │     ├── FlattenLayer        : Passe d'une image à une simple liste de nombres
   │     ├── EmbeddingLayer      : Transforme chaque octet en vecteur, et ajoute le vecteur de sa position
   │     ├── NormLayer           : LayerNorm, recentre chaque ligne (moyenne 0, écart-type 1)
   │     ├── AttentionLayer      : Attention à plusieurs têtes avec masque (les lettres échangent des informations)
   │     └── TransformerBloc     : LayerNorm, attention, LayerNorm, Dense, GELU, Dense, avec les deux résiduels
   │
   ├── Network                   : Stocke plusieurs couches, calcule dans les deux sens
   │
   ├── Loss
   │     ├── QuadraLoss          : Erreur quadratique, mesure l'écart entre la prédiction et la bonne réponse
   │     ├── BCE                 : Perte pour un choix entre 2 classes
   │     └── CCE                 : Perte pour 3 classes ou plus
   │
   ├── Optimizer
   │     ├── SGD                 : Méthode simple de mise à jour des poids
   │     └── Adam                : Méthode plus rapide et plus stable
   │
   ├── Initializers              : Choisit les poids de départ (Glorot, He, loi normale d'écart-type 0,02)
   ├── Random                    : Générateur de nombres aléatoires du projet
   └── Serializer                : Sauvegarde et recharge les poids d'un réseau entraîné

GPT(...)                         : Construit le LLM complet (Embedding, N Transformer, LayerNorm, Dense, Softmax)
Shape                            : Décrit une image : nombre de canaux, hauteur, largeur

Environment (Un monde dans lequel un agent peut agir)
   ├── CorridorEnv               : Un couloir simple pour vérifier que l'agent marche
   └── SnakeEnv                  : Le jeu Snake

Transition                       : Une situation vécue par l'agent (état, action, récompense, état suivant)
ReplayBuffer                     : Garde en mémoire les dernières situations vécues par l'agent
EpsilonSchedule                  : Dit à l'agent s'il explore au hasard ou non
DQNAgent                         : Assemble tout pour apprendre à jouer

DatasetImage                     : Charge un jeu d'images et leurs étiquettes et les mélange
DatasetText                      : Charge un fichier texte en octets et découpe des fenêtres
PredictionCnn                    : Prépare ce que l'interface affiche : la prédiction et les probabilités

gpu (calcul sur la carte graphique avec CUDA)
   ├── Gpu_calcul_matrix         : Produits de matrices (cuBLAS), produits par lots et opérations case par case
   ├── Gpu_Activation            : Fonctions d'activation et softmax (avec le masque de l'attention)
   ├── Gpu_Convolution           : im2col et son inverse pour la convolution
   ├── Gpu_MaxPooling            : Max pooling dans les deux sens
   ├── Gpu_Loss                  : Calcul des pertes et de leurs gradients
   ├── Gpu_Optimizer             : Mise à jour des poids avec Adam
   ├── Gpu_LLM                   : Embedding, LayerNorm, découpage et regroupement des têtes d'attention
   └── Memory_manage_matrix      : Réserve et libère la mémoire de la carte

Timer                            : Mesure les temps d'exécution
Logger                           : Affichage et suivi des entraînements (configuration, scores, journal)
```

## Agent DQN (Snake)

### Résultats

| Ce qui a été testé | Résultat |
|---|---|
| Calcul des gradients | Vérifié par une méthode numérique, écart inférieur à 1e-7 |
| Apprentissage du XOR, avec la méthode SGD | Erreur finale ≈ 0,0014 après 2000 essais |
| Apprentissage du XOR, avec l'optimiseur Adam | Erreur finale ≈ **8,7e-06** après 5000 essais (environ 4 fois plus rapide) |
| Agent qui apprend dans un couloir de 5 cases | Résolu à chaque fois, en 4 pas |
| Agent qui apprend à jouer à Snake (grille 10×10) | Score moyen sur 200 parties : 18.505 (Pommes mangées) |
| Série de tests | 580 tests, 0 échec |

Exemple de graphique représentant l'évolution de l'agent sur le jeu snake

![Graphique sur l'évolution de l'agent à snake](data/plot/training_snake_log_buffer_size.png)


### Démonstration

Une fois entraîné, l'agent joue tout seul, sans aide :

```
apps/agent_snake.cpp → charge le modèle entraîné et le fait jouer en temps réel dans le terminal
```

![GIF d'une démo de l'agent jouant à snake](docs/images/demo_snake_agent_gif.gif)

L'agent apprend bien à se diriger vers la nourriture. Sa limite principale est qu'il ne voit que les cases juste à côté de sa tête : il ne peut donc pas prévoir qu'il va s'enfermer dans son propre corps une fois qu'il devient trop long.

## Réseau convolutifs (CNN)

Après l'agent DQN, j'ai ajouté des classes permettant de constuire un réseau CNN permettant de travailler sur des images. J'ai pu traiter deux problèmes dans ce projet : reconnaître des chiffres écrits à la main (MNIST), puis détecter une pneumonie sur des radiographies du thorax.

La convolution, le max pooling, le chargement de dataset et l'interface graphique sont écrits à la main, sans bibliothèque externe (hormis pour extraire les pixels d'une image et l'interface graphique). Les CNN réutilisent le même `Network`, le même optimiseur Adam, la même perte CCE et le même `Serializer` que le reste du projet. Seules de nouvelles couches ont été ajoutées.

Les nouveaux éléments :

```
Layer (une couche du réseau)
   ├── ConvLayer             : Convolution : fait glisser des filtres sur l'image (via im2col)
   ├── MaxPoolingLayer       : Réduit l'image en gardant la plus grande valeur de chaque carré
   ├── FlattenLayer          : Passe d'une image à une simple liste de nombres
   └── SoftmaxLayer          : Transforme les sorties en probabilités (somme égale à 1)
 
Shape                        : Décrit une image : nombre de canaux, hauteur, largeur
DatasetImage                 : Charge un jeu d'images et leurs étiquettes et les mélange
PredictionCnn                : Prépare ce que l'interface affiche : la prédiction et les probabilités
```

### Chiffres manuscrits (MNIST)

MNIST contient 70 000 images de 28×28 pixels en niveaux de gris, dont 60 000 pour l'entraînement et 10 000 pour le test.

| Mesure | Valeur |
|---|---|
| Précision sur les 10 000 images de test | **97,76 %** (environ 224 images mal reconnues) |
| Entraînement | 10 epochs, lots de 64 images, Adam (taux d'apprentissage 0,001), perte CCE |
| Erreur d'entraînement finale | ≈ 0,057 |
| Temps d'entraînement | ≈ 376 s sur mon PC portable en Release avec RTX 4070 Mobile et intel ultra core 7 155h, soit environ 38 s par epoch |
| Nombre de poids du réseau | 13 610 |

| Étape | Couche | Taille de la sortie | Poids |
|---|---|---|---|
| Entrée | Image 1 × 28 × 28 | 784 | |
| 1 | Convolution : 8 filtres de 3×3 | 8 × 26 × 26 = 5 408 | 80 |
| 2 | ReLU | 5 408 | 0 |
| 3 | Max pooling 2×2 | 8 × 13 × 13 = 1 352 | 0 |
| 4 | Flatten | 1 352 | 0 |
| 5 | Couche dense | 10 | 13 530 |
| 6 | Softmax | 10 probabilités | 0 |

Le réseau donne une probabilité pour chacun des 10 chiffres, et le chiffre deviné est celui qui a la plus grande probabilité. Les poids se sauvegardent et se rechargent avec le même `Serializer` que pour le DQN.

### Radiographies pulmonaires (pneumonie)

Pour tester le CNN sur un vrai problème, je l'ai entraîné sur des radiographies du thorax. Le réseau doit dire si une image est "NORMAL" ou "PNEUMONIA". Les images viennent du jeu *Chest X-Ray Images (Pneumonia)* de Kermany et al. (Cell, 2018), disponible sur [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) sous licence CC BY 4.0. Il contient environ 5 200 images d'entraînement et 624 images de test. Elles sont converties en niveaux de gris et réduites à 64 × 64 pixels.

| Mesure | Valeur |
|---|---|
| Précision sur les 624 images de test | **76,60 %** (478 images bien classées) |
| Entraînement | 20 epochs depuis zéro, puis 67 epochs de plus en reprenant le modèle sauvegardé. Lots de 32 images, Adam (taux d'apprentissage 0,001), perte CCE |
| Précision après les 20 premières epochs | 76,28 % |
| Erreur d'entraînement finale | ≈ 0,000002 |
| Temps d'entraînement | environ 30 s par epoch sur mon PC portable en Release avec RTX 4070 Mobile et intel ultra core 7 155h (≈ 10 min pour 20 epochs, ≈ 33 min pour 67) |
| Nombre de poids du réseau | 202 146 |


| Étape | Couche | Taille de la sortie | Poids |
|---|---|---|---|
| Entrée | Image 1 × 64 × 64 | 4 096 | |
| 1 | Convolution : 8 filtres de 3×3 | 8 × 62 × 62 = 30 752 | 80 |
| 2 | ReLU | 30 752 | 0 |
| 3 | Max pooling 2×2 | 8 × 31 × 31 = 7 688 | 0 |
| 4 | Convolution : 16 filtres de 3×3 | 16 × 29 × 29 = 13 456 | 1 168 |
| 5 | ReLU | 13 456 | 0 |
| 6 | Max pooling 2×2 | 16 × 14 × 14 = 3 136 | 0 |
| 7 | Flatten | 3 136 | 0 |
| 8 | Couche dense | 64 | 200 768 |
| 9 | ReLU | 64 | 0 |
| 10 | Couche dense | 2 | 130 |
| 11 | Softmax | 2 probabilités | 0 |

Presque tous les poids (200 768 sur 202 146) sont dans la première couche dense, entre les 3 136 valeurs aplaties et les 64 neurones.

**Attention, ce n'est pas un outil médical. Ce projet sert à apprendre à écrire un CNN. Un score de 76,6 % n'a aucune valeur de diagnostic.**

### Les données et leur chargement

La classe `DatasetImage` charge un jeu d'images selon un type donné à sa création :

| Type | Format | État |
|---|---|---|
| 0 | Un dossier par classe, contenant des images PNG, JPG ou BMP (le nom du dossier est le nom de la classe). Les images sont converties au nombre de canaux voulu et redimensionnées à la taille voulue | Fonctionne (radiographies) |
| 1 | Fichiers IDX, le format de MNIST | Fonctionne (MNIST) |
| 2 et 3 | Étiquettes lues dans un fichier de noms ou dans un CSV | Prévus, pas encore écrits |

Les images sont rangées dans une matrice (une image par ligne : 784 colonnes pour MNIST, 4 096 pour les radiographies en 64×64). Les étiquettes sont dans une matrice à une seule valeur à 1 par ligne : pour un « 7 », la ligne contient dix valeurs, avec un 1 en position 7 et des 0 ailleurs. Pour le type 0, les classes sont rangées par ordre alphabétique des noms de dossiers (NORMAL est la classe 0, PNEUMONIA la classe 1). `shuffle()` mélange images et étiquettes de la même façon, et `get_batch(début, taille)` renvoie un lot d'images avec ses étiquettes.

Les jeux de données ne sont pas fournis dans le dépôt (ils sont volumineux). Il faut les télécharger et les placer dans le dossier `dataset_images` :



```
data/dataset_images/
├── MNIST/
│   ├── train/    train-images.idx3-ubyte    train-labels.idx1-ubyte
│   └── test/     t10k-images.idx3-ubyte     t10k-labels.idx1-ubyte
└── chest_xray/
    ├── train/    NORMAL/    PNEUMONIA/
    └── test/     NORMAL/    PNEUMONIA/
```
Pour MNIST, le chargeur repère les fichiers dont le nom contient `idx3` (images) et `idx1` (étiquettes). Les noms écrits avec des tirets, comme `train-images-idx3-ubyte`, fonctionnent donc aussi. Pour les dossiers d'images, le décodage des fichiers PNG, JPG et BMP est fait par `stb_image`, un fichier d'en-tête inclus dans `include/utils/`. Si le jeu des radiographies a été téléchargé depuis macOS, il faut supprimer les dossiers `__MACOSX` et les fichiers `.DS_Store`, qui ne sont pas des images.

### Entraîner, tester, visualiser

```bash
./build/train_mnist    # entraîne le CNN sur MNIST et sauvegarde models/reseau_mnist.txt
./build/test_mnist     # mesure la précision sur les 10 000 images de test de MNIST
./build/train_cnn      # entraîne le CNN sur les radiographies et sauvegarde models/chest_xray.txt
./build/test_cnn       # mesure la précision sur les 624 radiographies de test
./build/gui_cnn        # interface graphique sur les radiographies (nécessite le modèle entraîné)
```

- Le répertoire de travail doit être la racine du dépôt. Sous CLion, il faut régler le « Working directory » de la configuration de lancement, qui est par défaut le dossier de compilation. Sinon les chemins comme `models/chest_xray.txt` ne sont pas trouvés.
- Compiler en **Release** : l'entraînement est plusieurs fois plus lent en Debug.
- Le nombre d'epochs se règle dans le fichier du programme d'entraînement.
- `train_cnn` recharge d'abord `models/chest_xray.txt` s'il existe, puis continue l'entraînement. Pour repartir de zéro, il faut supprimer ce fichier avant de lancer.
- Chaque programme doit reconstruire exactement le même réseau que celui de l'entraînement avant de charger les poids, car le fichier de sauvegarde ne contient que les valeurs.

### Interface graphique

`gui_cnn` ouvre une fenêtre qui montre ce que voit le réseau et ce qu'il répond, image par image, sur les radiographies de test (mélangées au lancement) :

- La radiographie de 64×64 pixels, agrandie et dessinée pixel par pixel ;
- La classe devinée et la probabilité correspondante ;
- Une barre par classe (NORMAL et PNEUMONIA) : la barre de la classe devinée est verte si la réponse est bonne et rouge si elle est fausse, l'autre est grise ;
- La vraie classe et si la réponse est correcte ;
- Les flèches gauche et droite pour passer à l'image précédente ou suivante.


![exemple du réseau cnn sur ce dataset](docs/images/exemple_cnn.gif)

### Limites et pistes du CNN

**Le modèle des radiographies a appris ses images par cœur (surapprentissage).** L'erreur d'entraînement tombe à presque 0 alors que la précision sur le test reste autour de 76 %. Passer de 20 à 87 epochs n'a presque rien changé au test (76,28 % puis 76,60 %). Les images de test viennent aussi d'une autre distribution que celles d'entraînement, ce qui rend ce jeu difficile.

**Pas de jeu de validation.** Je n'ai pas mis de côté des images pour choisir le nombre d'epochs ou arrêter l'entraînement au bon moment.

**Convolution et pooling simples.** La convolution a toujours un pas de 1 et n'ajoute pas de padding.

**C'est lent.** La convolution est écrite avec des boucles simples, sur un seul fil d'exécution. Une epoch prend environ 38 s sur MNIST et 30 s sur les radiographies en Release sur mon PC portable.

**L'architecture est écrite à la main dans chaque programme.** `train_mnist`, `test_mnist`, `train_cnn`, `test_cnn` et `PredictionCnn` construisent chacun leur réseau, et le fichier de sauvegarde ne retient pas sa forme.

**L'interface graphique est branchée sur le modèle des radiographies.** Le réseau construit dans `PredictionCnn.cpp` est celui des radiographies. Pour afficher MNIST, il faut y remettre le réseau de MNIST. Les types 2 et 3 de `DatasetImage` ne sont pas écrits.

Pistes :

- Créer un jeu de validation et arrêter l'entraînement quand son erreur remonte ;
- Ajouter de l'augmentation de données (petits décalages, variations de luminosité) ;
- Ajouter le pas et le bord (padding) à la convolution ;
- Sauvegarder la forme du réseau dans le fichier du modèle, pour ne plus la réécrire dans chaque programme ;
- Choisir le réseau de l'interface graphique selon le jeu de données ;

## Petit LLM from scratch

La dernière partie du projet. Un petit llm construit sur l'architecture Transformer qui apprend à écrire des histoire en français, lettre par lettre. Il réutilise les mêmes briques de base que le DQN et le CNN. On ajoute à ces couches : L'embedding, la couche de normalisation, la fonction d'activation GeLU, l'attention à plusieurs têtes et le Transformer. Tout l'entraînement tourne sur la carte graphique avec CUDA.

Le fonctionnement de chaque couche est expliqué dans [Comment ça marche](#le-texte-en-nombres).

### Résultats du LLM

| Mesure | Valeur |
|---|---|
| Nombre de paramètres | **25 613 568** (≈ 25,6 M) |
| Données vues pendant l'entraînement | 120 000 tours × 4 096 lettres ≈ 491 millions de lettres (≈ 29 % du jeu d'entraînement) |
| Perte au départ | 5,70 |
| Perte de validation finale | **0,344** (≈ 0,5 bit par lettre) |
| Perte d'entraînement finale | 0,340 |
| Temps d'entraînement | **≈ 7 h 03** (25 379 s, soit ≈ 0,21 s par tour) sur mon PC portable en Release avec RTX 4070 Mobile et Intel Core Ultra 7 155H |

Exemple de texte écrit par le modèle (température 0,8). Le début vient du fichier de test de LeCarnet, qui contient des histoires coupées au milieu d'une phrase :

**Début donné au modèle :**

```
Il était une fois un petit lapin nommé Lulu. Lulu adorait jouer dans le jardin avec ses amis. Un jour, il trouva un vieux tuyau dans le jardin. Intrigué, il décida de viser un trou dans le sol avec le tuyau pour voir s'il pouvait l'utiliser comme un toboggan. Mais quand il essaya de glisser à travers le tuyau, il se retrouva coincé ! Lulu
```

**Suite écrite par le modèle, lettre par lettre, jusqu'à ce qu'il décide lui-même de s'arrêter :**

```
était triste et commença à pleurer. Soudain, il entendit une voix douce : « Ne t'inquiète pas, Lulu. Je vais t'aider à glisser. » C'était la sorcière qui parlait aux animaux de la forêt. Ensemble, ils trouvèrent un moyen de libérer Lulu. La sorcière, surprise par ce geste gentil, accepta de les aider. Ensemble, ils trouvèrent une feuille et Lulu put repartir en sécurité dans sa maison. Lulu apprit que parfois, il vaut mieux demander de l'aide plutôt que de tout faire seul. Depuis ce jour, Lulu n'hésitait plus à demander de l'aide quand il en aurait besoin. Il comprit que l'équilibre est une différence et que tout le monde dans la forêt pouvait être une force. Lulu apprit que la gentillesse et l'aide aux autres étaient plus importantes que de s'amuser tout seul.
```

Le modèle écrit en français correct, garde la structure du texte avec le nom des personnages, le problème, la morale, et termine l'histoire de lui-même. Le modèle à des limites. Par exemple la sorcière arrive sans explication, certaines phrases n'ont pas de sens ("l'équilibre est une différence"), et la morale est répétée. Il maîtrise bien la forme du texte, mais plus l'histoire est longue, moins il maitirse la logique.

### Les données : LeCarnet

Le modèle est entraîné sur [LeCarnet](https://huggingface.co/datasets/MaxLSB/LeCarnet) (licence ODC-BY). Il y environ 2 millions de courtes histoires pour enfants en français, écrites avec un vocabulaire simple par un grand modèle de langage sur le modèle du jeu anglais TinyStories.

Dans ce projet, le texte est découpé en octet. Les lettres accentuées comme "é" prennent deux octets en UTF-8. Le modèle doit donc apprendre à écrire les octets dans le bonne ordre.

`DatasetText::get_batch` tire au hasard 16 positions dans le texte et prend à chacune 256 octets pour l'entrée x et les mêmes 256 octets décalés d'un cran pour la cible y :

| position | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| x (entrée) | I | l | ␣ | é | t | a | i | t |
| y (à deviner) | l | ␣ | é | t | a | i | t | ␣ |

On note B le nombre de lot et T la taille de la fenêtre. x est rangée dans une matrice de $B \cdot T$ lignes et de 1 colonne. Chaque ligne représente une lettre auquel on lui associe sa valeur en octet. Si une lettre fait 2 octets par exemple, elle prendra 2 lignes dans la matrice. `get_batch_in_order` découpe le texte de validation dans l'ordre, pour mesurer la perte toujours sur les mêmes lots.

### Le modèle

| Réglage | Valeur |
|---|---|
| Vocabulaire V | 256 octets |
| Taille des vecteurs d | 512 |
| Nombre de blocs Transformer | 8 |
| Têtes d'attention H | 8 (64 colonnes par tête) |
| Couche cachée des blocs | 2 048 (4 x d) |
| Contexte T | 256 lettres |

| Étape | Couche | Taille de la sortie (pour un lot de $B \cdot T$ lettres) | Poids |
|---|---|---|---|
| 1 | Embedding (octets + positions) | $B \cdot T$ x 512 | 262 144 |
| 2 | 8 × TransformerBloc | $B \cdot T$ x 512 | 8 x 3 152 384 = 25 219 072 |
| 3 | LayerNorm finale | $B \cdot T$ x 512 | 1 024 |
| 4 | Couche dense | $B \cdot T$ x 256 | 131 328 |
| 5 | Softmax | 256 probabilités par lettre | 0 |

Dans chaque bloc, environ deux tiers des poids sont dans les deux couches denses ($512 \to 2048 \to 512$), et un tiers dans l'attention.

### Les nouvelles classes

```
Layer (une couche du réseau)
   ├── EmbeddingLayer        : Transforme chaque octet en vecteur, et ajoute le vecteur de sa position
   ├── NormLayer             : LayerNorm, recentre chaque ligne (moyenne 0, écart-type 1)
   ├── GeLU                  : Fonction d'activation des LLM
   ├── AttentionLayer        : Attention à plusieurs têtes avec masque (les lettres échangent des informations)
   └── TransformerBloc       : LayerNorm, attention, LayerNorm, Dense, GELU, Dense, avec les deux résiduels
 
GPT(...)                     : Construit le réseau complet (Embedding, N Transformer, LayerNorm, Dense, Softmax)
DatasetText                  : Charge un fichier texte en octets et découpe des fenêtres
```

Le contexte utilisé se règle avec `Network::set_actual_context`, qui le transmet à l'Embedding et à toutes les couches d'attention.

### Entraînement

| Réglage | Valeur |
|---|---|
| Lots | 16 fenêtres de 256 lettres, soit 4 096 lettres par tour |
| Optimiseur | Adam, $\beta_1$ = 0,9, $\beta_2$ = 0,99, $\varepsilon$ = 1e-8 |
| Taux d'apprentissage | montée de 0 à 3e-4 sur 2 000 tours, puis descente en cosinus jusqu'à 3e-5 |
| Durée | 120 000 tours |
| Initialisation | loi normale d'écart-type 0,02 pour l'Embedding et toutes les couches denses (comme GPT-2) |
| Suivi | perte de validation tous les 1 000 tours, sur les 50 mêmes lots |
| Sauvegardes | meilleur modèle et sauvegarde de secours tous les 5 000 tours, modèle final |

$\beta_2$ = 0,99 au lieu de 0,999 : avec de petits lots, une moyenne des carrés des gradients sur environ 100 pas au lieu de 1 000 réagit plus vite et rend l'entraînement plus stable.

**Les contrôles faits avant le grand entraînement :**

| Test | Résultat |
|---|---|
| Perte de départ | 5,70, proche de ln 256 ≈ 5,55 |
| Même calcul sur CPU et sur la carte | même perte (5,696) |
| Apprendre un seul lot par cœur, taux 1e-3 | bloqué vers 3,0 puis 2,4 : taux trop fort |
| Apprendre un seul lot par cœur, taux 3e-4 | 5,70 puis 0,005 en 1 000 tours |
| Masque : changer la dernière lettre d'une fenêtre | sorties des autres lettres identiques (écart 0) |

**Évolution de la perte de validation :**

| Tour | 1 000 | 5 000 | 10 000 | 20 000 | 40 000 | 60 000 | 80 000 | 100 000 | 119 000 |
|---|---|---|---|---|---|---|---|---|---|
| Perte de val | 1,194 | 0,585 | 0,503 | 0,445 | 0,401 | 0,378 | 0,360 | 0,349 | 0,344 |

![Courbe d'entraînement du LLM](data/plot/training_gpt_log.png)

La perte de validation reste très proche de la perte d'entraînement. Le modèle ne voit presque jamais deux fois le même passage, il ne peut donc pas apprendre le texte par cœur. La perte baissait encore légèrement à la fin donc un entraînement plus long gagnerait encore un peu.

### Limites et pistes du LLM

- **Le découpage en octets** est simple à mettre en place mais il coûte cher. Une lettre accentuée prend deux octets, et un contexte de 256 octets ne couvre qu'une quarantaine de mots. Un découpage en morceaux de mots donnerait un contexte bien plus long pour le même calcul.
- **Une perte basse grâce aux données** : les histoires de LeCarnet sont simples et répétitives, donc beaucoup plus faciles à prédire qu'un texte basique. Le modèle ne sait écrire que ce genre d'histoires.
- **La sauvegarde** est un fichier texte d'environ 360 Mo, long à écrire et à relire. L'état d'Adam n'est pas sauvegardé. Après une reprise, il faut refaire une courte montée du taux.
- **Piste** : un affinage sur des questions et réponses, avec une perte qui ne compte que les lettres de la réponse, pour transformer ce modèle qui raconte en un modèle qui répond.


## Compilation et exécution

Prérequis : CMake ≥ 3.22, un compilateur C++17 (GCC, Clang ou MSVC).

| Élément | À quoi il sert | Obligatoire ? |
|---|---|---|
| Git et une connexion internet pendant la première configuration de CMake | CMake télécharge et compile automatiquement raylib 6.2 (`FetchContent`). Cette première compilation prend quelques minutes | Oui, dès la configuration, même si l'on ne lance pas l'interface |
| Un pilote graphique OpenGL 3.3 et un environnement de bureau | Ouvrir la fenêtre de `gui_cnn` | Pour `gui_cnn` seulement |
| Sous Linux : les paquets de développement X11 et OpenGL | Compiler raylib. Sous Debian ou Ubuntu : `sudo apt install libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libgl1-mesa-dev libglu1-mesa-dev` | Linux seulement |
| Les 4 fichiers du jeu MNIST (format IDX) | Entraîner et tester le CNN sur les chiffres. Ils sont à placer dans `data/dataset_images/MNIST/` | Pour `train_mnist` et `test_mnist` |
| Le jeu de radiographies *Chest X-Ray Images (Pneumonia)* | Entraîner, tester et visualiser le CNN sur les radiographies. Il est à placer dans `data/dataset_images/chest_xray/` | Pour `train_cnn`, `test_cnn` et `gui_cnn` |
| CUDA Toolkit | Faire les grosses multiplications de matrices sur un GPU NVIDIA | Non. CMake le détecte seul, et sans lui le calcul se fait sur le processeur. L'architecture GPU est fixée à 89 (cartes RTX série 40) dans `CMakeLists.txt`, à changer pour une autre carte |

Le réseau, les couches et l'entraînement restent écrits avec la seule bibliothèque standard du C++. `stb` ne sert qu'à lire les images et raylib qu'à l'interface graphique.

```bash
git clone https://github.com/Riyaneb/Reseau_de_neurones_from_scratch_Cpp.git
cd Reseau_de_neurones_from_scratch_Cpp
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

Programmes produits :

| Programme | Ce qu'il fait |
|---|---|
| `test_module` | Lance tous les tests |
| `train_xor` | Entraîne un petit réseau sur XOR (avec les deux méthodes d'entraînement) |
| `train_corridor` | Entraîne l'agent sur l'environnement couloir |
| `train_snake` | Entraîne l'agent sur Snake, garde une trace de l'entraînement, sauvegarde le meilleur résultat |
| `play_snake` | Jouer à Snake soi-même, au clavier |
| `agent_snake` | Regarder l'agent entraîné jouer tout seul |
| `benchmark_matrix` | Mesure la vitesse du calcul matriciel selon la façon dont il est écrit |
| `train_mnist` | Entraîne le CNN sur MNIST et sauvegarde le modèle |
| `test_mnist` | Mesure la précision du CNN entraîné sur les images de test de MNIST |
| `train_cnn` | Entraîne le CNN sur les radiographies (reprend le modèle sauvegardé s'il existe) et sauvegarde le modèle |
| `test_cnn` | Mesure la précision du CNN entraîné sur les radiographies de test |
| `gui_cnn` | Interface graphique : parcourir les radiographies de test et voir la réponse du CNN |


Lancer les tests :

```bash
./build/test_module
```

Lancer un entraînement :

```bash
./build/train_snake
./build/agent_snake
```

> Le répertoire de travail doit être la racine du dépôt.

## Choix de conception

Quelques décisions importantes, et pourquoi je les ai prises :

**Les matrices sont stockées comme une seule liste de nombres, pas une liste de listes.**
Plus rapide à parcourir en mémoire.

**Un seul exemple est traité comme un groupe d'un seul élément.**
Le même code sert à traiter un exemple ou 64 en même temps. Pas besoin d'écrire deux versions différentes selon qu'on entraîne le réseau ou qu'on l'utilise.

**Le calcul des gradients a été vérifié à la main**
Pour chaque poids, j'ai comparé le gradient calculé par mon code à une estimation obtenue autrement (en bougeant légèrement le poids et en regardant comment l'erreur change). Les deux résultats concordent presque parfaitement, ce qui prouve que le calcul est juste.

**Le réseau ne sait pas comment il est entraîné.**
Il ne connaît ni la façon de mesurer l'erreur, ni la façon de corriger ses poids. Ça permet de changer de méthode d'entraînement (passer de la méthode simple à Adam par exemple) sans toucher au reste du code.

**Quand l'agent apprend, on ne corrige que l'action qu'il a réellement choisie.**
Pour chaque situation vécue, seule la valeur associée à l'action jouée est corrigée et les trois autres actions possibles restent inchangées.

**La sauvegarde ne garde que les poids, pas la structure du réseau.**
Pour recharger un modèle, il faut reconstruire un réseau avec exactement la même forme dans le code. Cela permet de simplifier la sauvegarde.

## Tests

La suite de tests (`test_module`) vérifie, dans l'ordre où le projet a été construit :

- toutes les opérations sur les matrices (addition, multiplication, transposée, etc...)
- le générateur de nombres aléatoires et les méthodes d'initialisation des poids
- les couches du réseau et les fonctions d'activation
- que le calcul des gradients est correct
- que la copie d'un réseau est bien indépendante de l'original
- l'optimiseur Adam
- les deux environnements de jeu (couloir et Snake) : collisions, demi-tour interdit, limite de temps, etc...
- la mémoire des situations passées de l'agent (replay buffer)

## Limites connues

**L'agent ne voit que 11 informations simples**, pas la grille entière : y a-t-il un danger devant / à gauche / à droite, dans quelle direction il va, et de quel côté se trouve la nourriture. Ce choix rend l'apprentissage plus rapide, mais l'agent ne voit jamais son propre corps au-delà des cases juste à côté de sa tête. Il apprend donc à aller vers la nourriture, mais peut se retrouver enfermé dans son propre corps une fois qu'il est devenu long, sans pouvoir le prévoir à l'avance.

**La sauvegarde ne garde que les poids, pas la forme du réseau.** Recharger un modèle veut dire reconstruire un réseau de la même forme dans le code avant de charger les valeurs dedans.

**Reprendre un entraînement ne se fait pas exactement où il s'était arrêté.** Certains réglages internes de l'optimiseur (Adam) repartent de zéro.

## Pistes d'amélioration

- **Double DQN** : une variante qui limite le risque que l'agent surestime la valeur de certaines actions.
- **Voir plus loin** : donner à l'agent une vision plus large que les cases juste à côté de lui, pour qu'il puisse anticiper les impasses.
- **Mémoire plus intelligente** : donner plus d'importance aux situations où l'agent s'est le plus trompé, plutôt que de piocher au hasard dans sa mémoire.
- **Récompense progressive** : donner une petite récompense à l'agent quand il se rapproche de la nourriture, pas seulement quand il la mange, pour qu'il apprenne plus vite au début.

## Structure du dépôt

```
include/           déclarations du code, rangées par thème (math, nn, rl, env, io, utils), cnn (convolution, pooling) et gpu (calcul CUDA) s'ajoutent à cette liste
src/               le code lui-même qui correspond à include/
tests/             tous les tests
apps/              les programmes qu'on peut lancer (entraînement, jeu, démonstration)
models/            les réseaux entraînés, sauvegardés
docs/              les images utilisées dans ce README
data/              les données récoltées pendant l'entraînement et les jeux d'images (dataset_images) ainsi que la police de l'interface graphique
```


*Riyane Bouakaz [GitHub](https://github.com/Riyaneb) · [Kaggle](https://kaggle.com/riyanebouakaz) · [LinkedIn](https://linkedin.com/in/riyane-bouakaz)*