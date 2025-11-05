# 06 — Approximation de Fonctions & Généralisation

## Introduction

Lorsqu'on traite des espaces d'états continus ou très grands, maintenir des représentations tabulaires explicites des fonctions de valeur (tables V, tables Q) devient computationnellement irréalisable ou impossible. L'**approximation de fonctions** (function approximation) répond à ce défi en représentant les fonctions de valeur par des fonctions paramétrées $\hat{V}(s;\theta)$ ou $\hat{Q}(s,a;\theta)$, où $\theta$ représente les paramètres apprenables.

```{admonition} Pourquoi l'Approximation de Fonctions ?
:class: tip
Cette approche permet :
- **Généralisation** : L'apprentissage sur des états visités se transfère aux états non-visités
- **Scalabilité** : Gestion d'espaces d'états de haute dimension ou continus
- **Efficacité** : Représentation compacte au lieu d'une énumération exhaustive
```

---

## 1. Approximation Linéaire (Linear Function Approximation)

### Concept

On construit une représentation de features $\phi(s) \in \mathbb{R}^d$ qui transforme les états en vecteurs de features de dimension fixe, puis on approxime la fonction de valeur comme une combinaison linéaire :

$$\hat{V}(s;\theta) = \theta^T \phi(s) = \sum_{i=1}^{d} \theta_i \phi_i(s)$$

```{admonition} Représentations de Features Courantes
:class: note
Le **feature engineering** est crucial pour l'approximation linéaire. Choix courants :
- **Fonctions polynomiales** (polynomial basis) : $\phi(s) = [1, s, s^2, s^3, ...]$
- **Fonctions à base radiale** (radial basis functions - RBFs) : $\phi_i(s) = \exp(-\|s - c_i\|^2 / 2\sigma^2)$
- **Tile coding** : Discrétisation avec recouvrement pour espaces continus
- **Base de Fourier** (Fourier basis) : $\phi_i(s) = \cos(\pi i^T s)$
```

### Mise à Jour Basée sur le Gradient (Semi-Gradient TD)

Pour TD(0), on minimise l'erreur de différence temporelle (temporal difference error) en utilisant la descente de gradient stochastique :

$$\theta \leftarrow \theta + \alpha \delta_t \nabla_{\theta} \hat{V}(S_t;\theta)$$

où l'erreur TD (TD error) est :

$$\delta_t = R_{t+1} + \gamma \hat{V}(S_{t+1};\theta) - \hat{V}(S_t;\theta)$$

Pour l'approximation linéaire, $\nabla_{\theta} \hat{V}(S_t;\theta) = \phi(S_t)$, ce qui donne :

$$\theta \leftarrow \theta + \alpha \delta_t \phi(S_t)$$

```{admonition} Méthodes Semi-Gradient
:class: warning
On appelle cela une méthode "semi-gradient" car on traite $\hat{V}(S_{t+1};\theta)$ dans la cible TD comme une constante (on ne dérive pas par rapport à elle). Cela brise le vrai gradient mais fonctionne souvent bien en pratique et est computationnellement plus simple.
```

### Algorithme : Semi-Gradient TD(0) avec Approximation Linéaire

```
Algorithme : Semi-Gradient TD(0) (Linear Function Approximation)
─────────────────────────────────────────────────────────────────
Entrée : learning rate α, discount γ, features φ(s)
Sortie : θ tel que V̂(s;θ) ≈ V^π(s)

// Initialisation
θ ← 0  (ou petit aléatoire)

// Apprentissage
répéter pour chaque épisode :
    Initialiser S
    
    répéter pour chaque step :
        Exécuter action selon π, observer R, S'
        
        // Calculer erreur TD
        δ ← R + γ·V̂(S';θ) - V̂(S;θ)
        
        // Mise à jour semi-gradient
        θ ← θ + α·δ·φ(S)
        
        S ← S'
        
    jusqu'à S terminal
    
jusqu'à convergence

retourner θ
```

### Avantages et Limitations

```{list-table}
:header-rows: 1
:widths: 50 50

* - ✓ Avantages
  - ✗ Limitations
* - Garanties de convergence sous certaines conditions
  - Expressivité limitée pour les patterns complexes
* - Simple et interprétable
  - Nécessite un design manuel des features
* - Calcul rapide
  - Ne peut pas capturer des relations non-linéaires arbitraires
* - Théorie bien comprise
  - Performance dépend fortement de la qualité des features
```

---

## 2. Approximation Non-Linéaire : Deep Reinforcement Learning

### Approximateurs par Réseaux de Neurones (Neural Network Approximators)

Les réseaux de neurones profonds fournissent une représentation de features flexible et apprenable. Pour les fonctions de valeur action-état (action-value functions) :

$$\hat{Q}(s,a;\theta)$$

où $\theta$ représente maintenant tous les poids et biais du réseau (potentiellement des millions de paramètres).

### Objectif d'Entraînement (Training Objective)

On utilise la descente de gradient stochastique sur l'erreur quadratique de Bellman (mean squared Bellman error) :

$$L(\theta) = \mathbb{E}_{(s,a,r,s')\sim \mathcal{D}}\left[ \left(y - Q(s,a;\theta)\right)^2 \right]$$

```{admonition} Variantes de Calcul de Cible (Target Computation)
:class: note
La cible $y$ dépend de l'algorithme :
- **DQN** : $y = r + \gamma \max_{a'} Q(s',a';\theta^-)$ (utilise le target network)
- **Double DQN** : $y = r + \gamma Q(s', \arg\max_{a'} Q(s',a';\theta); \theta^-)$ (réduit la surestimation)
- **Sarsa** : $y = r + \gamma Q(s',a';\theta)$ (on-policy, utilise l'action suivante réelle)
```

### Mise à Jour du Gradient

$$\theta \leftarrow \theta - \alpha \nabla_\theta L(\theta) = \theta + \alpha (y - Q(s,a;\theta)) \nabla_\theta Q(s,a;\theta)$$

---

## 3. Techniques de Stabilisation

```{admonition} Sources d'Instabilité en Deep RL
:class: danger
- **Cibles mouvantes** (moving targets) : La cible TD dépend de $\theta$, qui change constamment
- **Échantillons corrélés** (correlated samples) : Les expériences séquentielles sont fortement corrélées
- **Divergence de l'approximation** : Aucune garantie de convergence pour les approximateurs non-linéaires
```

### Méthodes Clés de Stabilisation

#### Experience Replay Buffer (Lin, 1992; Mnih et al., 2015)
Stocker les transitions $(s,a,r,s',\text{done})$ dans un buffer $\mathcal{D}$ et échantillonner des mini-batches aléatoires pour l'entraînement.

```{admonition} Bénéfices de l'Experience Replay
:class: tip
- **Brise la corrélation temporelle** : L'échantillonnage aléatoire décorrèle les données séquentielles
- **Efficacité des données** (data efficiency) : Chaque expérience peut être utilisée plusieurs fois
- **Stabilise l'apprentissage** : Lisse les changements dans la distribution des données
```

#### Target Network (Mnih et al., 2015)

Maintenir un réseau séparé $\theta^-$ pour calculer les cibles, mis à jour périodiquement :
- **Hard update** : $\theta^- \leftarrow \theta$ tous les $C$ pas
- **Soft update** : $\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$ avec $\tau \ll 1$

#### Techniques Additionnelles

```{list-table}
:header-rows: 1
:widths: 30 70

* - Technique
  - Objectif
* - Gradient clipping
  - Limiter la norme du gradient pour éviter l'explosion des gradients
* - Input normalization
  - Standardiser les observations d'état pour un entraînement stable
* - Reward scaling/clipping
  - Borner les magnitudes des récompenses à une plage raisonnable
* - Huber loss
  - Plus robuste aux outliers que le MSE
```

---

## 4. Algorithme DQN (Deep Q-Network)

```
Algorithme : DQN (Deep Q-Network)
──────────────────────────────────────────────────────────
Entrée : learning rate α, discount γ, exploration ε
        batch size B, buffer capacity N, update frequency C
Sortie : Q-network θ, policy π

// Initialisation
Initialiser replay buffer D avec capacité N (vide)
Initialiser Q-network avec poids aléatoires θ
Initialiser target network θ⁻ ← θ
step_count ← 0

// Apprentissage
répéter pour chaque épisode :
    Initialiser S
    
    répéter pour chaque step de l'épisode :
        
        // Sélection d'action (ε-greedy)
        avec probabilité ε :
            A ← action aléatoire
        sinon :
            A ← argmax_a' Q(S, a'; θ)
        
        // Interaction avec environnement
        Exécuter A, observer R, S', done
        
        // Stocker transition
        Stocker (S, A, R, S', done) dans D
        
        // Apprentissage (si assez de données)
        si |D| ≥ B alors :
            
            // Échantillonner mini-batch
            Échantillonner {(sⱼ, aⱼ, rⱼ, s'ⱼ, doneⱼ)} depuis D (taille B)
            
            // Calculer cibles
            pour chaque j dans batch faire :
                si doneⱼ alors :
                    yⱼ ← rⱼ
                sinon :
                    yⱼ ← rⱼ + γ·max_a' Q(s'ⱼ, a'; θ⁻)
                fin si
            fin pour
            
            // Mise à jour par gradient descent
            L(θ) ← (1/B)·Σⱼ (yⱼ - Q(sⱼ, aⱼ; θ))²
            θ ← θ - α·∇_θ L(θ)
        
        fin si
        
        // Mise à jour target network
        step_count ← step_count + 1
        si step_count mod C = 0 alors :
            θ⁻ ← θ
        fin si
        
        // Transition
        S ← S'
        
        // Décroissance exploration (optionnel)
        ε ← max(ε_min, ε·decay_rate)
        
    jusqu'à done = vrai
    
jusqu'à convergence ou nombre max d'épisodes

// Extraire policy
π(s) ← argmax_a Q(s, a; θ)

retourner θ, π
```

---

## 5. Considérations Théoriques & Bonnes Pratiques

### The Deadly Triad (Triade Mortelle)

```{admonition} Deadly Triad (Sutton & Barto)
:class: danger
La combinaison de ces trois éléments peut causer instabilité ou divergence :
1. **Function approximation** (surtout non-linéaire)
2. **Bootstrapping** (utiliser des valeurs estimées comme cibles, ex. apprentissage TD)
3. **Off-policy learning** (apprendre d'une politique différente de celle évaluée)

Quand les trois sont présents, la convergence n'est pas garantie !
```

### Trade-off Biais-Variance

```{list-table}
:header-rows: 1
:widths: 33 33 33

* - Problème
  - Cause
  - Solution
* - Underfitting (biais élevé)
  - Approximateur trop simple
  - Augmenter capacité, meilleures features
* - Overfitting (variance élevée)
  - Mémorise sans généraliser
  - Régularisation, données plus diverses
* - Instabilité
  - Les trois de la deadly triad
  - Utiliser techniques de stabilisation
```

---

## Exercices

### Exercice 1 : Approximation Linéaire

**Question** : Considérez un espace d'états 1D simple où $s \in [0, 1]$. Vous utilisez des features polynomiales : $\phi(s) = [1, s, s^2]^T$ et paramètres $\theta = [2, -3, 1]^T$. 

(a) Quelle est la valeur de $\hat{V}(s=0.5; \theta)$ ?

(b) Si vous observez une transition $(s=0.5, r=1.5, s'=0.6)$ avec $\gamma=0.9$ et learning rate $\alpha=0.1$, calculez le nouveau $\theta$ après une mise à jour TD(0).

```{admonition} Solution
:class: dropdown

**(a)** Calcul de $\hat{V}(0.5; \theta)$ :

$$\hat{V}(0.5; \theta) = \theta^T \phi(0.5) = [2, -3, 1]^T \cdot [1, 0.5, 0.25]^T$$
$$= 2(1) + (-3)(0.5) + 1(0.25) = 2 - 1.5 + 0.25 = 0.75$$

**(b)** Mise à jour TD(0) :

D'abord, calculer $\hat{V}(0.6; \theta)$ :
$$\phi(0.6) = [1, 0.6, 0.36]^T$$
$$\hat{V}(0.6; \theta) = 2(1) + (-3)(0.6) + 1(0.36) = 2 - 1.8 + 0.36 = 0.56$$

Erreur TD :
$$\delta = r + \gamma \hat{V}(s'; \theta) - \hat{V}(s; \theta)$$
$$\delta = 1.5 + 0.9(0.56) - 0.75 = 1.5 + 0.504 - 0.75 = 1.254$$

Mise à jour :
$$\theta \leftarrow \theta + \alpha \delta \phi(s)$$
$$\theta \leftarrow [2, -3, 1]^T + 0.1(1.254)[1, 0.5, 0.25]^T$$
$$\theta \leftarrow [2, -3, 1]^T + [0.1254, 0.0627, 0.03135]^T$$
$$\theta \leftarrow [2.1254, -2.9373, 1.03135]^T$$
```

---

### Exercice 2 : Semi-Gradient vs Vrai Gradient

**Question** : Expliquez pourquoi la mise à jour TD(0) pour l'approximation de fonctions est appelée "semi-gradient" plutôt que vraie descente de gradient. Qu'est-ce qui serait différent si on utilisait le vrai gradient de l'erreur TD quadratique ?

```{admonition} Solution
:class: dropdown

La mise à jour TD(0) est appelée "semi-gradient" car elle ne prend le gradient que par rapport à l'estimation de valeur courante $\hat{V}(S_t;\theta)$, tout en traitant la valeur de l'état suivant $\hat{V}(S_{t+1};\theta)$ comme une constante.

**Semi-gradient TD(0) :**
$$\theta \leftarrow \theta + \alpha [R_{t+1} + \gamma \hat{V}(S_{t+1};\theta) - \hat{V}(S_t;\theta)] \nabla_\theta \hat{V}(S_t;\theta)$$

Le **vrai gradient** dériverait l'erreur quadratique entière :
$$L(\theta) = \frac{1}{2}[R_{t+1} + \gamma \hat{V}(S_{t+1};\theta) - \hat{V}(S_t;\theta)]^2$$

Le vrai gradient serait :
$$\nabla_\theta L(\theta) = [R_{t+1} + \gamma \hat{V}(S_{t+1};\theta) - \hat{V}(S_t;\theta)][\gamma \nabla_\theta \hat{V}(S_{t+1};\theta) - \nabla_\theta \hat{V}(S_t;\theta)]$$

Notez le terme additionnel $\gamma \nabla_\theta \hat{V}(S_{t+1};\theta)$ qui prend en compte comment $\theta$ affecte la cible.

**Pourquoi utiliser semi-gradient ?**
- Computationnellement plus simple (pas besoin de dériver à travers la cible)
- Souvent plus stable en pratique
- Converge sous certaines conditions (cas linéaire avec apprentissage on-policy)
- Le vrai gradient correspond aux algorithmes de gradient résiduel, qui peuvent être plus lents
```

---

### Exercice 3 : Analyse de l'Experience Replay

**Question** : Vous avez un replay buffer de taille $N=1000$ et utilisez une batch size de 32. Après que le buffer soit plein, approximativement combien de fois chaque transition sera utilisée pour l'entraînement en moyenne avant d'être remplacée (en supposant un échantillonnage uniforme) ?

Si les transitions restent dans le buffer pendant 1000 pas de temps en moyenne après avoir été ajoutées, et vous échantillonnez un batch à chaque pas de temps, calculez le nombre attendu de fois que chaque transition est échantillonnée.

```{admonition} Solution
:class: dropdown

**Configuration :**
- Capacité du buffer : $N = 1000$
- Batch size : $B = 32$
- Durée de vie d'une transition dans le buffer : $L = 1000$ pas de temps (en moyenne)

**Analyse :**

À chaque pas de temps (après que le buffer soit plein) :
- 1 nouvelle transition est ajoutée
- 1 ancienne transition est retirée
- 1 batch de taille $B = 32$ est échantillonné

Pour une seule transition qui reste dans le buffer pendant $L$ pas de temps :
- Nombre d'événements d'échantillonnage : $L = 1000$
- Probabilité d'être sélectionnée en un tirage : $\frac{1}{N} = \frac{1}{1000}$
- Nombre attendu de fois sélectionnée par batch : $B \cdot \frac{1}{N} = \frac{32}{1000} = 0.032$

**Nombre attendu de fois échantillonnée :**
$$\text{Échantillonnages attendus} = L \cdot B \cdot \frac{1}{N} = 1000 \cdot \frac{32}{1000} = 32$$

**Interprétation :** En moyenne, chaque transition est utilisée pour l'entraînement **32 fois** avant d'être évincée du buffer. Cela démontre le bénéfice d'efficacité des données de l'experience replay—chaque expérience est réutilisée 32 fois plutôt qu'une seule fois (comme dans l'apprentissage en ligne).

**Note :** Cela suppose un échantillonnage aléatoire uniforme. L'échantillonnage basé sur les priorités (prioritized experience replay) changerait cette distribution.
```

---

### Exercice 4 : Impact du Target Network

**Question** : Considérez un scénario simple où la vraie Q-value est $Q^*(s,a) = 10$, mais votre estimation actuelle est $Q(s,a;\theta) = 2$. Vous recevez une récompense $r=1$ et l'état suivant est terminal.

(a) Sans target network, quelle est la cible $y$ et l'erreur TD ?

(b) Maintenant supposez que vous avez un target network où $Q(s,a;\theta^-) = 2$ (même que le courant) mais après entraînement pendant un moment, le réseau en ligne s'améliore à $Q(s,a;\theta) = 6$. Le target network n'a pas encore été mis à jour. Quelle est la nouvelle erreur TD ?

(c) Expliquez comment le target network aide à stabiliser l'apprentissage dans ce cas.

```{admonition} Solution
:class: dropdown

**(a) Sans target network :**

Puisque l'état suivant est terminal :
$$y = r = 1$$
$$\text{Erreur TD} = y - Q(s,a;\theta) = 1 - 2 = -1$$

L'erreur négative va pousser $Q(s,a;\theta)$ vers le bas vers 1, ce qui s'éloigne en fait de la vraie valeur de 10 !

**(b) Avec target network :**

La cible est toujours calculée en utilisant $\theta^-$ :
$$y = r = 1$$ (état terminal, même qu'avant)

Mais maintenant avec le réseau en ligne amélioré :
$$\text{Erreur TD} = y - Q(s,a;\theta) = 1 - 6 = -5$$

L'erreur est maintenant plus grande en magnitude parce que le réseau en ligne a plus divergé de la cible (incorrecte).

**(c) Mécanisme de stabilisation :**

Le target network aide à stabiliser l'apprentissage par :

1. **Prévenir les cibles mouvantes** : Sans target network, améliorer $Q(s,a;\theta)$ changerait immédiatement les cibles pour les paires état-action liées, créant un effet de "poursuite".

2. **Lisser les mises à jour** : En mettant à jour $\theta^-$ seulement périodiquement, on donne au réseau en ligne le temps d'apprendre d'un ensemble cohérent de cibles avant que les cibles ne changent.

3. **Briser les boucles de feedback nuisibles** : Si le réseau en ligne fait une erreur (ex. surestimer certaines Q-values), cette erreur ne se propagera pas immédiatement pour devenir la cible d'autres mises à jour.

**Dans cet exemple spécifique** : Le target network nous donne en fait une mauvaise cible (1 au lieu de 10), mais au moins elle est cohérente. En pratique, sur de nombreuses mises à jour avec diverses expériences, le réseau convergera finalement plus près des vraies valeurs. Sans le target network, l'instabilité des cibles constamment changeantes pourrait empêcher toute convergence.

**Note importante** : Les target networks ne garantissent pas une convergence correcte—ils améliorent juste la stabilité. La qualité de la convergence dépend toujours de l'exploration, du design des récompenses et d'autres facteurs.
```

---

### Exercice 5 : Concevoir des Features pour l'Approximation Linéaire

**Question** : Vous construisez un agent RL pour une tâche de navigation de robot 2D. L'état est $s = (x, y, \theta, v)$ où $(x,y)$ est la position, $\theta$ est l'angle de cap (heading), et $v$ est la vitesse. Le but est à la position $(x_g, y_g)$.

Concevez un vecteur de features $\phi(s)$ pour l'approximation linéaire qui aiderait l'agent à apprendre une bonne fonction de valeur. Justifiez chaque feature que vous incluez.

```{admonition} Solution
:class: dropdown

**Vecteur de features proposé** $\phi(s)$ :

1. **Terme de biais** (bias term) : $\phi_1 = 1$
   - Permet à la fonction de valeur d'avoir une baseline non-nulle
   
2. **Distance au but** : $\phi_2 = \sqrt{(x-x_g)^2 + (y-y_g)^2}$
   - Mesure directe du progrès ; plus proche = valeur plus élevée
   
3. **Distance au carré au but** : $\phi_3 = (x-x_g)^2 + (y-y_g)^2$
   - Capture la relation non-linéaire ; permet une croissance plus rapide en s'approchant
   
4. **Alignement du cap** (heading alignment) : $\phi_4 = \cos(\theta - \theta_{\text{goal}})$
   - Où $\theta_{\text{goal}} = \text{atan2}(y_g - y, x_g - x)$
   - Mesure si le robot fait face vers le but (+1 = aligné, -1 = opposé)
   
5. **Vitesse** : $\phi_5 = v$
   - Se déplacer peut être récompensé ou pénalisé selon la tâche
   
6. **Vitesse × alignement du cap** : $\phi_6 = v \cdot \cos(\theta - \theta_{\text{goal}})$
   - Capture la "vitesse utile" (se déplacer vers le but vs. s'en éloigner)
   
7. **Features de position** (si l'environnement a une structure) :
   - $\phi_7 = x$, $\phi_8 = y$
   - Utile si différentes régions ont différentes valeurs (ex. obstacles)

8. **Features indicatrices** (si nécessaire) :
   - $\phi_9 = \mathbb{1}[\text{distance} < \epsilon]$ pour région proche du but
   - $\phi_{10} = \mathbb{1}[\text{dans région obstacle}]$

**Vecteur de features final :**
$$\phi(s) = [1, d, d^2, \cos(\Delta\theta), v, v\cos(\Delta\theta), x, y, \mathbb{1}_{\text{proche}}, \mathbb{1}_{\text{obstacle}}]^T$$

où $d$ est la distance au but et $\Delta\theta$ est l'erreur de cap.

**Justification :**
- **Connaissance du domaine** : Les features encodent ce qu'on sait être important (distance, alignement, vitesse)
- **Non-linéarité** : Inclure $d^2$ et des produits comme $v\cos(\Delta\theta)$ ajoute du pouvoir expressif
- **Normalisation** : En pratique, normaliser les features à des échelles similaires (ex. distance par distance max)
- **Dimensionnalité** : 10 features est gérable ; des environnements plus complexes pourraient nécessiter 50-100+

**Test** : Après implémentation, vérifier si :
- Les valeurs sont plus élevées près du but (tester sur des états construits à la main)
- Le gradient pointe dans des directions sensées
- L'apprentissage converge sur des scénarios simples
```

---

## Points Clés à Retenir

```{admonition} Concepts Fondamentaux
:class: important

**1. Nécessité de l'Approximation**
- Les espaces d'états continus ou de grande dimension rendent les méthodes tabulaires impossibles
- L'approximation de fonctions permet la **généralisation** : transférer l'apprentissage vers des états non-visités
- Compromis inévitable entre expressivité et stabilité

**2. Approximation Linéaire vs Non-Linéaire**
- **Linéaire** : Stable, convergence garantie (sous conditions), mais nécessite un bon feature engineering
- **Non-linéaire (Deep RL)** : Très expressif, apprend ses propres features, mais instable et difficile à optimiser

**3. La Deadly Triad**
- Combinaison dangereuse : Function approximation + Bootstrapping + Off-policy learning
- Peut mener à la divergence sans techniques de stabilisation appropriées
- Comprendre cette triade aide à anticiper et résoudre les problèmes d'instabilité
```

```{admonition} Techniques Pratiques Essentielles
:class: tip

**Stabilisation en Deep RL** (indispensables pour DQN et variantes) :
- **Experience replay** : Brise les corrélations temporelles et améliore l'efficacité des données
- **Target network** : Stabilise les cibles d'apprentissage en les maintenant fixes temporairement
- **Gradient clipping** : Prévient l'explosion des gradients
- **Normalisation** : Des inputs et des récompenses pour un apprentissage stable

**Pipeline de Développement** :
1. Commencer avec des environnements simples (CartPole, MountainCar)
2. Tester d'abord l'approximation linéaire pour valider le pipeline
3. Monitorer les bonnes métriques (erreur TD, Q-values, retours)
4. Augmenter la complexité progressivement seulement si nécessaire
```

```{admonition} Trade-offs et Décisions de Design
:class: note

**Choix de l'Approximateur** :
- Linéaire si : espace d'états petit/moyen, features bien comprises, besoin de garanties théoriques
- Non-linéaire si : espace d'états très grand/continu, patterns complexes, données abondantes

**Hyperparamètres Critiques** :
- **Learning rate α** : Impact majeur sur convergence (trop grand → instabilité, trop petit → lent)
- **Replay buffer** : Équilibre mémoire vs. diversité des données
- **Target update frequency** : Équilibre stabilité vs. réactivité aux changements

**Debugging** :
- Si divergence : Réduire α, augmenter fréquence de mise à jour du target, vérifier échelle des récompenses
- Si apprentissage trop lent : Augmenter α, vérifier exploration, améliorer features/architecture
- Si overfitting : Augmenter taille du buffer, ajouter régularisation, diversifier données
```

```{admonition} Applications et Limitations
:class: warning

**Quand Utiliser Function Approximation** :
- ✓ Espaces d'états continus (robotique, contrôle)
- ✓ Espaces d'états discrets mais très grands (jeux complexes, planification)
- ✓ Besoin de généralisation entre états similaires
- ✓ États représentés par images ou signaux haute dimension

**Limitations à Connaître** :
- ✗ Pas de garantie de convergence (surtout avec deadly triad)
- ✗ Sensibilité aux hyperparamètres
- ✗ Temps d'entraînement long pour deep RL
- ✗ Peut nécessiter beaucoup de données pour bien généraliser
- ✗ Interprétabilité réduite (surtout réseaux profonds)

**En Pratique** :
- Toujours comparer avec baseline simple (random policy, heuristique)
- Utiliser plusieurs seeds aléatoires pour évaluer la robustesse
- Documenter tous les hyperparamètres et choix d'architecture
- Valider sur environnements de test différents de l'entraînement
```

```{admonition} Formules Clés à Retenir
:class: seealso

**Approximation Linéaire** :
$$\hat{V}(s;\theta) = \theta^T \phi(s)$$
$$\theta \leftarrow \theta + \alpha \delta_t \phi(s_t) \quad \text{où} \quad \delta_t = r_{t+1} + \gamma \hat{V}(s_{t+1};\theta) - \hat{V}(s_t;\theta)$$

**Deep RL (DQN)** :
$$L(\theta) = \mathbb{E}_{(s,a,r,s')\sim \mathcal{D}}\left[ \left(r + \gamma \max_{a'} Q(s',a';\theta^-) - Q(s,a;\theta)\right)^2 \right]$$

**Mise à Jour Gradient** :
$$\theta \leftarrow \theta + \alpha (y - Q(s,a;\theta)) \nabla_\theta Q(s,a;\theta)$$

**Target Network Update** :
- Hard : $\theta^- \leftarrow \theta$ tous les $C$ pas
- Soft : $\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$ avec $\tau \ll 1$
```