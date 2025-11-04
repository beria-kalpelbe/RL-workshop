# Monte Carlo Methods

Les méthodes **Monte Carlo (MC)** estiment les value functions en moyennant les **returns observés** sur des épisodes complets, sans nécessiter de modèle du MDP.

---

## 1. Principe fondamental

```{admonition} Différence clé avec Dynamic Programming
:class: important

- **DP** : utilise le modèle $(P, R)$ et les équations de Bellman pour calculer les valeurs
- **MC** : apprend directement des **expériences** (échantillons de trajectoires)

**Idée centrale :** La value function $V^{\pi}(s)$ est définie comme l'espérance du return :

$$V^{\pi}(s) = \mathbb{E}_{\pi}[G_t | S_t = s]$$

Monte Carlo **estime cette espérance** par la **moyenne empirique** des returns observés :

$$V^{\pi}(s) \approx \frac{1}{N(s)} \sum_{i=1}^{N(s)} G_t^{(i)}$$

où $N(s)$ est le nombre de visites de l'état $s$ et $G_t^{(i)}$ le return observé à la $i$-ème visite.
```

```{admonition} Conditions d'application
:class: note

1. **Épisodes complets** : Le MDP doit être **épisodique** (termine en temps fini)
2. **Génération de trajectoires** : Capacité à générer des épisodes selon une policy
3. **Exploration** : Tous les états doivent être visités (hypothèse d'exploration)

**Avantage majeur :** Pas besoin du modèle $(P, R)$ → **model-free**
```

---

## 2. First-visit vs Every-visit Monte Carlo

Il existe deux variantes pour compter les visites d'un état dans un épisode :

### 2.1 First-visit MC

```{admonition} First-visit Monte Carlo
:class: note

Pour chaque épisode :
- On ne compte que la **première occurrence** de chaque état
- Si un état apparaît plusieurs fois, seule la première visite contribue à l'estimation

**Propriété :** Les returns de différents épisodes sont **indépendants** (même état, épisodes différents)

**Convergence :** Par la loi des grands nombres, $V(s) \to V^{\pi}(s)$ quand $N(s) \to \infty$
```

### 2.2 Every-visit MC

```{admonition} Every-visit Monte Carlo
:class: note

Pour chaque épisode :
- On compte **toutes les occurrences** de chaque état
- Chaque visite contribue à l'estimation

**Propriété :** Les returns d'un même épisode sont **corrélés**

**Convergence :** $V(s) \to V^{\pi}(s)$ asymptotiquement (convergence garantie mais avec corrélation)
```

### 2.3 Comparaison

| Critère | First-visit | Every-visit |
|---------|-------------|-------------|
| **Comptage** | Première occurrence seulement | Toutes les occurrences |
| **Indépendance** | Returns indépendants entre épisodes | Returns corrélés dans un épisode |
| **Convergence** | Loi des grands nombres directe | Convergence asymptotique |
| **Variance** | Plus faible (moins de samples corrélés) | Peut être plus élevée |
| **Usage pratique** | Plus courant | Parfois meilleure convergence |

---

## 3. Algorithmes

### 3.1 First-visit MC Prediction

```
Algorithme : First-visit MC Prediction
────────────────────────────────────────────────
Entrée : policy π à évaluer
Sortie : V^π estimée

// Initialisation
pour chaque s ∈ S faire :
    V(s) ← 0
    Returns(s) ← liste vide
fin pour

// Générer et analyser des épisodes
répéter pour chaque épisode :
    
    // Générer un épisode selon π
    Générer épisode : S₀, A₀, R₁, S₁, A₁, R₂, ..., Sₜ₋₁, Aₜ₋₁, Rₜ, Sₜ
    
    // Calculer le return pour chaque step
    G ← 0
    
    pour t = T-1, T-2, ..., 0 faire :
        G ← γ·G + R_{t+1}
        
        // Vérifier si première visite
        si S_t ∉ {S₀, S₁, ..., S_{t-1}} alors :
            Ajouter G à Returns(S_t)
            V(S_t) ← moyenne(Returns(S_t))
        fin si
    fin pour
    
jusqu'à convergence

retourner V
```

### 3.2 MC Control avec ε-greedy

Pour apprendre une policy optimale, on estime $Q^{\pi}(s,a)$ et on améliore la policy :

```
Algorithme : Monte Carlo Control (ε-greedy)
────────────────────────────────────────────────
Entrée : paramètre ε > 0
Sortie : policy π ≈ π*, Q ≈ Q*

// Initialisation
pour chaque s ∈ S, a ∈ A faire :
    Q(s,a) ← 0
    Returns(s,a) ← liste vide
    π(a|s) ← policy ε-greedy basée sur Q
fin pour

// Boucle d'amélioration
répéter pour chaque épisode :
    
    // Générer épisode selon π
    Générer : S₀, A₀, R₁, S₁, A₁, R₂, ..., Sₜ
    
    // Calculer returns
    G ← 0
    pour t = T-1, ..., 0 faire :
        G ← γ·G + R_{t+1}
        
        si (S_t, A_t) est première visite alors :
            Ajouter G à Returns(S_t, A_t)
            Q(S_t, A_t) ← moyenne(Returns(S_t, A_t))
            
            // Amélioration ε-greedy
            a* ← argmax_a Q(S_t, a)
            pour chaque a ∈ A faire :
                si a = a* alors :
                    π(a|S_t) ← 1 - ε + ε/|A|
                sinon :
                    π(a|S_t) ← ε/|A|
                fin si
            fin pour
        fin si
    fin pour
    
jusqu'à convergence

retourner π, Q
```

---

## 4. Incremental implementation

Au lieu de stocker tous les returns, on peut mettre à jour incrémentalement :

```{admonition} Mise à jour incrémentale
:class: important

Après avoir observé le return $G$ pour l'état $s$ :

$$V(s) \leftarrow V(s) + \alpha [G - V(s)]$$

où $\alpha \in (0,1]$ est le **learning rate** (taux d'apprentissage).

**Avantage :** Pas besoin de stocker l'historique complet des returns.

**Équivalence :** Pour $\alpha = \frac{1}{N(s)}$, c'est équivalent à la moyenne arithmétique.

**Pratique courante :** Utiliser un $\alpha$ constant (e.g., 0.1) pour donner plus de poids aux observations récentes.
```

---

## 5. Off-policy Monte Carlo

### 5.1 Principe

```{admonition} Off-policy learning
:class: note

On veut estimer $V^{\pi}$ ou $Q^{\pi}$ (policy **cible**) en utilisant des trajectoires générées par une policy différente $b$ (policy **comportement**).

**Motivation :**
- Explorer largement avec $b$ (e.g., uniforme)
- Apprendre une policy optimale $\pi$ (déterministe, greedy)

**Exigence :** $\pi(a|s) > 0 \Rightarrow b(a|s) > 0$ (hypothèse de **coverage**)
```

### 5.2 Importance Sampling

```{admonition} Importance sampling ratio
:class: important

Pour corriger le biais dû à l'utilisation de $b$ au lieu de $\pi$, on pondère par :

$$\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(A_k|S_k)}{b(A_k|S_k)}$$

**Ordinary importance sampling :**

$$V^{\pi}(s) = \mathbb{E}_b[\rho_{t:T-1} G_t | S_t = s]$$

Estimateur :

$$V(s) = \frac{1}{N(s)} \sum_{i=1}^{N(s)} \rho_{t:T-1}^{(i)} G_t^{(i)}$$

**Weighted importance sampling :**

$$V(s) = \frac{\sum_{i=1}^{N(s)} \rho_{t:T-1}^{(i)} G_t^{(i)}}{\sum_{i=1}^{N(s)} \rho_{t:T-1}^{(i)}}$$

**Propriété :** Weighted IS a généralement **moins de variance** que ordinary IS.
```

### 5.3 Algorithme Off-policy MC

```
Algorithme : Off-policy MC Prediction
────────────────────────────────────────────────
Entrée : policy cible π, policy comportement b
Sortie : V ≈ V^π

// Initialisation
pour chaque s ∈ S faire :
    V(s) ← 0
    C(s) ← 0  // Somme des poids cumulés
fin pour

répéter pour chaque épisode :
    
    Générer épisode selon b : S₀, A₀, R₁, ..., Sₜ
    G ← 0
    W ← 1  // Importance sampling ratio
    
    pour t = T-1, ..., 0 faire :
        G ← γ·G + R_{t+1}
        C(S_t) ← C(S_t) + W
        V(S_t) ← V(S_t) + (W/C(S_t))[G - V(S_t)]
        
        W ← W · π(A_t|S_t) / b(A_t|S_t)
        
        si W = 0 alors :
            sortir de la boucle  // Trajectoire incompatible
        fin si
    fin pour
    
jusqu'à convergence

retourner V
```

---

## 6. Exercices théoriques

### Exercice 1 — Calcul de returns

```{admonition} Exercice
:class: tip

Un agent génère l'épisode suivant :

$$S_0 \xrightarrow{A_0, R_1=2} S_1 \xrightarrow{A_1, R_2=5} S_2 \xrightarrow{A_2, R_3=-1} S_3 \xrightarrow{A_3, R_4=10} S_4$$

avec $S_0 = s_A$, $S_1 = s_B$, $S_2 = s_A$, $S_3 = s_C$, $S_4 = \text{terminal}$

**Paramètres :** $\gamma = 0.9$

**Questions :**

a) Calculez le return $G_0$ depuis le début de l'épisode

b) Calculez le return $G_2$ depuis le second état $s_A$

c) Pour **first-visit MC**, quels returns seraient utilisés pour mettre à jour $V(s_A)$ ?

d) Pour **every-visit MC**, quels returns seraient utilisés pour mettre à jour $V(s_A)$ ?
```

````{dropdown} Solution — Exercice 1

**a) Return depuis $t=0$ :**

$$G_0 = R_1 + \gamma R_2 + \gamma^2 R_3 + \gamma^3 R_4$$

$$G_0 = 2 + 0.9 \times 5 + (0.9)^2 \times (-1) + (0.9)^3 \times 10$$

$$G_0 = 2 + 4.5 - 0.81 + 7.29 = 12.98$$

**b) Return depuis $t=2$ (seconde visite de $s_A$) :**

$$G_2 = R_3 + \gamma R_4$$

$$G_2 = -1 + 0.9 \times 10 = -1 + 9 = 8$$

**c) First-visit MC pour $s_A$ :**

État $s_A$ apparaît à $t=0$ (première fois) et $t=2$ (deuxième fois).

First-visit ne compte que la **première occurrence** : $t=0$

**Return utilisé :** $G_0 = 12.98$

Mise à jour : $V(s_A) \leftarrow \text{moyenne des } G_0 \text{ sur tous les épisodes}$

**d) Every-visit MC pour $s_A$ :**

Every-visit compte **toutes les occurrences** : $t=0$ et $t=2$

**Returns utilisés :** $G_0 = 12.98$ et $G_2 = 8.0$

Mise à jour : moyenne de tous les returns observés depuis $s_A$ dans cet épisode et les autres.

Pour cet épisode seul : $\frac{12.98 + 8.0}{2} = 10.49$
````

---

### Exercice 2 — Mise à jour incrémentale

```{admonition} Exercice
:class: tip

On utilise MC avec mise à jour incrémentale : $V(s) \leftarrow V(s) + \alpha[G - V(s)]$

**État initial :** $V(s_1) = 5.0$, learning rate $\alpha = 0.2$

**Observations successives :**
1. Premier épisode : return observé $G = 12.0$
2. Deuxième épisode : return observé $G = 8.0$
3. Troisième épisode : return observé $G = 10.0$

**Calculez** la valeur de $V(s_1)$ après chaque mise à jour.
```

````{dropdown} Solution — Exercice 2

**Initial :** $V(s_1) = 5.0$

**Après épisode 1 :** $G = 12.0$

$$V(s_1) \leftarrow 5.0 + 0.2 \times [12.0 - 5.0]$$

$$V(s_1) \leftarrow 5.0 + 0.2 \times 7.0 = 5.0 + 1.4 = 6.4$$

**Après épisode 2 :** $G = 8.0$

$$V(s_1) \leftarrow 6.4 + 0.2 \times [8.0 - 6.4]$$

$$V(s_1) \leftarrow 6.4 + 0.2 \times 1.6 = 6.4 + 0.32 = 6.72$$

**Après épisode 3 :** $G = 10.0$

$$V(s_1) \leftarrow 6.72 + 0.2 \times [10.0 - 6.72]$$

$$V(s_1) \leftarrow 6.72 + 0.2 \times 3.28 = 6.72 + 0.656 = 7.376$$

**Valeurs successives :**
- Après épisode 1 : $V(s_1) = 6.4$
- Après épisode 2 : $V(s_1) = 6.72$
- Après épisode 3 : $V(s_1) = 7.376$

**Observation :** La valeur converge progressivement vers la moyenne des returns (ici ≈ 10).
````

---

### Exercice 3 — First-visit vs Every-visit

```{admonition} Exercice
:class: tip

On observe 2 épisodes pour un état $s$ :

**Épisode 1 :**
- État $s$ visité aux temps $t=0$ et $t=3$
- Returns : $G_0 = 15$, $G_3 = 6$

**Épisode 2 :**
- État $s$ visité au temps $t=1$
- Return : $G_1 = 10$

**Questions :**

a) Quelle est l'estimation $V(s)$ avec **first-visit MC** ?

b) Quelle est l'estimation $V(s)$ avec **every-visit MC** ?

c) Si la vraie valeur est $V^{\pi}(s) = 11$, quelle méthode donne la meilleure estimation ?
```

````{dropdown} Solution — Exercice 3

**a) First-visit MC :**

On ne compte que la **première visite** de chaque épisode :
- Épisode 1 : première visite à $t=0$ → $G_0 = 15$
- Épisode 2 : première (et unique) visite à $t=1$ → $G_1 = 10$

$$V_{\text{first}}(s) = \frac{15 + 10}{2} = \frac{25}{2} = 12.5$$

**b) Every-visit MC :**

On compte **toutes les visites** :
- Épisode 1 : visites à $t=0$ et $t=3$ → $G_0 = 15$, $G_3 = 6$
- Épisode 2 : visite à $t=1$ → $G_1 = 10$

$$V_{\text{every}}(s) = \frac{15 + 6 + 10}{3} = \frac{31}{3} \approx 10.33$$

**c) Comparaison avec $V^{\pi}(s) = 11$ :**

- First-visit : $|12.5 - 11| = 1.5$
- Every-visit : $|10.33 - 11| = 0.67$

Dans cet exemple, **every-visit MC** donne une meilleure estimation.

**Note :** Ceci dépend des données ; asymptotiquement les deux convergent vers la vraie valeur.
````

---

### Exercice 4 — Importance Sampling

```{admonition} Exercice
:class: tip

Un agent génère un épisode selon une policy comportement $b$ :

$$S_0 \xrightarrow{A_0} S_1 \xrightarrow{A_1} S_2 \text{ (terminal)}$$

**Actions choisies :** $A_0 = a_1$, $A_1 = a_2$

**Rewards :** $R_1 = 5$, $R_2 = 3$, discount $\gamma = 1.0$

**Policies :**

| État | Action | $\pi$ (cible) | $b$ (comportement) |
|------|--------|---------------|---------------------|
| $S_0$ | $a_1$ | 0.8 | 0.5 |
| $S_0$ | $a_2$ | 0.2 | 0.5 |
| $S_1$ | $a_1$ | 0.3 | 0.6 |
| $S_1$ | $a_2$ | 0.7 | 0.4 |

**Questions :**

a) Calculez le return $G_0$

b) Calculez l'importance sampling ratio $\rho_{0:1}$

c) Quelle est la contribution pondérée pour estimer $V^{\pi}(S_0)$ ?

d) Si $\rho$ était très grand (e.g., 100), quel problème cela poserait-il ?
```

````{dropdown} Solution — Exercice 4

**a) Return depuis $t=0$ :**

$$G_0 = R_1 + \gamma R_2 = 5 + 1.0 \times 3 = 8$$

**b) Importance sampling ratio :**

$$\rho_{0:1} = \prod_{t=0}^{1} \frac{\pi(A_t|S_t)}{b(A_t|S_t)}$$

$$= \frac{\pi(a_1|S_0)}{b(a_1|S_0)} \times \frac{\pi(a_2|S_1)}{b(a_2|S_1)}$$

$$= \frac{0.8}{0.5} \times \frac{0.7}{0.4}$$

$$= 1.6 \times 1.75 = 2.8$$

**c) Contribution pondérée :**

Pour **ordinary importance sampling** :

$$\text{Contribution} = \rho_{0:1} \times G_0 = 2.8 \times 8 = 22.4$$

Pour **weighted importance sampling**, on diviserait par la somme des poids.

**d) Problème avec $\rho$ très grand :**

Si $\rho = 100$, la contribution serait $100 \times 8 = 800$ !

**Problèmes :**
- **Variance très élevée** : une seule trajectoire domine l'estimation
- **Instabilité** : estimations très sensibles aux trajectoires rares
- **Convergence lente** : beaucoup de trajectoires nécessaires pour stabiliser

**Solution :** Utiliser weighted importance sampling ou limiter les ratios (capping).
````

---

### Exercice 5 — MC Control avec ε-greedy

```{admonition} Exercice
:class: tip

On apprend une policy avec MC Control et stratégie ε-greedy ($\varepsilon = 0.2$, $|A| = 3$ actions).

**Q-values actuelles pour état $s$ :**
- $Q(s, a_1) = 8.0$
- $Q(s, a_2) = 12.0$
- $Q(s, a_3) = 5.0$

**Questions :**

a) Quelle est l'action greedy $a^*$ ?

b) Calculez $\pi(a_1|s)$, $\pi(a_2|s)$, $\pi(a_3|s)$ sous la policy ε-greedy

c) Vérifiez que $\sum_a \pi(a|s) = 1$

d) Quelle est la valeur espérée $V^{\pi}(s) = \sum_a \pi(a|s) Q(s,a)$ ?
```

````{dropdown} Solution — Exercice 5

**a) Action greedy :**

$$a^* = \arg\max_a Q(s,a) = a_2 \quad \text{(car } 12.0 > 8.0 > 5.0\text{)}$$

**b) Probabilités ε-greedy :**

La policy ε-greedy attribue :
- Probabilité $1 - \varepsilon + \frac{\varepsilon}{|A|}$ à l'action greedy
- Probabilité $\frac{\varepsilon}{|A|}$ aux autres actions

Avec $\varepsilon = 0.2$ et $|A| = 3$ :

$$\pi(a_2|s) = 1 - 0.2 + \frac{0.2}{3} = 0.8 + 0.0667 = 0.8667$$

$$\pi(a_1|s) = \frac{0.2}{3} = 0.0667$$

$$\pi(a_3|s) = \frac{0.2}{3} = 0.0667$$

**c) Vérification :**

$$\sum_a \pi(a|s) = 0.8667 + 0.0667 + 0.0667 = 1.0001 \approx 1$$ ✓

(Petite erreur d'arrondi acceptable)

**d) Valeur espérée sous $\pi$ :**

$$V^{\pi}(s) = \sum_a \pi(a|s) Q(s,a)$$

$$= 0.0667 \times 8.0 + 0.8667 \times 12.0 + 0.0667 \times 5.0$$

$$= 0.534 + 10.40 + 0.334$$

$$= 11.27$$

La policy ε-greedy atteint environ 94% de la valeur optimale (12.0) tout en maintenant l'exploration.
````

---

## 7. Avantages et limitations

### 7.1 Avantages

```{admonition} Points forts de Monte Carlo
:class: note

✅ **Model-free** : Pas besoin de $P(s'|s,a)$ ni $R(s,a)$

✅ **Estimation non biaisée** : Moyenne directe des returns observés

✅ **Simple conceptuellement** : Intuition claire (moyenne d'expériences)

✅ **Applicable aux MDPs avec modèle inconnu** : Apprentissage par interaction

✅ **Peut se concentrer sur états importants** : Pas besoin de balayer tout l'espace

✅ **Parallélisable** : Générer plusieurs épisodes indépendamment
```

### 7.2 Limitations

```{admonition} Contraintes de Monte Carlo
:class: warning

❌ **Épisodes complets requis** : Impossible pour tâches continues ou très longues

❌ **Variance élevée** : Les returns peuvent varier beaucoup entre épisodes

❌ **Convergence lente** : Beaucoup d'épisodes nécessaires pour estimation précise

❌ **Pas d'apprentissage online** : Doit attendre la fin de l'épisode

❌ **Problème d'exploration** : États rarement visités → estimation imprécise

❌ **Off-policy peut avoir variance explosive** : Importance sampling problématique
```

---

## 8. Points clés à retenir

```{admonition} Concepts essentiels
:class: important

1. **MC = apprentissage par expérience** sans modèle du MDP
2. **Return empirique** : Moyenne des $G_t$ observés pour estimer $V^{\pi}(s)$
3. **First-visit vs Every-visit** : Trade-off indépendance vs nombre de samples
4. **Mise à jour incrémentale** : $V(s) \leftarrow V(s) + \alpha[G - V(s)]$
5. **MC Control** : Alterner génération d'épisodes et amélioration ε-greedy
6. **Off-policy** : Importance sampling pour apprendre $\pi$ depuis trajectoires de $b$
7. **Épisodes complets** : Limitation principale de MC
8. **Variance élevée** : Convergence plus lente que méthodes bootstrapping
```

---

## 9. Comparaison avec autres méthodes

| Méthode | Modèle requis | Bootstrap | Épisodes complets | Variance | Biais |
|---------|---------------|-----------|-------------------|----------|-------|
| **DP** | ✅ Oui | ✅ Oui | ❌ Non | Faible | Aucun |
| **MC** | ❌ Non | ❌ Non | ✅ Oui | Élevée | Aucun |
| **TD** | ❌ Non | ✅ Oui | ❌ Non | Moyenne | Faible |

```{note}
**Bootstrapping** = utiliser des estimations pour mettre à jour d'autres estimations (ex: utiliser $V(s')$ pour estimer $V(s)$).

MC n'utilise **pas** de bootstrapping → attends les vrais returns $G_t$.
```

---

## 10. Extensions et variantes

```{admonition} Développements avancés
:class: note

- **Batch MC** : Mettre à jour toutes les valeurs après plusieurs épisodes
- **Monte Carlo Tree Search (MCTS)** : Planification avec simulations MC (AlphaGo)
- **Gradient MC** : Utiliser MC avec approximateurs de fonctions
- **Eligibility traces** : Combiner MC et TD (TD(λ), prochains chapitres)
- **Per-decision importance sampling** : Réduire variance en off-policy
```