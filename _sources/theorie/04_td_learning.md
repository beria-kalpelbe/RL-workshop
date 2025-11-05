# Temporal-Difference (TD) Learning

Les méthodes **Temporal-Difference (TD)** combinent les avantages de **Monte Carlo** (model-free, apprentissage par expérience) et de **Dynamic Programming** (bootstrapping, pas besoin d'épisodes complets).

---

## 1. Principe fondamental du TD Learning

```{important}
**L'idée centrale du TD**

Au lieu d'attendre le return complet $G_t$ (comme MC), TD utilise une **estimation bootstrap** :

$$\text{TD target} = R_{t+1} + \gamma V(S_{t+1})$$

**Mise à jour TD :**

$$V(S_t) \leftarrow V(S_t) + \alpha \underbrace{\left[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)\right]}_{\text{TD error } \delta_t}$$

**Comparaison :**

| Méthode | Cible (target) | Bootstrap | Épisode complet |
|---------|----------------|-----------|-----------------|
| **MC** | $G_t$ (return réel) | ❌ Non | ✅ Oui |
| **TD** | $R_{t+1} + \gamma V(S_{t+1})$ | ✅ Oui | ❌ Non |
| **DP** | $\sum_{s',r} P(s',r\|s,a)[r + \gamma V(s')]$ | ✅ Oui | ❌ Non |
```

### 1.1 TD Error

```{note}
**TD Error (erreur temporelle)**

$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

**Interprétation :**
- Si $\delta_t > 0$ : la reward + valeur future > estimation actuelle → **sous-estimation**
- Si $\delta_t < 0$ : la reward + valeur future < estimation actuelle → **surestimation**
- Le TD error est le **signal d'apprentissage** qui guide la mise à jour

**Propriété importante :** $\mathbb{E}[\delta_t] = 0$ à la convergence (quand $V = V^{\pi}$)
```

---

## 2. TD(0) — Prediction

### 2.1 Algorithme

```
Algorithme : TD(0) Prediction
────────────────────────────────────────────────
Entrée : policy π à évaluer, learning rate α
Sortie : V ≈ V^π

// Initialisation
pour chaque s ∈ S faire :
    V(s) ← 0  (ou valeurs arbitraires)
fin pour

// Apprentissage
répéter pour chaque épisode :
    Initialiser S (état de départ)
    
    répéter pour chaque step de l'épisode :
        A ← action donnée par π(·|S)
        Exécuter A, observer R, S'
        
        // Mise à jour TD(0)
        V(S) ← V(S) + α[R + γ·V(S') - V(S)]
        
        S ← S'
    jusqu'à S terminal
    
jusqu'à convergence

retourner V
```

### 2.2 Propriétés de convergence

```{important}
**Théorème de convergence TD(0)**

Sous les conditions :
1. $\sum_{t=1}^{\infty} \alpha_t = \infty$ (somme infinie)
2. $\sum_{t=1}^{\infty} \alpha_t^2 < \infty$ (somme des carrés finie)
3. Tous les états visités infiniment souvent

Alors TD(0) converge vers $V^{\pi}$ avec probabilité 1.

**Exemple de learning rate valide :** $\alpha_t = \frac{1}{t}$ ou $\alpha_t = \frac{1}{\sqrt{t}}$

**Pratique courante :** $\alpha$ constant (e.g., 0.1) pour adaptation continue
```

---

## 3. TD Control : SARSA (On-policy)

### 3.1 Principe

```{note}
**SARSA = State-Action-Reward-State-Action**

SARSA apprend $Q^{\pi}(s,a)$ en suivant **la même policy** $\pi$ pour :
- Choisir l'action $A_t$ en $S_t$
- Choisir l'action suivante $A_{t+1}$ en $S_{t+1}$

**Mise à jour :**

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$$

**On-policy :** La policy d'exploration et d'évaluation est la **même** (typiquement ε-greedy).
```

### 3.2 Algorithme

```
Algorithme : SARSA (on-policy TD Control)
────────────────────────────────────────────────
Entrée : learning rate α, exploration ε
Sortie : Q ≈ Q*, π ≈ π*

// Initialisation
pour chaque s ∈ S, a ∈ A faire :
    Q(s,a) ← 0  (ou arbitraire)
fin pour

// Apprentissage
répéter pour chaque épisode :
    Initialiser S
    Choisir A depuis S avec policy ε-greedy basée sur Q
    
    répéter pour chaque step :
        Exécuter A, observer R, S'
        
        Choisir A' depuis S' avec policy ε-greedy basée sur Q
        
        // Mise à jour SARSA
        Q(S,A) ← Q(S,A) + α[R + γ·Q(S',A') - Q(S,A)]
        
        S ← S'
        A ← A'
        
    jusqu'à S terminal
    
jusqu'à convergence

retourner Q
```

### 3.3 Caractéristiques

```{note}
**Propriétés de SARSA**

✅ **On-policy** : Apprend la valeur de la policy qu'il suit (ε-greedy)

✅ **Convergence garantie** : Vers $Q^{\pi_{\varepsilon\text{-greedy}}}$ sous conditions standards

✅ **Prudent** : Tient compte de l'exploration dans l'apprentissage

❌ **Pas optimal** : Converge vers la policy ε-greedy, pas la policy optimale pure

**Astuce pratique :** Réduire $\varepsilon$ progressivement pour s'approcher de l'optimal
```

---

## 4. Q-Learning (Off-policy)

### 4.1 Principe

```{important}
**Q-Learning : apprentissage off-policy**

Q-Learning apprend **directement** $Q^*$ (optimal) indépendamment de la policy suivie.

**Mise à jour :**

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t)\right]$$

**Différence clé avec SARSA :**
- **SARSA** : utilise $Q(S', A')$ (action **réellement choisie**)
- **Q-Learning** : utilise $\max_a Q(S', a)$ (action **optimale**)

**Off-policy :** La policy d'exploration (ε-greedy) et la policy cible (greedy) sont **différentes**.
```

### 4.2 Algorithme

```
Algorithme : Q-Learning (off-policy TD Control)
────────────────────────────────────────────────
Entrée : learning rate α, exploration ε
Sortie : Q ≈ Q*, π* greedy basée sur Q

// Initialisation
pour chaque s ∈ S, a ∈ A faire :
    Q(s,a) ← 0  (ou arbitraire)
fin pour

// Apprentissage
répéter pour chaque épisode :
    Initialiser S
    
    répéter pour chaque step :
        Choisir A depuis S avec policy ε-greedy basée sur Q
        Exécuter A, observer R, S'
        
        // Mise à jour Q-Learning (MAX bootstrap)
        Q(S,A) ← Q(S,A) + α[R + γ·max_a Q(S',a) - Q(S,A)]
        
        S ← S'
        
    jusqu'à S terminal
    
jusqu'à convergence

// Extraire policy optimale
pour chaque s ∈ S faire :
    π(s) ← argmax_a Q(s,a)
fin pour

retourner Q, π
```

### 4.3 Convergence

```{important}
**Théorème de convergence Q-Learning**

Sous les conditions :
1. Tous les états et actions visités infiniment souvent
2. Learning rate satisfait les conditions de Robbins-Monro
3. Q-values bornées

Alors Q-Learning converge vers $Q^*$ avec probabilité 1.

**Résultat remarquable :** Converge vers l'optimal même en explorant avec ε-greedy !
```

---

## 5. SARSA vs Q-Learning

### 5.1 Comparaison théorique

| Critère | SARSA | Q-Learning |
|---------|-------|------------|
| **Type** | On-policy | Off-policy |
| **Cible** | $R + \gamma Q(S', A')$ | $R + \gamma \max_a Q(S', a)$ |
| **Converge vers** | $Q^{\pi_{\varepsilon}}$ | $Q^*$ |
| **Comportement** | Prudent | Optimiste |
| **Variance** | Plus faible | Plus élevée |
| **Biais** | Léger | Plus important initialement |
| **Usage** | Environnements dangereux | Maximiser performance |

### 5.2 Exemple illustratif : Cliff Walking

```{admonition} Problème du Cliff Walking
:class: note

![Clif walking animation](https://imgs.search.brave.com/ZczVz-qSmdpDpsE3yOzBVw2PezrrltI21QefkGVoUKQ/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9neW1u/YXNpdW0uZmFyYW1h/Lm9yZy9faW1hZ2Vz/L2NsaWZmX3dhbGtp/bmcuZ2lm.gif)
Imaginons un gridworld avec :
- État départ : en bas à gauche
- État goal : en bas à droite
- Falaise (cliff) : ligne du bas entre départ et goal (reward = -100)
- Chemin sûr : contourner par le haut (plus long)

**Comportement observé :**
- **SARSA** : Apprend un chemin **sûr** (loin de la falaise) car il explore avec ε-greedy
- **Q-Learning** : Apprend le chemin **optimal** (proche de la falaise) mais risqué pendant l'apprentissage

**Raison :** SARSA prend en compte les erreurs d'exploration dans son apprentissage.
```

---

## 6. TD(λ) et Eligibility Traces

### 6.1 N-step TD

```{admonition} Généralisation : n-step returns
:class: note

Au lieu d'un seul step (TD(0)), on peut utiliser $n$ steps :

**1-step (TD(0))** : $G_t^{(1)} = R_{t+1} + \gamma V(S_{t+1})$

**2-step** : $G_t^{(2)} = R_{t+1} + \gamma R_{t+2} + \gamma^2 V(S_{t+2})$

**n-step** : $G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n V(S_{t+n})$

**MC (∞-step)** : $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$

**Trade-off :** $n$ petit → moins de variance mais plus de biais ; $n$ grand → inverse
```

### 6.2 TD(λ) avec eligibility traces

```{admonition} TD(λ) : combinaison de tous les n-step returns
:class: note

Le paramètre $\lambda \in [0,1]$ contrôle le mélange :

**λ-return :**

$$G_t^{\lambda} = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$

**Cas particuliers :**
- $\lambda = 0$ → TD(0) (bootstrap complet)
- $\lambda = 1$ → Monte Carlo (pas de bootstrap)

**Eligibility trace** $e_t(s)$ : mémoire de quels états doivent être crédités

$$e_t(s) = \begin{cases}
\gamma \lambda e_{t-1}(s) + 1 & \text{si } s = S_t \\
\gamma \lambda e_{t-1}(s) & \text{sinon}
\end{cases}$$

**Mise à jour TD(λ) :**

Pour tous les états $s$ à chaque step :

$$V(s) \leftarrow V(s) + \alpha \delta_t e_t(s)$$

où $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$
```

---

## 7. Exercices théoriques

### Exercice 1 — Calcul de TD error

```{admonition} Exercice
:class: tip

Un agent observe la transition suivante :

$$S_t = s_1 \xrightarrow{A_t = a_1, R_{t+1} = 3} S_{t+1} = s_2$$

**Value functions actuelles :**
- $V(s_1) = 10.0$
- $V(s_2) = 15.0$

**Paramètres :** $\alpha = 0.1$, $\gamma = 0.9$

**Questions :**

a) Calculez le TD error $\delta_t$

b) Quelle est la nouvelle valeur $V(s_1)$ après mise à jour TD(0) ?

c) Si $V(s_2)$ était en réalité 5.0 (au lieu de 15.0), quel serait $\delta_t$ ?

d) Interprétez le signe des TD errors en (a) et (c)
```

````{dropdown} Solution — Exercice 1

**a) TD error :**

$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

$$\delta_t = 3 + 0.9 \times 15.0 - 10.0$$

$$\delta_t = 3 + 13.5 - 10.0 = 6.5$$

**b) Mise à jour TD(0) :**

$$V(S_t) \leftarrow V(S_t) + \alpha \delta_t$$

$$V(s_1) \leftarrow 10.0 + 0.1 \times 6.5$$

$$V(s_1) \leftarrow 10.0 + 0.65 = 10.65$$

**c) TD error avec $V(s_2) = 5.0$ :**

$$\delta_t = 3 + 0.9 \times 5.0 - 10.0$$

$$\delta_t = 3 + 4.5 - 10.0 = -2.5$$

**d) Interprétation :**

**Cas (a)** : $\delta_t = 6.5 > 0$
- La reward + valeur future (16.5) > estimation actuelle (10.0)
- L'état $s_1$ était **sous-estimé**
- La mise à jour **augmente** $V(s_1)$

**Cas (c)** : $\delta_t = -2.5 < 0$
- La reward + valeur future (7.5) < estimation actuelle (10.0)
- L'état $s_1$ était **surestimé**
- La mise à jour **diminuerait** $V(s_1)$
````

---

### Exercice 2 — SARSA step by step

```{admonition} Exercice
:class: tip

Un agent utilise SARSA avec ε-greedy ($\varepsilon = 0.2$).

**Transition observée :**

$$S_t = s_A, A_t = a_1 \xrightarrow{R_{t+1} = 5} S_{t+1} = s_B, A_{t+1} = a_2$$

**Q-values actuelles :**
- $Q(s_A, a_1) = 8.0$
- $Q(s_A, a_2) = 6.0$
- $Q(s_B, a_1) = 12.0$
- $Q(s_B, a_2) = 10.0$

**Paramètres :** $\alpha = 0.2$, $\gamma = 0.9$

**Questions :**

a) Calculez le TD target pour SARSA

b) Calculez le TD error

c) Quelle est la nouvelle valeur $Q(s_A, a_1)$ ?

d) L'agent a-t-il choisi l'action greedy en $s_B$ ? Est-ce important pour SARSA ?
```

````{dropdown} Solution — Exercice 2

**a) TD target pour SARSA :**

$$\text{Target} = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})$$

$$= 5 + 0.9 \times Q(s_B, a_2)$$

$$= 5 + 0.9 \times 10.0 = 5 + 9.0 = 14.0$$

**b) TD error :**

$$\delta_t = \text{Target} - Q(S_t, A_t)$$

$$= 14.0 - 8.0 = 6.0$$

**c) Mise à jour Q-value :**

$$Q(s_A, a_1) \leftarrow Q(s_A, a_1) + \alpha \delta_t$$

$$Q(s_A, a_1) \leftarrow 8.0 + 0.2 \times 6.0$$

$$Q(s_A, a_1) \leftarrow 8.0 + 1.2 = 9.2$$

**d) Action greedy en $s_B$ ?**

L'action greedy serait $\arg\max_a Q(s_B, a) = a_1$ (car 12.0 > 10.0).

L'agent a choisi $a_2$, donc c'était une **action d'exploration** (probabilité $\varepsilon$).

**Pour SARSA, c'est crucial !** SARSA utilise $Q(S', A')$ où $A'$ est l'action **réellement choisie**, donc il apprend la valeur de la policy ε-greedy (qui inclut l'exploration).
````

---

### Exercice 3 — Q-Learning vs SARSA

```{admonition} Exercice
:class: tip

Même situation que l'exercice 2, mais maintenant comparons Q-Learning.

**Rappel de la transition :**

$$S_t = s_A, A_t = a_1 \xrightarrow{R_{t+1} = 5} S_{t+1} = s_B, A_{t+1} = a_2$$

**Q-values :** (identiques)
- $Q(s_A, a_1) = 8.0$
- $Q(s_B, a_1) = 12.0$
- $Q(s_B, a_2) = 10.0$

**Paramètres :** $\alpha = 0.2$, $\gamma = 0.9$

**Questions :**

a) Calculez le TD target pour Q-Learning

b) Calculez le TD error

c) Quelle est la nouvelle valeur $Q(s_A, a_1)$ avec Q-Learning ?

d) Comparez avec le résultat SARSA (exercice 2). Pourquoi la différence ?
```

````{dropdown} Solution — Exercice 3

**a) TD target pour Q-Learning :**

$$\text{Target} = R_{t+1} + \gamma \max_a Q(S_{t+1}, a)$$

$$= 5 + 0.9 \times \max\{Q(s_B, a_1), Q(s_B, a_2)\}$$

$$= 5 + 0.9 \times \max\{12.0, 10.0\}$$

$$= 5 + 0.9 \times 12.0 = 5 + 10.8 = 15.8$$

**b) TD error :**

$$\delta_t = 15.8 - 8.0 = 7.8$$

**c) Mise à jour Q-value :**

$$Q(s_A, a_1) \leftarrow 8.0 + 0.2 \times 7.8$$

$$Q(s_A, a_1) \leftarrow 8.0 + 1.56 = 9.56$$

**d) Comparaison SARSA vs Q-Learning :**

| Méthode | Target | Nouvelle $Q(s_A, a_1)$ |
|---------|--------|------------------------|
| SARSA | 14.0 | 9.2 |
| Q-Learning | 15.8 | 9.56 |

**Différence :**
- **SARSA** utilise $Q(s_B, a_2) = 10.0$ (action explorée)
- **Q-Learning** utilise $\max_a Q(s_B, a) = 12.0$ (action optimale)

Q-Learning est plus **optimiste** car il suppose toujours l'action optimale au prochain step, alors que SARSA prend en compte l'exploration réelle.
````

---

### Exercice 4 — Séquence de mises à jour TD

```{admonition} Exercice
:class: tip

Un agent apprend avec TD(0) sur la séquence suivante :

**Épisode :**

$$s_1 \xrightarrow{r=2} s_2 \xrightarrow{r=3} s_3 \xrightarrow{r=5} \text{terminal}$$

**Valeurs initiales :** $V(s_1) = 0$, $V(s_2) = 0$, $V(s_3) = 0$

**Paramètres :** $\alpha = 0.5$, $\gamma = 1.0$

**Calculez** les valeurs après une passe **backward** (de la fin vers le début) à travers l'épisode.

**Note :** En TD, on met à jour pendant l'épisode, mais pour cet exercice, on simule les mises à jour dans l'ordre inverse pour voir la propagation.
```

````{dropdown} Solution — Exercice 4

**Approche :** On traite les transitions de la fin vers le début.

**Transition 3 :** $s_3 \xrightarrow{r=5} \text{terminal}$

$$\delta = 5 + 1.0 \times 0 - V(s_3) = 5 - 0 = 5$$

$$V(s_3) \leftarrow 0 + 0.5 \times 5 = 2.5$$

**Transition 2 :** $s_2 \xrightarrow{r=3} s_3$

Utiliser $V(s_3) = 2.5$ (valeur mise à jour)

$$\delta = 3 + 1.0 \times 2.5 - V(s_2) = 5.5 - 0 = 5.5$$

$$V(s_2) \leftarrow 0 + 0.5 \times 5.5 = 2.75$$

**Transition 1 :** $s_1 \xrightarrow{r=2} s_2$

Utiliser $V(s_2) = 2.75$

$$\delta = 2 + 1.0 \times 2.75 - V(s_1) = 4.75 - 0 = 4.75$$

$$V(s_1) \leftarrow 0 + 0.5 \times 4.75 = 2.375$$

**Valeurs finales après une passe :**
- $V(s_1) = 2.375$
- $V(s_2) = 2.75$
- $V(s_3) = 2.5$

**Observation :** Les valeurs se propagent depuis l'état terminal. $V(s_1)$ devrait converger vers $2 + 3 + 5 = 10$ avec plus d'épisodes.
````

---

### Exercice 5 — Convergence avec learning rate

```{admonition} Exercice
:class: tip

On applique TD(0) sur un état $s$ avec observations successives :

**Séquence de TD targets observés :**
- Step 1 : Target = 12.0
- Step 2 : Target = 8.0
- Step 3 : Target = 10.0
- Step 4 : Target = 11.0

**Valeur initiale :** $V(s) = 5.0$

**Comparez** l'évolution de $V(s)$ pour deux learning rates :

a) $\alpha = 0.1$ (petit)

b) $\alpha = 0.5$ (grand)

**Question :** Quel learning rate converge plus vite ? Lequel est plus stable ?
```

````{dropdown} Solution — Exercice 5

**a) Avec $\alpha = 0.1$ :**

**Step 1 :** Target = 12.0
$$V(s) \leftarrow 5.0 + 0.1 \times (12.0 - 5.0) = 5.0 + 0.7 = 5.7$$

**Step 2 :** Target = 8.0
$$V(s) \leftarrow 5.7 + 0.1 \times (8.0 - 5.7) = 5.7 + 0.23 = 5.93$$

**Step 3 :** Target = 10.0
$$V(s) \leftarrow 5.93 + 0.1 \times (10.0 - 5.93) = 5.93 + 0.407 = 6.337$$

**Step 4 :** Target = 11.0
$$V(s) \leftarrow 6.337 + 0.1 \times (11.0 - 6.337) = 6.337 + 0.466 = 6.803$$

**Séquence :** 5.0 → 5.7 → 5.93 → 6.337 → 6.803

**b) Avec $\alpha = 0.5$ :**

**Step 1 :** Target = 12.0
$$V(s) \leftarrow 5.0 + 0.5 \times (12.0 - 5.0) = 5.0 + 3.5 = 8.5$$

**Step 2 :** Target = 8.0
$$V(s) \leftarrow 8.5 + 0.5 \times (8.0 - 8.5) = 8.5 - 0.25 = 8.25$$

**Step 3 :** Target = 10.0
$$V(s) \leftarrow 8.25 + 0.5 \times (10.0 - 8.25) = 8.25 + 0.875 = 9.125$$

**Step 4 :** Target = 11.0
$$V(s) \leftarrow 9.125 + 0.5 \times (11.0 - 9.125) = 9.125 + 0.9375 = 10.0625$$

**Séquence :** 5.0 → 8.5 → 8.25 → 9.125 → 10.0625

**Comparaison :**

| Critère | $\alpha = 0.1$ | $\alpha = 0.5$ |
|---------|----------------|----------------|
| **Valeur finale** | 6.803 | 10.0625 |
| **Vitesse** | Lente | Rapide |
| **Oscillations** | Faibles | Plus importantes |
| **Stabilité** | Élevée | Moyenne |

**Conclusion :**
- **$\alpha$ grand** (0.5) : converge plus vite vers la moyenne des targets (~10.25), mais plus sensible au bruit
- **$\alpha$ petit** (0.1) : plus stable, moins d'oscillations, mais convergence plus lente

**Moyenne des targets :** $(12 + 8 + 10 + 11)/4 = 10.25$ — $\alpha = 0.5$ est plus proche après 4 steps.
````

---

## 8. Avantages et limitations

### 8.1 Avantages des méthodes TD

```{admonition} Points forts
:class: note

✅ **Model-free** : Pas besoin de $P$ et $R$

✅ **Apprentissage online** : Mise à jour à chaque step (pas d'attendre la fin de l'épisode)

✅ **Fonctionne avec tâches continues** : Pas besoin d'épisodes terminaux

✅ **Faible variance** : Moins que MC (utilise bootstrap)

✅ **Convergence rapide** : Généralement plus rapide que MC

✅ **Flexible** : Peut être on-policy (SARSA) ou off-policy (Q-Learning)
```

### 8.2 Limitations

```{admonition} Contraintes
:class: note

❌ **Biais d'estimation** : Bootstrap introduit du biais (contrairement à MC)

❌ **Sensible aux hyperparamètres** : Learning rate $\alpha$ crucial

❌ **Peut diverger** : Avec function approximation (voir deadly triad)

❌ **Exploration nécessaire** : Tous états doivent être visités

❌ **Q-Learning peut surestimer** : Maximization bias (résolu par Double Q-Learning)
```