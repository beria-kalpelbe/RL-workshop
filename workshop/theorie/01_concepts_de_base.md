# Concepts fondamentaux du RL

Ce chapitre introduit les bases conceptuelles du Reinforcement Learning (RL) en combinant **intuition**, **formalisme mathématique**, et **pseudocode** pour une compréhension à la fois théorique et pratique.

---

## 1. Vue d'ensemble

```{admonition} Intuition
:class: tip

Un **agent** interagit avec un **environment** en observant un **state** $s$, en choisissant une **action** $a$, et en recevant une **reward** $r$ ainsi qu'un nouvel état $s'$. 

L'objectif est d'apprendre une **policy** $\pi(a|s)$ maximisant la somme des rewards futures.
```
![MDP figure](../_static/mdp.png)

```{admonition} Formulation mathématique
:class: important

Un **Processus de Décision de Markov (MDP)** est défini par le quintuplet :

$$(S, A, P, R, \gamma)$$


- $S$ : ensemble (fini ou infini) des **states**
- $A$ : ensemble (fini ou infini) des **actions**  
- $P : S \times A \times S \to [0,1]$ : **fonction de transition**
  - $P(s'|s,a)$ = probabilité d'atteindre $s'$ depuis $s$ en exécutant $a$
  - Propriété : $\sum_{s' \in S} P(s'|s,a) = 1$ pour tout $(s,a)$
- $R : S \times A \to \mathbb{R}$ : **fonction de reward**
  - $R(s,a)$ = reward espérée en exécutant $a$ dans $s$
  - Variante : $R(s,a,s')$ pour des rewards dépendant de la transition
- $\gamma \in [0,1]$ : **discount factor**

On peut aussi définir $P(s',r|s,a)$ comme distribution conjointe sur les transitions et les rewards.
```

```{admonition} Hypothèse de Markov
:class: note

La probabilité de transition ne dépend que du **state actuel** et de l'**action choisie**, pas de l'historique :

$$\mathbb{P}(S_{t+1}=s', R_{t+1}=r | S_t=s, A_t=a, S_{t-1}, A_{t-1}, \ldots, S_0, A_0)$$
$$= \mathbb{P}(S_{t+1}=s', R_{t+1}=r | S_t=s, A_t=a)$$

Cette propriété simplifie énormément l'analyse mathématique et permet l'existence de solutions optimales stationnaires.
```
<img src= "https://imgs.search.brave.com/M7dyYhd84YRqccKC6mEh7V9ShR0R-6QvNrpKHVBBX0A/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9kczA1/NXV6ZXRhb2JiLmNs/b3VkZnJvbnQubmV0/L2JyaW9jaGUvdXBs/b2Fkcy9ZNDFsTDkx/emJnLWJpZ2dlci1t/YXJrb3YtY2hhaW4u/cG5nP3dpZHRoPTEy/MDA" width="100%">


Le **return** à l'instant $t$ est :

$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

L'objectif de l'agent est de maximiser $\mathbb{E}[G_t]$.

Les **value functions** sont :

$$V^{\pi}(s) = \mathbb{E}_{\pi}[G_t | S_t=s]$$

$$Q^{\pi}(s,a) = \mathbb{E}_{\pi}[G_t | S_t=s, A_t=a]$$

Elles satisfont les **équations de Bellman** :

$$V^{\pi}(s) = \sum_a \pi(a|s) \sum_{s',r} P(s',r|s,a) [r + \gamma V^{\pi}(s')]$$

$$Q^{\pi}(s,a) = \sum_{s',r} P(s',r|s,a) [r + \gamma \sum_{a'} \pi(a'|s') Q^{\pi}(s',a')]$$

Une **optimal policy** $\pi^*$ maximise $V^{\pi}(s)$ pour tout $s$. On note $V^*$ et $Q^*$ les fonctions optimales correspondantes.

---

## 2. Boucle d'interaction agent–environment
```
Algorithme : Boucle d'apprentissage RL
──────────────────────────────────────────────
Entrée : environment E, policy initiale π
Sortie : policy améliorée π*

pour chaque episode faire :
    s ← initialiser_état(E)
    terminé ← FAUX
    
    tant que non terminé faire :
        a ← sélectionner_action(s, π)
        (s', r, terminé) ← E.step(a)
        
        mettre_à_jour(π, s, a, r, s')
        
        s ← s'
    fin tant que
fin pour

retourner π
```

Cette boucle illustre le cycle d'apprentissage fondamental : **observation → décision → mise à jour**. Tous les algorithmes de RL se basent sur cette structure, avec des variations dans la fonction **mettre_à_jour** (Q-learning, SARSA, Policy Gradient, etc.).

---

## 3. Exploration vs Exploitation

**Le dilemme fondamental du RL :** faut-il explorer de nouvelles actions pour découvrir de meilleures stratégies, ou exploiter les connaissances actuelles pour maximiser les rewards ?

### Stratégies courantes

**ε-greedy**

```
avec probabilité ε :
    a ← action_aléatoire()
sinon :
    a ← argmax_a' Q(s, a')
```

**Softmax / Boltzmann**

La probabilité de choisir l'action $a$ est :

$$\pi(a|s) = \frac{\exp(Q(s,a)/\tau)}{\sum_{a'} \exp(Q(s,a')/\tau)}$$

où $\tau > 0$ contrôle la **temperature** (niveau d'exploration).

### Décroissance de l'exploration

Pour réduire progressivement l'exploration :

**Décroissance exponentielle :**
$\varepsilon_t = \varepsilon_0 \cdot e^{-kt}$

**Décroissance linéaire :**
$\varepsilon_t = \max(\varepsilon_{min}, \varepsilon_0 - kt)$

---

## 4. Value Iteration

L'algorithme **Value Iteration** calcule la value function optimale $V^*$ :

```
Algorithme : Value Iteration
──────────────────────────────────────────────
Entrée : MDP (S, A, P, R, γ), seuil θ
Sortie : V* ≈ value function optimale

initialiser V(s) ← 0 pour tout s ∈ S

répéter :
    Δ ← 0
    pour chaque état s ∈ S faire :
        v ← V(s)
        V(s) ← max_a Σ_{s',r} P(s',r|s,a)[r + γV(s')]
        Δ ← max(Δ, |v - V(s)|)
    fin pour
jusqu'à Δ < θ

retourner V
```

**Théorème :** Value Iteration converge vers $V^*$ en temps polynomial.

---

## 5. Policy Evaluation et Policy Improvement

### Policy Evaluation

Évaluer une policy fixée $\pi$ :

```
Algorithme : Policy Evaluation (itératif)
──────────────────────────────────────────────
Entrée : policy π, seuil θ
Sortie : V^π

initialiser V(s) ← 0 pour tout s ∈ S

répéter :
    Δ ← 0
    pour chaque état s ∈ S faire :
        v ← V(s)
        V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a)[r + γV(s')]
        Δ ← max(Δ, |v - V(s)|)
    fin pour
jusqu'à Δ < θ

retourner V
```

### Policy Improvement

Améliorer la policy de manière gloutonne (greedy) :

```
Algorithme : Policy Improvement
──────────────────────────────────────────────
Entrée : V^π (value function de π)
Sortie : π' (policy améliorée)

pour chaque état s ∈ S faire :
    π'(s) ← argmax_a Σ_{s',r} P(s',r|s,a)[r + γV^π(s')]
fin pour

retourner π'
```

**Théorème de Policy Improvement :** $V^{\pi'}(s) \geq V^{\pi}(s)$ pour tout $s$.

---

## 6. Exercices théoriques

### Exercice 1 — Calcul de return

```{admonition} Exercice
:class: tip

Considérons une **trajectory** : $(s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2, r_3, s_3)$ avec :
- $r_1 = 5$, $r_2 = -2$, $r_3 = 10$
- Discount factor $\gamma = 0.9$

**Calculez à la main :**

a) Le return $G_0$ depuis $t=0$

b) Le return $G_1$ depuis $t=1$

c) Le return $G_2$ depuis $t=2$
```

````{dropdown} Solution — Exercice 1

**a) Return depuis $t=0$ :**

$$G_0 = r_1 + \gamma r_2 + \gamma^2 r_3$$

$$G_0 = 5 + (0.9)(-2) + (0.9)^2(10)$$

$$G_0 = 5 - 1.8 + 8.1 = 11.3$$

**b) Return depuis $t=1$ :**

$$G_1 = r_2 + \gamma r_3$$

$$G_1 = -2 + (0.9)(10) = -2 + 9 = 7$$

**c) Return depuis $t=2$ :**

$$G_2 = r_3 = 10$$

**Vérification :** On doit avoir $G_0 = r_1 + \gamma G_1$

$ 5 + 0.9 \times 7 = 5 + 6.3 = 11.3 $ ✓
````

---

### Exercice 2 — Policy Evaluation (calcul matriciel)

```{admonition} Exercice
:class: tip

Soit un MDP à 3 états avec $\gamma = 0.8$ et une policy déterministe $\pi$.

**Matrice de transition** $P^\pi$ (probabilité d'aller de $s_i$ à $s_j$ sous $\pi$) :

$$P^\pi = \begin{pmatrix}
0.5 & 0.3 & 0.2 \\
0.1 & 0.7 & 0.2 \\
0.2 & 0.2 & 0.6
\end{pmatrix}$$

**Vecteur de rewards** sous $\pi$ : $R^\pi = \begin{pmatrix} 10 \\ 0 \\ -5 \end{pmatrix}$

**Calculez** la value function $V^\pi$ en résolvant :

$$V^\pi = (I - \gamma P^\pi)^{-1} R^\pi$$

où $I$ est la matrice identité $3 \times 3$.
```

````{dropdown} Solution — Exercice 2

**Étape 1 :** Calculer $I - \gamma P^\pi$

$$I - 0.8 P^\pi = \begin{pmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{pmatrix} - 0.8 \begin{pmatrix}
0.5 & 0.3 & 0.2 \\
0.1 & 0.7 & 0.2 \\
0.2 & 0.2 & 0.6
\end{pmatrix}$$

$$= \begin{pmatrix}
0.6 & -0.24 & -0.16 \\
-0.08 & 0.44 & -0.16 \\
-0.16 & -0.16 & 0.52
\end{pmatrix}$$

**Étape 2 :** Inverser la matrice (calcul à faire numériquement ou avec Gauss-Jordan)

$$(I - 0.8 P^\pi)^{-1} \approx \begin{pmatrix}
2.18 & 1.45 & 1.09 \\
0.73 & 2.91 & 1.09 \\
1.09 & 1.09 & 2.55
\end{pmatrix}$$

**Étape 3 :** Multiplier par $R^\pi$

$$V^\pi = \begin{pmatrix}
2.18 & 1.45 & 1.09 \\
0.73 & 2.91 & 1.09 \\
1.09 & 1.09 & 2.55
\end{pmatrix} \begin{pmatrix} 10 \\ 0 \\ -5 \end{pmatrix}$$

$$V^\pi = \begin{pmatrix}
2.18 \times 10 + 1.45 \times 0 + 1.09 \times (-5) \\
0.73 \times 10 + 2.91 \times 0 + 1.09 \times (-5) \\
1.09 \times 10 + 1.09 \times 0 + 2.55 \times (-5)
\end{pmatrix}$$

$$V^\pi = \begin{pmatrix}
21.8 - 5.45 \\
7.3 - 5.45 \\
10.9 - 12.75
\end{pmatrix} = \begin{pmatrix}
16.35 \\
1.85 \\
-1.85
\end{pmatrix}$$

**Interprétation :** L'état $s_1$ a la valeur la plus élevée, tandis que $s_3$ a une valeur négative.
````

---

### Exercice 3 — Bellman Equation (vérification)

```{admonition} Exercice
:class: tip

Considérons un gridworld 2×2 avec 4 états. Un agent reçoit une reward de $-1$ à chaque step (sauf dans l'état terminal). Le discount factor est $\gamma = 1.0$.

**États :** 
- État 1 (terminal) : reward = 0
- États 2, 3, 4 (non-terminaux) : reward = -1 par transition

**Policy uniforme** : l'agent choisit chaque action disponible avec probabilité égale.

Supposons que vous avez calculé : $V^\pi(1) = 0$, $V^\pi(2) = -3$, $V^\pi(3) = -2$, $V^\pi(4) = -2$

**Vérifiez** que ces valeurs satisfont l'équation de Bellman pour l'état 2, sachant que depuis l'état 2 :
- Action "haut" → état 1 (prob. 0.5)
- Action "gauche" → reste en 2 (prob. 0.5)
```

````{dropdown} Solution — Exercice 3

L'équation de Bellman pour l'état 2 est :

$V^\pi(2) = \sum_a \pi(a|2) \sum_{s'} P(s'|2,a)[r + \gamma V^\pi(s')]$

Avec une policy uniforme sur 2 actions : $\pi(a|2) = 0.5$ pour chaque action.

**Action "haut"** (mène à l'état 1, terminal) :
$Q(2, \text{haut}) = -1 + 1.0 \times V^\pi(1) = -1 + 0 = -1$

**Action "gauche"** (reste en état 2) :
$Q(2, \text{gauche}) = -1 + 1.0 \times V^\pi(2) = -1 + V^\pi(2)$

Donc :

$$V^\pi(2) = 0.5 \times (-1) + 0.5 \times (-1 + V^\pi(2))$$

$$V^\pi(2) = -0.5 - 0.5 + 0.5 \times V^\pi(2)$$

$$V^\pi(2) = -1 + 0.5 \times V^\pi(2)$$

$$0.5 \times V^\pi(2) = -1$$

$$V^\pi(2) = -2$$

**Problème !** La valeur donnée était $V^\pi(2) = -3$, mais le calcul montre qu'elle devrait être $-2$.

**Conclusion :** Les valeurs fournies ne satisfont **pas** l'équation de Bellman pour l'état 2. Il faut recalculer.
````

---

### Exercice 4 — Optimal Policy

```{admonition} Exercice
:class: tip

Soit un MDP avec 2 états $\{s_1, s_2\}$ et 2 actions $\{a_1, a_2\}$.

**Q-values optimales connues :**

| État | $Q^*(s, a_1)$ | $Q^*(s, a_2)$ |
|------|---------------|---------------|
| $s_1$ | 5.2 | 3.8 |
| $s_2$ | 2.1 | 4.5 |

**Questions :**

a) Quelle est l'optimal policy déterministe $\pi^*(s)$ ?

b) Calculez $V^*(s_1)$ et $V^*(s_2)$.

c) Si on utilise une policy softmax avec $\tau = 0.5$, quelle est $\pi(a_1|s_1)$ ?
```

````{dropdown} Solution — Exercice 4

**a) Optimal policy déterministe :**

L'optimal policy choisit l'action qui maximise la Q-value :

$$\pi^*(s_1) = \arg\max_a Q^*(s_1, a) = a_1 \quad \text{(car } 5.2 > 3.8\text{)}$$

$$\pi^*(s_2) = \arg\max_a Q^*(s_2, a) = a_2 \quad \text{(car } 4.5 > 2.1\text{)}$$

**b) Optimal value function :**

$$V^*(s) = \max_a Q^*(s, a)$$

$$V^*(s_1) = \max(5.2, 3.8) = 5.2$$

$$V^*(s_2) = \max(2.1, 4.5) = 4.5$$

**c) Policy softmax pour $s_1$ avec $\tau = 0.5$ :**

$$\pi(a_1|s_1) = \frac{\exp(Q^*(s_1, a_1)/\tau)}{\exp(Q^*(s_1, a_1)/\tau) + \exp(Q^*(s_1, a_2)/\tau)}$$

$$= \frac{\exp(5.2/0.5)}{\exp(5.2/0.5) + \exp(3.8/0.5)}$$

$$= \frac{\exp(10.4)}{\exp(10.4) + \exp(7.6)}$$

$$= \frac{32844.7}{32844.7 + 2000.3} = \frac{32844.7}{34845} \approx 0.943$$

L'agent choisira $a_1$ avec ~94.3% de probabilité et $a_2$ avec ~5.7%.
````

---

### Exercice 5 — TD Error et Bellman Residual

```{admonition} Exercice
:class: tip

Dans l'algorithme **TD(0)**, l'agent met à jour ses estimations avec le **TD error** :

$$\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$$

**Scénario :** Un agent observe la transition suivante :
- État actuel : $s_t = s_3$, avec $V(s_3) = 8.0$
- Action effectuée : $a_t$
- Reward obtenue : $r_{t+1} = 2.5$
- Nouvel état : $s_{t+1} = s_5$, avec $V(s_5) = 10.0$
- Discount factor : $\gamma = 0.95$

**Calculez :**

a) Le TD error $\delta_t$

b) La nouvelle valeur $V(s_3)$ après mise à jour avec learning rate $\alpha = 0.1$

c) Interprétez le signe du TD error
```

````{dropdown} Solution — Exercice 5

**a) Calcul du TD error :**

$$\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$$

$$\delta_t = 2.5 + 0.95 \times 10.0 - 8.0$$

$$\delta_t = 2.5 + 9.5 - 8.0 = 4.0$$

**b) Mise à jour de la value function :**

La règle de mise à jour TD(0) est :

$$V(s_t) \leftarrow V(s_t) + \alpha \delta_t$$

$$V(s_3) \leftarrow 8.0 + 0.1 \times 4.0$$

$$V(s_3) \leftarrow 8.0 + 0.4 = 8.4$$

**c) Interprétation :**

Le TD error est **positif** ($\delta_t = 4.0 > 0$), ce qui signifie que :
- La reward immédiate + valeur future estimée ($2.5 + 9.5 = 12.0$) est **supérieure** à l'estimation actuelle ($8.0$)
- L'agent a sous-estimé la valeur de l'état $s_3$
- La mise à jour **augmente** $V(s_3)$ pour corriger cette sous-estimation

Si le TD error était négatif, cela indiquerait une surestimation de la valeur.
````

---

## 7. Résumé

```{important}
**Concepts essentiels**

1. **MDP** : Formalisation mathématique du problème RL $(S, A, P, R, \gamma)$
2. **Value functions** : $V^\pi(s)$ mesure la qualité d'un état, $Q^\pi(s,a)$ d'une paire état-action
3. **Équations de Bellman** : Relations récursives qui permettent de calculer les value functions
4. **Return** : $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$ — somme pondérée des rewards futures
5. **Exploration/Exploitation** : Dilemme central résolu par des stratégies comme ε-greedy
6. **Policy Evaluation** : Calcul de $V^\pi$ pour une policy fixée
7. **Policy Improvement** : Amélioration greedy basée sur $V^\pi$
8. **TD Error** : $\delta_t = r + \gamma V(s') - V(s)$ — signal d'apprentissage fondamental
```