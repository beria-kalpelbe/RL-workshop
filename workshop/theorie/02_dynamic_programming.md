# Dynamic Programming (DP)

Le **Dynamic Programming** regroupe des méthodes qui exploitent les équations de Bellman et la **connaissance complète du modèle** (fonctions $P$ et $R$) pour calculer des policies optimales de manière exacte et efficace.

---

## 1. Principes fondamentaux

```{admonition} Conditions d'application du DP
:class: important

Le Dynamic Programming s'applique aux problèmes qui satisfont deux propriétés :

1. **Structure optimale** : Une solution optimale du problème global contient des solutions optimales aux sous-problèmes
2. **Sous-problèmes chevauchants** : Les mêmes sous-problèmes sont résolus plusieurs fois

Les MDPs satisfont ces deux propriétés via les équations de Bellman.

**Hypothèses du DP en RL :**
- Modèle **complet** : $P(s'|s,a)$ et $R(s,a)$ connus
- Espace d'états **fini** : $|S| < \infty$
- Espace d'actions **fini** : $|A| < \infty$
```

Le terme **backup** désigne la mise à jour de la valeur d'un état basée sur les valeurs de ses successeurs.

**Full backup :** mise à jour utilisant **toutes** les actions et **tous** les successeurs possibles.

**Expected update :** moyenner sur les transitions stochastiques.

**Max backup :** prendre le maximum sur les actions (Value Iteration).

---

## 2. Value Iteration

### 2.1 Principe

Value Iteration applique directement l'**équation de Bellman optimale** de manière itérative :

$$V_{k+1}(s) = \max_{a \in A} \sum_{s' \in S} P(s'|s,a) \left[ R(s,a) + \gamma V_k(s') \right]$$

```{admonition} Intuition
:class: note

On démarre avec $V_0(s) = 0$ (ou valeurs arbitraires). À chaque itération :
1. Pour chaque état, on calcule la valeur en supposant qu'on joue optimalement à partir du prochain step
2. On utilise les valeurs actuelles $V_k$ pour estimer les valeurs futures
3. On prend le **maximum** sur toutes les actions possibles

Progressivement, les valeurs **se propagent** depuis les états terminaux/récompensants vers tous les états.
```

### 2.2 Algorithme

```
Algorithme : Value Iteration
────────────────────────────────────────────────
Entrée : MDP (S, A, P, R, γ), seuil θ > 0
Sortie : V* (approximé), π* (optimal policy)

// Initialisation
pour chaque s ∈ S faire :
    V(s) ← 0
fin pour

// Itération jusqu'à convergence
répéter :
    Δ ← 0
    
    pour chaque s ∈ S faire :
        v ← V(s)
        
        // Backup de Bellman optimale
        V(s) ← max_a Σ_{s'∈S} P(s'|s,a)[R(s,a) + γ·V(s')]
        
        // Mesurer le changement
        Δ ← max(Δ, |v - V(s)|)
    fin pour
    
jusqu'à Δ < θ

// Extraction de la policy optimale
pour chaque s ∈ S faire :
    π(s) ← argmax_a Σ_{s'∈S} P(s'|s,a)[R(s,a) + γ·V(s')]
fin pour

retourner V, π
```

### 2.3 Convergence

```{admonition} Théorème de convergence
:class: important

Si $\gamma < 1$ et $|S| < \infty$, alors Value Iteration converge vers $V^*$ en un nombre fini d'itérations.

**Taux de convergence :** La contraction garantit :

$$\|V_{k+1} - V^*\|_\infty \leq \gamma \|V_k - V^*\|_\infty$$

Donc la convergence est **géométrique** avec taux $\gamma$.

**Critère d'arrêt :** Quand $\|V_{k+1} - V_k\|_\infty < \theta$, on a :

$$\|V_k - V^*\|_\infty \leq \frac{\theta}{1-\gamma}$$
```

---

## 3. Policy Iteration

### 3.1 Principe

Policy Iteration alterne entre deux phases :

1. **Policy Evaluation** : Calculer $V^{\pi}$ pour la policy courante $\pi$
2. **Policy Improvement** : Améliorer $\pi$ de manière gloutonne basée sur $V^{\pi}$

```{admonition} Différence avec Value Iteration
:class: note

- **Value Iteration** : combine évaluation et amélioration en une seule étape (max)
- **Policy Iteration** : sépare explicitement les deux phases
- Policy Iteration converge souvent en **moins d'itérations** mais chaque itération coûte plus cher
```

### 3.2 Algorithme

```
Algorithme : Policy Iteration
────────────────────────────────────────────────
Entrée : MDP (S, A, P, R, γ), seuil θ > 0
Sortie : π* (optimal policy)

// Initialisation
pour chaque s ∈ S faire :
    π(s) ← action aléatoire de A
    V(s) ← 0
fin pour

répéter :
    
    // ═══════════════════════════════════════
    // PHASE 1 : Policy Evaluation
    // ═══════════════════════════════════════
    répéter :
        Δ ← 0
        pour chaque s ∈ S faire :
            v ← V(s)
            V(s) ← Σ_{s'∈S} P(s'|s,π(s))[R(s,π(s)) + γ·V(s')]
            Δ ← max(Δ, |v - V(s)|)
        fin pour
    jusqu'à Δ < θ
    
    // ═══════════════════════════════════════
    // PHASE 2 : Policy Improvement
    // ═══════════════════════════════════════
    policy_stable ← VRAI
    
    pour chaque s ∈ S faire :
        old_action ← π(s)
        
        // Amélioration gloutonne
        π(s) ← argmax_a Σ_{s'∈S} P(s'|s,a)[R(s,a) + γ·V(s')]
        
        si old_action ≠ π(s) alors :
            policy_stable ← FAUX
        fin si
    fin pour
    
jusqu'à policy_stable = VRAI

retourner π
```

### 3.3 Convergence

```{admonition} Théorème de convergence
:class: important

Policy Iteration converge vers la policy optimale $\pi^*$ en un **nombre fini** d'itérations.

**Borne supérieure :** Au plus $|A|^{|S|}$ itérations (en pratique, beaucoup moins).

**Monotonie :** Chaque nouvelle policy est au moins aussi bonne : $V^{\pi_{k+1}}(s) \geq V^{\pi_k}(s)$ pour tout $s$.
```

---

## 4. Comparaison Value Iteration vs Policy Iteration

| Critère | Value Iteration | Policy Iteration |
|---------|-----------------|------------------|
| **Mise à jour** | $\max_a$ à chaque step | Évaluation complète puis amélioration |
| **Convergence** | Géométrique vers $V^*$ | Finie vers $\pi^*$ |
| **Nombre d'itérations** | Plus d'itérations | Moins d'itérations |
| **Coût par itération** | Léger (un sweep) | Élevé (évaluation complète) |
| **Quand l'utiliser** | Espace d'états modéré | Convergence rapide souhaitée |

```{tip}
**Modified Policy Iteration** combine les avantages : on fait quelques steps d'évaluation (pas jusqu'à convergence) puis on améliore. C'est un compromis efficace.
```

---

## 5. Complexité algorithmique

### 5.1 Value Iteration

**Par itération :**
- Pour chaque état $s$ : $O(|A| \cdot |S|)$ opérations (max sur actions, somme sur transitions)
- Total par itération : $O(|S| \cdot |A| \cdot |S|) = O(|S|^2 |A|)$

**Nombre d'itérations :** $O\left(\log \frac{1}{\theta(1-\gamma)}\right)$ pour atteindre précision $\theta$

**Complexité totale :** $O\left(|S|^2 |A| \log \frac{1}{\theta(1-\gamma)}\right)$

### 5.2 Policy Iteration

**Policy Evaluation :**
- Méthode itérative : $O(|S|^2)$ par sweep, plusieurs sweeps
- Méthode directe : $O(|S|^3)$ (inversion matricielle)

**Policy Improvement :** $O(|S|^2 |A|)$

**Complexité totale :** $O(k \cdot |S|^3)$ où $k$ est le nombre d'améliorations (souvent petit)

---

## 6. Exercices théoriques

### Exercice 1 — Value Iteration sur gridworld

```{admonition} Exercice
:class: tip

Considérons un **gridworld 1D** avec 4 états : $S = \{s_1, s_2, s_3, s_4\}$

**Dynamique (déterministe) :**
- Actions : $A = \{\text{gauche}, \text{droite}\}$
- "droite" depuis $s_i$ → $s_{i+1}$ (si existe, sinon reste en place)
- "gauche" depuis $s_i$ → $s_{i-1}$ (si existe, sinon reste en place)

**Rewards :**
- $R(s_4, \text{droite}) = +10$ (goal atteint)
- $R(s, a) = 0$ pour toutes les autres transitions

**Paramètres :** $\gamma = 0.9$

**Effectuez 3 itérations de Value Iteration** en partant de $V_0(s) = 0$ pour tout $s$.

Calculez $V_1$, $V_2$, $V_3$.
```

````{dropdown} Solution — Exercice 1

**Itération 0 :** $V_0 = [0, 0, 0, 0]$

**Itération 1 :**

Pour chaque état, calculer $V_1(s) = \max_a \sum_{s'} P(s'|s,a)[R(s,a) + \gamma V_0(s')]$

- $V_1(s_1) = \max\{0 + 0.9 \times 0, \quad 0 + 0.9 \times 0\} = 0$
- $V_1(s_2) = \max\{0 + 0.9 \times 0, \quad 0 + 0.9 \times 0\} = 0$
- $V_1(s_3) = \max\{0 + 0.9 \times 0, \quad 0 + 0.9 \times 0\} = 0$
- $V_1(s_4) = \max\{0 + 0.9 \times 0, \quad 10 + 0.9 \times 0\} = 10$

$$V_1 = [0, 0, 0, 10]$$

**Itération 2 :**

- $V_2(s_1) = \max\{0, 0 + 0.9 \times 0\} = 0$
- $V_2(s_2) = \max\{0 + 0.9 \times 0, \quad 0 + 0.9 \times 0\} = 0$
- $V_2(s_3) = \max\{0 + 0.9 \times 0, \quad 0 + 0.9 \times 10\} = 9$
- $V_2(s_4) = \max\{0 + 0.9 \times 0, \quad 10 + 0.9 \times 10\} = 19$

$$V_2 = [0, 0, 9, 19]$$

**Itération 3 :**

- $V_3(s_1) = 0$
- $V_3(s_2) = \max\{0, \quad 0 + 0.9 \times 9\} = 8.1$
- $V_3(s_3) = \max\{0 + 0.9 \times 8.1, \quad 0 + 0.9 \times 19\} = \max\{7.29, 17.1\} = 17.1$
- $V_3(s_4) = \max\{0 + 0.9 \times 17.1, \quad 10 + 0.9 \times 19\} = \max\{15.39, 27.1\} = 27.1$

$$V_3 = [0, 8.1, 17.1, 27.1]$$

**Observation :** Les valeurs se **propagent** de droite à gauche depuis l'état goal. Chaque itération propage la valeur d'un état supplémentaire vers la gauche.
````

---

### Exercice 2 — Policy Evaluation analytique

```{admonition} Exercice
:class: tip

Soit un MDP à 2 états avec une policy fixe $\pi$.

**États :** $S = \{s_A, s_B\}$

**Policy :** $\pi(s_A) = \text{action } a$, $\pi(s_B) = \text{action } b$

**Transitions sous $\pi$ :**
- Depuis $s_A$ : aller en $s_B$ avec probabilité 1, reward = 2
- Depuis $s_B$ : aller en $s_A$ avec probabilité 1, reward = 4

**Paramètres :** $\gamma = 0.8$

**Calculez analytiquement** $V^{\pi}(s_A)$ et $V^{\pi}(s_B)$ en résolvant le système d'équations de Bellman.
```

````{dropdown} Solution — Exercice 2

**Système d'équations de Bellman :**

$$V^{\pi}(s_A) = R(s_A, \pi(s_A)) + \gamma \sum_{s'} P(s'|s_A, \pi(s_A)) V^{\pi}(s')$$

$$V^{\pi}(s_A) = 2 + 0.8 \times V^{\pi}(s_B)$$

$$V^{\pi}(s_B) = 4 + 0.8 \times V^{\pi}(s_A)$$

**Résolution par substitution :**

De la première équation : $V^{\pi}(s_A) = 2 + 0.8 V^{\pi}(s_B)$

Substituons dans la seconde :

$$V^{\pi}(s_B) = 4 + 0.8(2 + 0.8 V^{\pi}(s_B))$$

$$V^{\pi}(s_B) = 4 + 1.6 + 0.64 V^{\pi}(s_B)$$

$$V^{\pi}(s_B) - 0.64 V^{\pi}(s_B) = 5.6$$

$$0.36 V^{\pi}(s_B) = 5.6$$

$$V^{\pi}(s_B) = \frac{5.6}{0.36} = \frac{140}{9} \approx 15.56$$

Puis :

$$V^{\pi}(s_A) = 2 + 0.8 \times 15.56 = 2 + 12.45 = 14.45$$

**Vérification :**

$$V^{\pi}(s_A) = 2 + 0.8 \times 15.56 = 14.45$$ ✓

$$V^{\pi}(s_B) = 4 + 0.8 \times 14.45 = 4 + 11.56 = 15.56$$ ✓

**Solution :**

$$V^{\pi}(s_A) = \frac{130}{9} \approx 14.44$$

$$V^{\pi}(s_B) = \frac{140}{9} \approx 15.56$$
````

---

### Exercice 3 — Policy Improvement step

```{admonition} Exercice
:class: tip

On a évalué une policy $\pi$ et obtenu :

| État | $V^{\pi}(s)$ |
|------|--------------|
| $s_1$ | 10.0 |
| $s_2$ | 15.0 |
| $s_3$ | 8.0 |

**Transitions et rewards :**

Depuis $s_1$ :
- Action $a_1$ : 70% → $s_2$, 30% → $s_3$, reward = 1
- Action $a_2$ : 100% → $s_1$, reward = 5

Depuis $s_2$ :
- Action $a_1$ : 50% → $s_1$, 50% → $s_3$, reward = 2
- Action $a_2$ : 100% → $s_3$, reward = 0

**Paramètres :** $\gamma = 0.9$

**Policy actuelle :** $\pi(s_1) = a_1$, $\pi(s_2) = a_1$

**Questions :**

a) Calculez $Q^{\pi}(s_1, a_1)$ et $Q^{\pi}(s_1, a_2)$

b) Quelle action devrait choisir la nouvelle policy améliorée pour $s_1$ ?

c) Calculez $Q^{\pi}(s_2, a_1)$ et $Q^{\pi}(s_2, a_2)$

d) La policy actuelle doit-elle être changée pour $s_2$ ?
```

````{dropdown} Solution — Exercice 3

**a) Q-values pour $s_1$ :**

$$Q^{\pi}(s_1, a_1) = R(s_1, a_1) + \gamma \sum_{s'} P(s'|s_1, a_1) V^{\pi}(s')$$

$$= 1 + 0.9 [0.7 \times 15.0 + 0.3 \times 8.0]$$

$$= 1 + 0.9 [10.5 + 2.4]$$

$$= 1 + 0.9 \times 12.9 = 1 + 11.61 = 12.61$$

$$Q^{\pi}(s_1, a_2) = 5 + 0.9 \times 1.0 \times 10.0$$

$$= 5 + 9.0 = 14.0$$

**b) Amélioration pour $s_1$ :**

La nouvelle policy choisit : $\pi'(s_1) = \arg\max_a Q^{\pi}(s_1, a)$

Puisque $Q^{\pi}(s_1, a_2) = 14.0 > Q^{\pi}(s_1, a_1) = 12.61$ :

$$\pi'(s_1) = a_2$$

La policy doit être **changée** de $a_1$ à $a_2$ pour $s_1$.

**c) Q-values pour $s_2$ :**

$$Q^{\pi}(s_2, a_1) = 2 + 0.9 [0.5 \times 10.0 + 0.5 \times 8.0]$$

$$= 2 + 0.9 \times 9.0 = 2 + 8.1 = 10.1$$

$$Q^{\pi}(s_2, a_2) = 0 + 0.9 \times 1.0 \times 8.0$$

$$= 7.2$$

**d) Amélioration pour $s_2$ :**

Puisque $Q^{\pi}(s_2, a_1) = 10.1 > Q^{\pi}(s_2, a_2) = 7.2$ :

$$\pi'(s_2) = a_1$$

La policy actuelle ($\pi(s_2) = a_1$) est **déjà optimale** pour $s_2$. Pas de changement nécessaire.
````

---

### Exercice 4 — Convergence de Value Iteration

```{admonition} Exercice
:class: tip

On effectue Value Iteration avec $\gamma = 0.9$, $\theta = 0.01$.

Après l'itération $k$, on mesure : $\|V_{k+1} - V_k\|_\infty = 0.05$

**Questions :**

a) L'algorithme doit-il continuer ou peut-il s'arrêter ?

b) Donnez une borne supérieure sur l'erreur $\|V_k - V^*\|_\infty$

c) Si $\|V_{k+1} - V_k\|_\infty = 0.005$, quelle serait la nouvelle borne ?
```

````{dropdown} Solution — Exercice 4

**a) Critère d'arrêt :**

On arrête quand $\|V_{k+1} - V_k\|_\infty < \theta$

Ici : $0.05 > 0.01$

L'algorithme doit **continuer** (pas encore convergé).

**b) Borne sur l'erreur :**

Le théorème de convergence nous donne :

$$\|V_k - V^*\|_\infty \leq \frac{\gamma}{1-\gamma} \|V_{k+1} - V_k\|_\infty$$

Avec $\gamma = 0.9$ et $\|V_{k+1} - V_k\|_\infty = 0.05$ :

$$\|V_k - V^*\|_\infty \leq \frac{0.9}{1-0.9} \times 0.05 = \frac{0.9}{0.1} \times 0.05$$

$$= 9 \times 0.05 = 0.45$$

**Borne supérieure :** L'erreur est au plus **0.45**.

**c) Nouvelle borne avec $\|V_{k+1} - V_k\|_\infty = 0.005$ :**

$$\|V_k - V^*\|_\infty \leq 9 \times 0.005 = 0.045$$

Avec ce changement, l'algorithme peut s'arrêter (0.005 < 0.01) et l'erreur garantie serait **au plus 0.045**.
````

---

### Exercice 5 — Nombre d'itérations nécessaires

```{admonition} Exercice
:class: tip

On veut garantir une précision $\epsilon = 0.001$ dans Value Iteration.

**Paramètres :** $\gamma = 0.95$, valeur maximale possible $V_{\max} = 100$

**Combien d'itérations** $k$ sont nécessaires pour garantir $\|V_k - V^*\|_\infty < \epsilon$ ?

**Indice :** Utilisez $\|V_k - V^*\|_\infty \leq \gamma^k \frac{V_{\max}}{1-\gamma}$
```

````{dropdown} Solution — Exercice 5

**Formule de convergence :**

En partant de $V_0 = 0$, on a :

$$\|V_k - V^*\|_\infty \leq \gamma^k \|V_0 - V^*\|_\infty \leq \gamma^k \frac{V_{\max}}{1-\gamma}$$

**Condition pour précision $\epsilon$ :**

$$\gamma^k \frac{V_{\max}}{1-\gamma} < \epsilon$$

$$\gamma^k < \epsilon \frac{1-\gamma}{V_{\max}}$$

$$k \log(\gamma) < \log\left(\epsilon \frac{1-\gamma}{V_{\max}}\right)$$

Attention : $\log(\gamma) < 0$ car $\gamma < 1$, donc l'inégalité se renverse :

$$k > \frac{\log\left(\epsilon \frac{1-\gamma}{V_{\max}}\right)}{\log(\gamma)}$$

**Application numérique :**

$$k > \frac{\log\left(0.001 \times \frac{0.05}{100}\right)}{\log(0.95)}$$

$$= \frac{\log(0.0000005)}{\log(0.95)}$$

$$= \frac{\log(5 \times 10^{-7})}{\log(0.95)}$$

$$= \frac{-14.51}{-0.0513} \approx 283$$

**Réponse :** Il faut environ **283 itérations** pour garantir la précision souhaitée.

**Note :** En pratique, la convergence est souvent plus rapide car cette borne est pessimiste.
````

---

## 7. Visualisation de la convergence

```{admonition} Analyse de la convergence
:class: note

Pour visualiser la convergence de Value Iteration ou Policy Iteration, on peut tracer :

1. **Norme infinie des changements :** $\|V_{k+1} - V_k\|_\infty$ vs itération $k$
   - Devrait décroître géométriquement : ligne droite en échelle log
   - Pente = $\log(\gamma)$

2. **Erreur absolue (si $V^*$ connu) :** $\|V_k - V^*\|_\infty$ vs $k$
   - Montre la vraie distance à l'optimal

3. **Valeur par état :** $V_k(s)$ pour quelques états représentatifs vs $k$
   - Montre comment les valeurs évoluent individuellement

**Décroissance géométrique :**

$$\|V_{k+1} - V_k\|_\infty \leq \gamma \|V_k - V_{k-1}\|_\infty$$

En échelle logarithmique : $\log(\|V_{k+1} - V_k\|) \leq \log(\gamma) + \log(\|V_k - V_{k-1}\|)$

Cela forme une **droite de pente** $\log(\gamma) < 0$.
```

---

## 8. Limitations du Dynamic Programming

```{admonition} Quand DP n'est pas applicable
:class: warning

1. **Modèle inconnu** : DP requiert $P(s'|s,a)$ et $R(s,a)$ explicites
   - Dans beaucoup d'applications réelles, le modèle n'est pas disponible
   - Solution : méthodes **model-free** (Q-Learning, SARSA, etc.)

2. **Grands espaces d'états** : Complexité $O(|S|^2 |A|)$ par itération
   - Pour $|S| = 10^6$ états : impraticable sans approximation
   - Solution : **function approximation** (Deep RL)

3. **Espaces continus** : Impossibilité de représenter toutes les valeurs
   - Solution : discrétisation ou approximateurs de fonctions

4. **MDPs partiellement observables** : DP standard assume observabilité complète
   - Solution : algorithmes pour POMDPs (plus complexes)
```

---

## 9. Points clés à retenir

```{admonition} Concepts essentiels
:class: important

1. **DP = planification** avec modèle complet du MDP
2. **Value Iteration** : itère l'équation de Bellman optimale directement
3. **Policy Iteration** : alterne évaluation et amélioration
4. **Convergence garantie** : pour MDPs finis avec $\gamma < 1$
5. **Convergence géométrique** : taux $\gamma$ pour Value Iteration
6. **Complexité** : $O(|S|^2 |A|)$ par itération
7. **Limitation** : nécessite connaissance du modèle
8. **Backup operations** : mises à jour exploitant la structure récursive
```

---

## 10. Extensions et variantes

```{admonition} Algorithmes dérivés
:class: note

- **Asynchronous DP** : mise à jour des états dans un ordre arbitraire (pas de sweep complet)
- **Prioritized Sweeping** : priorité aux états avec plus grand changement potentiel
- **Modified Policy Iteration** : évaluation partielle + amélioration (compromis)
- **Gauss-Seidel Value Iteration** : utilise immédiatement les nouvelles valeurs
- **Real-Time Dynamic Programming (RTDP)** : focus sur états atteignables depuis état initial
```