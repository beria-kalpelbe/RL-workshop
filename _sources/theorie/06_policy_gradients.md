# 07 — Policy Gradient Methods (REINFORCE, Actor-Critic)

## Introduction

Jusqu'à présent, nous avons vu des méthodes basées sur les **fonctions de valeur** (value-based methods) comme Q-Learning et DQN, qui apprennent d'abord une fonction de valeur puis en déduisent une policy (généralement greedy). Les **méthodes par gradient de politique** (policy gradient methods) adoptent une approche différente : elles apprennent **directement** une policy paramétrée $\pi(a|s;\theta)$ en optimisant l'espérance du return via gradient ascent.

```{admonition} Pourquoi les Policy Gradient Methods ?
:class: tip
- **Actions continues** : Gèrent naturellement les espaces d'actions continus (difficile pour Q-Learning)
- **Policies stochastiques** : Peuvent apprendre des policies intrinsèquement stochastiques
- **Convergence** : Garanties de convergence plus fortes (vers minimum local au moins)
- **Stabilité** : Changements graduels de policy (pas de changements brusques comme greedy)
```

```{admonition} Value-Based vs Policy-Based
:class: note
**Value-Based** (Q-Learning, DQN) :
- Apprend $Q(s,a)$ ou $V(s)$
- Policy dérivée implicitement (argmax)
- Difficile pour actions continues
- Peut avoir des changements brusques de policy

**Policy-Based** (Policy Gradient) :
- Apprend directement $\pi(a|s;\theta)$
- Optimisation explicite de la policy
- Naturel pour actions continues
- Changements graduels et stables
```

---

## 1. Théorème du Policy Gradient

### Objectif d'Optimisation

On cherche à maximiser l'espérance du return total :

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G_0] = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t R_{t+1}\right]$$

où $\tau = (S_0, A_0, R_1, S_1, A_1, R_2, \ldots)$ est une trajectoire générée par la policy $\pi_\theta$.

### Le Théorème

```{admonition} Policy Gradient Theorem (Sutton et al., 2000)
:class: important
Le gradient de l'objectif par rapport aux paramètres de la policy est :

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(A_t|S_t) \, G_t \right]$$

où $G_t = \sum_{k=t}^{T} \gamma^{k-t} R_{k+1}$ est le return depuis le temps $t$.

**Interprétation intuitive** :
- $\nabla_\theta \log \pi_\theta(A_t|S_t)$ : Direction qui augmente la probabilité de l'action $A_t$
- $G_t$ : "Score" de cette action (return obtenu après l'avoir prise)
- Si $G_t$ est élevé → augmenter $\pi(A_t|S_t)$
- Si $G_t$ est faible → diminuer $\pi(A_t|S_t)$
```

### Dérivation Intuitive

```{admonition} Pourquoi cette forme ?
:class: note
Le gradient $\nabla_\theta \log \pi_\theta(a|s)$ a une propriété utile appelée **log-derivative trick** :

$$\nabla_\theta \pi_\theta(a|s) = \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s)$$

Cela permet de transformer le gradient d'une espérance en espérance d'un gradient, ce qui est essentiel pour l'estimation Monte Carlo.

**Propriétés importantes** :
- Ne nécessite pas de modèle de l'environnement
- Fonctionne avec des espaces d'actions discrets ou continus
- Peut être estimé par échantillonnage (Monte Carlo)
```

---

## 2. REINFORCE Algorithm

REINFORCE (Williams, 1992) est l'algorithme de policy gradient le plus simple, utilisant des estimations Monte Carlo du return.

### Algorithme de Base

```
Algorithme : REINFORCE (Monte Carlo Policy Gradient)
────────────────────────────────────────────────────────
Entrée : learning rate α, discount γ
Sortie : policy paramétrée π(a|s;θ)

// Initialisation
Initialiser paramètres de policy θ aléatoirement

// Apprentissage
répéter pour chaque épisode :
    
    // Générer un épisode complet
    Générer épisode S₀, A₀, R₁, S₁, A₁, R₂, ..., S_T
    selon policy π(·|·;θ)
    
    // Mise à jour après l'épisode
    pour t = 0 à T-1 faire :
        
        // Calculer return depuis t
        G_t ← 0
        pour k = t à T-1 faire :
            G_t ← G_t + γ^(k-t) · R_(k+1)
        fin pour
        
        // Mise à jour policy gradient
        θ ← θ + α · ∇_θ log π_θ(A_t|S_t) · G_t
        
    fin pour
    
jusqu'à convergence

retourner θ
```

```{admonition} Caractéristiques de REINFORCE
:class: note
**Avantages** :
- Simple et intuitif
- Unbiased gradient estimate (estimation non biaisée)
- Fonctionne avec actions discrètes et continues
- Garantie de convergence vers minimum local

**Inconvénients** :
- **Variance très élevée** : Les returns $G_t$ peuvent varier énormément
- Nécessite des épisodes complets (on-policy, Monte Carlo)
- Apprentissage lent dû à la haute variance
- Sample inefficient (besoin de nombreux épisodes)
```

### Paramétrisations Courantes

```{admonition} Représentations de Policy
:class: tip

**Actions Discrètes** (ex: CartPole, Atari) :
- Softmax policy : $\pi(a|s;\theta) = \frac{\exp(h(s,a;\theta))}{\sum_{a'} \exp(h(s,a';\theta))}$
- Réseau de neurones : $s \rightarrow$ hidden layers $\rightarrow$ softmax sur actions

**Actions Continues** (ex: robotique, contrôle) :
- Gaussian policy : $\pi(a|s;\theta) = \mathcal{N}(a|\mu_\theta(s), \sigma^2_\theta(s))$
- Réseau de neurones produit $\mu_\theta(s)$ et $\sigma_\theta(s)$
- Log-probabilité : $\log \pi(a|s;\theta) = -\frac{1}{2}\left[\frac{(a-\mu_\theta(s))^2}{\sigma^2_\theta(s)} + \log(2\pi\sigma^2_\theta(s))\right]$
```

---

## 3. Baseline et Réduction de Variance

### Le Problème de Variance

```{admonition} Pourquoi la Variance est un Problème
:class: danger
Dans REINFORCE, $G_t$ peut varier énormément entre épisodes même pour le même état-action :
- Un épisode chanceux → $G_t$ très élevé → sur-ajustement
- Un épisode malchanceux → $G_t$ très faible → sous-estimation
- Résultat : Apprentissage instable et lent

**Variance élevée** → Besoin de beaucoup d'échantillons → Inefficacité
```

### Introduction d'une Baseline

On peut soustraire une **baseline** $b(S_t)$ du return sans introduire de biais :

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(A_t|S_t) \left(G_t - b(S_t)\right)\right]$$

```{admonition} Pourquoi la Baseline Ne Change Pas l'Espérance
:class: note
Preuve que soustraire $b(S_t)$ ne biaise pas le gradient :

$$\mathbb{E}_{A_t \sim \pi}\left[\nabla_\theta \log \pi_\theta(A_t|S_t) \cdot b(S_t)\right]$$
$$= b(S_t) \sum_{a} \pi_\theta(a|S_t) \nabla_\theta \log \pi_\theta(a|S_t)$$
$$= b(S_t) \sum_{a} \nabla_\theta \pi_\theta(a|S_t) = b(S_t) \nabla_\theta 1 = 0$$

La baseline réduit la variance sans changer l'espérance !
```

### Choix de Baseline Optimal

```{admonition} Baselines Courantes
:class: tip

**1. Baseline Constante** :
- $b = \bar{G}$ (moyenne des returns observés)
- Simple mais sous-optimal

**2. State-Dependent Baseline** :
- $b(s) = V(s)$ (valeur de l'état)
- **Optimal en théorie** : minimise la variance
- Donne l'**avantage** : $A_t = G_t - V(S_t)$

**3. Learned Baseline** :
- Approximer $V(s;w)$ avec un réseau de neurones
- Mise à jour simultanée avec la policy
- Conduit naturellement aux méthodes Actor-Critic
```

### REINFORCE avec Baseline

```
Algorithme : REINFORCE avec Baseline
──────────────────────────────────────────────────────────
Entrée : learning rates α_θ, α_w, discount γ
Sortie : policy π(a|s;θ), value function V(s;w)

// Initialisation
Initialiser θ et w aléatoirement

// Apprentissage
répéter pour chaque épisode :
    
    Générer épisode S₀, A₀, R₁, ..., S_T selon π(·|·;θ)
    
    pour t = 0 à T-1 faire :
        
        // Calculer return
        G_t ← Σ_(k=t)^(T-1) γ^(k-t) · R_(k+1)
        
        // Calculer avantage (advantage)
        δ_t ← G_t - V(S_t; w)
        
        // Mise à jour value function (critic)
        w ← w + α_w · δ_t · ∇_w V(S_t; w)
        
        // Mise à jour policy (actor)
        θ ← θ + α_θ · ∇_θ log π_θ(A_t|S_t) · δ_t
        
    fin pour
    
jusqu'à convergence

retourner θ, w
```

---

## 4. Actor-Critic Methods

Les méthodes **Actor-Critic** combinent le meilleur des deux mondes : policy gradient (actor) et value function approximation (critic).

```{admonition} Architecture Actor-Critic
:class: important
**Actor** (politique) :
- Paramètres : $\theta$
- Apprend la policy $\pi(a|s;\theta)$
- Mise à jour par policy gradient

**Critic** (critique) :
- Paramètres : $w$
- Apprend la value function $V(s;w)$ ou $Q(s,a;w)$
- Mise à jour par TD learning

**Avantage** : Le critic réduit la variance du gradient en fournissant une baseline apprise.
```

### One-Step Actor-Critic

Contrairement à REINFORCE qui attend la fin de l'épisode, Actor-Critic peut faire des mises à jour **à chaque step** en utilisant bootstrapping (comme TD).

```
Algorithme : One-Step Actor-Critic
────────────────────────────────────────────────────────
Entrée : learning rates α_θ, α_w, discount γ
Sortie : policy π(a|s;θ), value function V(s;w)

// Initialisation
Initialiser θ et w aléatoirement

// Apprentissage
répéter pour chaque épisode :
    Initialiser S
    
    répéter pour chaque step (jusqu'à S terminal) :
        
        // Actor : sélectionner action
        A ~ π(·|S; θ)
        
        // Exécuter action
        Exécuter A, observer R, S'
        
        // Critic : calculer TD error
        si S' est terminal alors :
            δ ← R - V(S; w)
        sinon :
            δ ← R + γ·V(S'; w) - V(S; w)
        fin si
        
        // Mise à jour Critic (TD learning)
        w ← w + α_w · δ · ∇_w V(S; w)
        
        // Mise à jour Actor (policy gradient)
        θ ← θ + α_θ · ∇_θ log π_θ(A|S) · δ
        
        // Transition
        S ← S'
        
    jusqu'à S terminal
    
jusqu'à convergence

retourner θ, w
```

```{admonition} REINFORCE vs Actor-Critic
:class: note
**REINFORCE** :
- Utilise $G_t$ (return complet Monte Carlo)
- Estimation non biaisée mais haute variance
- Nécessite épisodes complets
- Mise à jour en batch après épisode

**Actor-Critic** :
- Utilise $\delta_t = R + \gamma V(S') - V(S)$ (TD error)
- Estimation biaisée mais faible variance
- Peut faire des mises à jour online (chaque step)
- Plus efficace en termes d'échantillons
```

### Avantage de TD sur Monte Carlo

```{list-table}
:header-rows: 1
:widths: 25 37 38

* - Aspect
  - Monte Carlo (REINFORCE)
  - Temporal Difference (Actor-Critic)
* - **Biais**
  - Aucun (unbiased)
  - Présent (dépend de $V$)
* - **Variance**
  - Élevée
  - Faible
* - **Vitesse**
  - Lente
  - Rapide
* - **Online/Offline**
  - Nécessite épisodes complets
  - Mises à jour à chaque step
* - **Environnements**
  - Épisodiques seulement
  - Épisodiques et continus
```

---

## 5. Avantages et Applications

```{admonition} Avantages des Policy Gradient Methods
:class: tip

**1. Actions Continues** :
- Gère naturellement les espaces d'actions continus
- Essentiel pour robotique, contrôle de moteurs, etc.

**2. Policies Stochastiques** :
- Peut apprendre des stratégies intrinsèquement stochastiques
- Utile pour jeux à information partielle (ex: pierre-papier-ciseaux)

**3. Stabilité** :
- Changements graduels de policy (pas de changements brusques)
- Garanties de convergence (vers minimum local)

**4. Simplicité** :
- Pas besoin d'operator max (difficile en continu)
- Architecture neurale directe : états → probabilités d'actions
```

```{admonition} Limitations
:class: warning

**1. Convergence Locale** :
- Garantie seulement vers minimum local (pas global)
- Sensible à l'initialisation

**2. Sample Inefficiency** :
- Nécessite beaucoup d'interactions avec l'environnement
- Surtout pour REINFORCE vanilla

**3. Hyperparamètres Sensibles** :
- Learning rates critiques ($\alpha_\theta$, $\alpha_w$)
- Nécessite tuning soigneux

**4. Exploration** :
- Dépend de la stochasticité de la policy
- Peut nécessiter mécanismes d'exploration additionnels (entropy bonus)
```

---

## Exercices

### Exercice 1 : Calcul du Policy Gradient

**Question** : Considérez une policy softmax sur deux actions $a \in \{0, 1\}$ :

$$\pi(a|s;\theta) = \frac{\exp(\theta_a)}{\exp(\theta_0) + \exp(\theta_1)}$$

avec paramètres $\theta = [\theta_0, \theta_1]^T$.

(a) Calculez $\nabla_\theta \log \pi(a=0|s;\theta)$.

(b) Si vous observez une trajectoire $(s, a=0, G=10)$, quelle est la direction de mise à jour de $\theta$ avec learning rate $\alpha=0.1$ ?

```{admonition} Solution
:class: dropdown

**(a)** Calcul du gradient :

Pour la policy softmax :
$$\pi(0|s;\theta) = \frac{\exp(\theta_0)}{\exp(\theta_0) + \exp(\theta_1)}$$

Le log-probabilité :
$$\log \pi(0|s;\theta) = \theta_0 - \log(\exp(\theta_0) + \exp(\theta_1))$$

Gradient par rapport à $\theta_0$ :
$$\frac{\partial}{\partial \theta_0} \log \pi(0|s;\theta) = 1 - \frac{\exp(\theta_0)}{\exp(\theta_0) + \exp(\theta_1)} = 1 - \pi(0|s;\theta)$$

Gradient par rapport à $\theta_1$ :
$$\frac{\partial}{\partial \theta_1} \log \pi(0|s;\theta) = 0 - \frac{\exp(\theta_1)}{\exp(\theta_0) + \exp(\theta_1)} = -\pi(1|s;\theta)$$

Donc :
$$\nabla_\theta \log \pi(0|s;\theta) = \begin{bmatrix} 1 - \pi(0|s;\theta) \\ -\pi(1|s;\theta) \end{bmatrix}$$

**Interprétation** : Le gradient augmente la probabilité de l'action choisie (0) et diminue celle de l'autre action (1).

**(b)** Mise à jour :

$$\theta \leftarrow \theta + \alpha \cdot \nabla_\theta \log \pi(0|s;\theta) \cdot G$$

Si $\theta = [0, 0]^T$ initialement, alors $\pi(0|s;\theta) = \pi(1|s;\theta) = 0.5$.

$$\nabla_\theta \log \pi(0|s;\theta) = \begin{bmatrix} 1 - 0.5 \\ -0.5 \end{bmatrix} = \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix}$$

Mise à jour :
$$\theta \leftarrow \begin{bmatrix} 0 \\ 0 \end{bmatrix} + 0.1 \cdot \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix} \cdot 10 = \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix}$$

**Résultat** : $\theta_0$ augmente (favorise action 0) et $\theta_1$ diminue (défavorise action 1), car l'action 0 a donné un bon return ($G=10$).
```

---

### Exercice 2 : Variance de REINFORCE

**Question** : Expliquez pourquoi REINFORCE a une variance élevée et comment une baseline $b(s) = V(s)$ réduit cette variance sans introduire de biais. Donnez un exemple numérique simple.

```{admonition} Solution
:class: dropdown

**Pourquoi variance élevée ?**

Dans REINFORCE, on utilise $G_t = \sum_{k=t}^T \gamma^{k-t} R_{k+1}$ comme estimation du gradient. Même pour le même état-action, $G_t$ peut varier énormément selon :
- Le hasard dans l'environnement
- Les actions futures (qui sont stochastiques)
- La longueur de l'épisode

**Exemple numérique** :

Supposons deux épisodes depuis le même état $s$ avec même action $a$ :
- Épisode 1 : $G = 100$ (chanceux)
- Épisode 2 : $G = 10$ (malchanceux)

Sans baseline :
- Update 1 : $\theta \leftarrow \theta + \alpha \cdot \nabla \log \pi(a|s) \cdot 100$
- Update 2 : $\theta \leftarrow \theta + \alpha \cdot \nabla \log \pi(a|s) \cdot 10$

Énorme différence ! Variance $= \text{Var}(G) = \text{Var}([100, 10]) = 2025$

**Avec baseline** $b(s) = V(s) = 50$ (valeur moyenne de l'état) :

- Update 1 : $\theta \leftarrow \theta + \alpha \cdot \nabla \log \pi(a|s) \cdot (100 - 50) = 50$
- Update 2 : $\theta \leftarrow \theta + \alpha \cdot \nabla \log \pi(a|s) \cdot (10 - 50) = -40$

Variance réduite : $\text{Var}(G - b) = \text{Var}([50, -40]) = 2025$ → Variance toujours présente mais les valeurs sont plus "centrées".

**Pourquoi pas de biais ?**

La baseline $b(s)$ ne dépend pas de l'action $a$, donc :
$$\mathbb{E}_{a \sim \pi}[\nabla \log \pi(a|s) \cdot b(s)] = b(s) \cdot \mathbb{E}_{a \sim \pi}[\nabla \log \pi(a|s)]$$

Or $\mathbb{E}_{a \sim \pi}[\nabla \log \pi(a|s)] = \nabla \mathbb{E}_{a \sim \pi}[\pi(a|s)] = \nabla 1 = 0$.

Donc soustraire $b(s)$ ne change pas l'espérance du gradient !

**Baseline optimale** : Théoriquement, la baseline qui minimise la variance est $b(s) = V(s)$, car elle "centre" les returns autour de leur valeur attendue.
```

---

### Exercice 3 : Implémentation de REINFORCE

**Question** : Vous implémentez REINFORCE sur CartPole. Après 100 épisodes, votre agent n'apprend pas (performance plateau). Listez 3 causes possibles et leurs solutions.

```{admonition} Solution
:class: dropdown

**Causes possibles et solutions** :

**1. Learning Rate Inadéquat**
- **Symptôme** : Pas d'amélioration ou instabilité
- **Cause** : $\alpha$ trop grand (instabilité) ou trop petit (apprentissage trop lent)
- **Solution** : 
  - Essayer $\alpha \in \{10^{-4}, 3 \times 10^{-4}, 10^{-3}, 3 \times 10^{-3}\}$
  - Utiliser learning rate decay : $\alpha_t = \alpha_0 / (1 + t/1000)$
  - Monitorer la norme des gradients

**2. Variance Trop Élevée**
- **Symptôme** : Performance très erratique, oscillations importantes
- **Cause** : REINFORCE vanilla sans baseline
- **Solution** :
  - Ajouter une baseline apprise : $V(s;w)$
  - Utiliser plusieurs épisodes avant la mise à jour (batch)
  - Normaliser les returns : $G_t \leftarrow (G_t - \mu) / (\sigma + \epsilon)$

**3. Exploration Insuffisante**
- **Symptôme** : Agent bloqué dans comportement sous-optimal
- **Cause** : Policy trop déterministe trop tôt
- **Solution** :
  - Ajouter un bonus d'entropie : $J(\theta) = \mathbb{E}[G] + \beta H(\pi)$ où $H$ est l'entropie
  - Augmenter la température du softmax temporairement
  - Initialiser les poids du réseau avec petites valeurs

**4. Architecture Inadéquate (bonus)**
- **Cause** : Réseau trop simple ou trop complexe
- **Solution** :
  - Pour CartPole : 2 hidden layers de 64 neurones suffisent
  - Utiliser activation ReLU ou tanh
  - Vérifier que le réseau peut représenter la policy désirée

**5. Discount Factor Sous-Optimal (bonus)**
- **Cause** : $\gamma$ trop proche de 1 ou trop petit
- **Solution** :
  - Pour environnements courts (CartPole) : $\gamma \in [0.95, 0.99]$
  - Pour environnements longs : $\gamma > 0.99$

**Debugging pratique** :
```python
# Vérifier les gradients
print(f"Grad norm: {torch.nn.utils.clip_grad_norm_(params, 1.0)}")

# Monitorer les returns
print(f"Mean return: {np.mean(returns[-10:])}")

# Vérifier les log-probs
print(f"Log prob range: [{log_probs.min()}, {log_probs.max()}]")
```

### Exercice 4 : Actor-Critic vs REINFORCE

**Question** : Comparez Actor-Critic one-step avec REINFORCE sur les aspects suivants :
(a) Biais et variance
(b) Vitesse de convergence
(c) Applicabilité aux environnements continus (non-épisodiques)

```{admonition} Solution
:class: dropdown

**(a) Biais et Variance**

**REINFORCE** :
- **Biais** : Aucun (unbiased) car utilise le vrai return $G_t$
- **Variance** : Très élevée car $G_t$ dépend de toute la trajectoire future
- $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots$ accumule du bruit

**Actor-Critic** :
- **Biais** : Présent car utilise $\delta = R + \gamma V(S') - V(S)$ qui dépend de l'approximation $V$
- **Variance** : Beaucoup plus faible car dépend seulement de $R$ et $V(S')$ (pas toute la trajectoire)
- Bootstrap sur $V(S')$ au lieu d'attendre le return complet

**Trade-off** : Actor-Critic échange un peu de biais contre beaucoup moins de variance → convergence plus rapide en pratique.

**(b) Vitesse de Convergence**

**REINFORCE** :
- Apprentissage lent dû à la haute variance
- Nécessite beaucoup d'épisodes pour moyenner le bruit
- Mises à jour en batch après épisode complet
- Sample inefficient

**Actor-Critic** :
- Apprentissage plus rapide grâce à faible variance
- Mises à jour online à chaque step
- Meilleure utilisation des échantillons
- Sample efficient

**En pratique** : Actor-Critic converge typiquement 5-10x plus vite que REINFORCE sur la plupart des tâches.

**(c) Environnements Continus (non-épisodiques)**

**REINFORCE** :
- **Ne fonctionne pas** sur environnements continus
- Nécessite absolument des épisodes complets pour calculer $G_t$
- Limité aux tâches épisodiques

**Actor-Critic** :
- **Fonctionne parfaitement** sur environnements continus
- Pas besoin d'attendre la fin d'un épisode
- Peut faire des mises à jour infiniment avec TD learning
- Applicable à tous types d'environnements

**Exemple** : Contrôle de température d'un bâtiment (pas de "fin" naturelle) → Actor-Critic only.

**Tableau récapitulatif** :

| Aspect | REINFORCE | Actor-Critic |
|--------|-----------|--------------|
| Biais | Aucun | Oui |
| Variance | Très élevée | Faible |
| Convergence | Lente | Rapide |
| Sample efficiency | Faible | Élevée |
| Environnements continus | ✗ Non | ✓ Oui |
| Mise à jour | Après épisode | Chaque step |
| Complexité implémentation | Simple | Moyenne |
```

---

### Exercice 5 : Design d'une Gaussian Policy

**Question** : Pour un problème de contrôle continu (action $a \in \mathbb{R}$), vous utilisez une Gaussian policy :

$$\pi(a|s;\theta) = \mathcal{N}(a | \mu_\theta(s), \sigma^2)$$

où $\mu_\theta(s)$ est produit par un réseau de neurones et $\sigma$ est un hyperparamètre fixe.

(a) Écrivez l'expression de $\nabla_\theta \log \pi(a|s;\theta)$ pour cette policy.

(b) Quel est l'effet de $\sigma$ sur l'exploration ? Que se passe-t-il si $\sigma$ est trop grand ou trop petit ?

(c) Proposez une amélioration où $\sigma$ est aussi appris.

```{admonition} Solution
:class: dropdown

**(a)** Expression de $\nabla_\theta \log \pi(a|s;\theta)$ :

Pour une Gaussian policy :
$$\pi(a|s;\theta) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(a-\mu_\theta(s))^2}{2\sigma^2}\right)$$

Le log-probabilité :
$$\log \pi(a|s;\theta) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(a-\mu_\theta(s))^2}{2\sigma^2}$$

Comme $\sigma$ est fixe, seul le terme avec $\mu_\theta(s)$ dépend de $\theta$ :
$$\log \pi(a|s;\theta) = \text{constante} - \frac{(a-\mu_\theta(s))^2}{2\sigma^2}$$

Gradient :
$$\nabla_\theta \log \pi(a|s;\theta) = -\frac{1}{2\sigma^2} \cdot 2(a-\mu_\theta(s)) \cdot (-\nabla_\theta \mu_\theta(s))$$

$$= \frac{a - \mu_\theta(s)}{\sigma^2} \nabla_\theta \mu_\theta(s)$$

**Interprétation** : 
- Si $a > \mu_\theta(s)$ (action plus grande que la moyenne) : gradient pousse $\mu$ vers le haut
- Si $a < \mu_\theta(s)$ (action plus petite) : gradient pousse $\mu$ vers le bas
- L'ampleur dépend de l'écart $(a - \mu)$ et est inversement proportionnelle à $\sigma^2$

**(b)** Effet de $\sigma$ sur l'exploration :

**$\sigma$ contrôle l'exploration** :

**Si $\sigma$ trop grand** :
- ✗ Exploration excessive (actions très aléatoires)
- ✗ Échantillonne souvent des actions sous-optimales
- ✗ Apprentissage lent car signal bruité
- ✗ Agent n'exploite jamais vraiment la policy apprise
- Exemple : $\sigma = 10$ pour contrôle de pendule → actions complètement erratiques

**Si $\sigma$ trop petit** :
- ✗ Exploration insuffisante (policy quasi-déterministe)
- ✗ Convergence prématurée vers minimum local
- ✗ Ne découvre pas de bonnes stratégies alternatives
- ✗ Gradients très faibles si action échantillonnée loin de $\mu$
- Exemple : $\sigma = 0.01$ → agent fait presque toujours la même action

**$\sigma$ optimal** :
- ✓ Équilibre exploration-exploitation
- ✓ Généralement décroître $\sigma$ au cours de l'entraînement :
  - Début : $\sigma$ élevé pour explorer
  - Fin : $\sigma$ faible pour exploiter
- Typique : $\sigma \in [0.1, 1.0]$ selon l'échelle des actions

**(c)** Amélioration : Apprendre $\sigma$ (ou $\log \sigma$) :

**Approche 1 : Paramétrer $\sigma$ directement**

Le réseau produit à la fois $\mu_\theta(s)$ et $\sigma_\theta(s)$ :


État s → Neural Network → [μ_θ(s), log σ_θ(s)]
                           ↓           ↓
                         mean      exp(·) → σ_θ(s)


On paramétrise $\log \sigma$ plutôt que $\sigma$ pour garantir $\sigma > 0$.

**Gradient complet** :

$$\log \pi(a|s;\theta) = -\log \sigma_\theta(s) - \frac{1}{2}\log(2\pi) - \frac{(a-\mu_\theta(s))^2}{2\sigma^2_\theta(s)}$$

Gradient par rapport à $\mu$ (comme avant) :
$$\frac{\partial}{\partial \mu_\theta} \log \pi = \frac{a - \mu_\theta(s)}{\sigma^2_\theta(s)} \nabla_\theta \mu_\theta(s)$$

Gradient par rapport à $\log \sigma$ :
$$\frac{\partial}{\partial \log \sigma_\theta} \log \pi = -1 + \frac{(a-\mu_\theta(s))^2}{\sigma^2_\theta(s)}$$

**Interprétation** :
- Si $(a-\mu)^2 > \sigma^2$ : augmenter $\sigma$ (action loin de la moyenne)
- Si $(a-\mu)^2 < \sigma^2$ : diminuer $\sigma$ (action proche de la moyenne)
- Le réseau apprend automatiquement quand explorer vs exploiter

**Approche 2 : $\sigma$ dépendant de l'état**

Architecture :

```python
class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.mean = nn.Linear(64, action_dim)
        self.log_std = nn.Linear(64, action_dim)
    
    def forward(self, state):
        x = self.shared(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        std = torch.exp(log_std)
        return mean, std
```

**Avantages** :
- ✓ Exploration adaptative selon l'état
- ✓ Plus d'exploration dans états incertains
- ✓ Moins d'exploration dans états bien connus
- ✓ Convergence potentiellement plus rapide

**Approche 3 : $\sigma$ global appris**

$\sigma$ est un paramètre scalaire global appris (indépendant de $s$) :

```python
log_std = nn.Parameter(torch.zeros(action_dim))
```

**Avantages** :
- ✓ Plus simple (moins de paramètres)
- ✓ Suffit souvent en pratique
- ✓ Plus stable

**Recommandation pratique** :
- Commencer avec $\sigma$ global appris
- Si besoin de plus d'expressivité → $\sigma$ dépendant de l'état
- Toujours paramétrer $\log \sigma$ plutôt que $\sigma$ directement
---

## Points Clés à Retenir

```{admonition} Concepts Fondamentaux
:class: important

**1. Philosophie des Policy Gradients**
- Apprendre **directement** la policy $\pi(a|s;\theta)$ (pas indirectement via value function)
- Optimiser l'espérance du return : $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G_0]$
- Utiliser gradient ascent pour améliorer $\theta$

**2. Policy Gradient Theorem**
- Formule clé : $\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_t \nabla_\theta \log \pi_\theta(A_t|S_t) \, G_t\right]$
- Intuition : Augmenter probabilité des actions qui mènent à haut return
- Estimable par Monte Carlo (échantillonnage)

**3. Trade-off Biais-Variance**
- **REINFORCE** : Unbiased mais haute variance (Monte Carlo)
- **Actor-Critic** : Biaisé mais faible variance (Bootstrapping)
- En pratique : Actor-Critic converge plus vite
```

```{admonition} Techniques Essentielles
:class: tip

**Réduction de Variance** :
- **Baseline** $b(s)$ : Réduit variance sans biais
- Baseline optimale : $b(s) = V(s)$
- Conduit à l'**advantage** : $A(s,a) = Q(s,a) - V(s)$ ou $A \approx G_t - V(s)$

**Actor-Critic Architecture** :
- **Actor** : Policy $\pi(a|s;\theta)$ mise à jour par policy gradient
- **Critic** : Value function $V(s;w)$ mise à jour par TD learning
- Synergie : Critic fournit baseline pour réduire variance de l'Actor

**Mises à Jour** :
- REINFORCE : Après épisode complet (batch)
- Actor-Critic : À chaque step (online)
- Actor-Critic plus sample efficient
```

```{admonition} Avantages et Limitations
:class: note

**Quand Utiliser Policy Gradients** :
- ✓ **Actions continues** : Robotique, contrôle de moteurs, véhicules autonomes
- ✓ **Policies stochastiques** : Jeux avec stratégies mixtes
- ✓ **Changements graduels** : Quand stabilité importante
- ✓ **Environnements continus** : Actor-Critic fonctionne sans épisodes

**Limitations** :
- ✗ Convergence locale seulement (pas global optimum)
- ✗ Sample inefficiency (surtout REINFORCE)
- ✗ Sensibilité aux hyperparamètres (learning rates)
- ✗ Peut nécessiter beaucoup de tuning

**vs Value-Based Methods** :
- Policy Gradients : Mieux pour actions continues, policies stochastiques
- Value-Based (DQN) : Mieux pour actions discrètes, sample efficiency
```

```{admonition} Formules Clés
:class: seealso

**REINFORCE** :
$$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(A_t|S_t) \cdot G_t$$

**REINFORCE avec Baseline** :
$$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(A_t|S_t) \cdot (G_t - V(S_t))$$

**Actor-Critic (TD error)** :
$$\delta_t = R_{t+1} + \gamma V(S_{t+1};w) - V(S_t;w)$$
$$\theta \leftarrow \theta + \alpha_\theta \nabla_\theta \log \pi_\theta(A_t|S_t) \cdot \delta_t$$
$$w \leftarrow w + \alpha_w \delta_t \nabla_w V(S_t;w)$$

**Gaussian Policy (actions continues)** :
$$\pi(a|s;\theta) = \mathcal{N}(a|\mu_\theta(s), \sigma^2)$$
$$\nabla_\theta \log \pi(a|s;\theta) = \frac{a - \mu_\theta(s)}{\sigma^2} \nabla_\theta \mu_\theta(s)$$
```

```{admonition} Tableau Récapitulatif
:class: note

| Méthode | Update | Biais | Variance | Online | Continu |
|---------|--------|-------|----------|--------|---------|
| **REINFORCE** | Après épisode | Non | Très haute | Non | Non |
| **REINFORCE + Baseline** | Après épisode | Non | Haute | Non | Non |
| **Actor-Critic** | Chaque step | Oui | Faible | Oui | Oui |

**Règle générale** : Actor-Critic préféré dans la plupart des cas modernes.
```
