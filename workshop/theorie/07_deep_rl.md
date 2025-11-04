# 08 — Deep Reinforcement Learning (DQN, PPO, autres)

Ce chapitre donne un panorama des méthodes Deep RL populaires et des bonnes pratiques pour stabiliser l'entraînement.

## 1. Deep Q-Network (DQN)

Idée clé : approximer $Q(s,a)$ par un réseau neuronal $Q(s,a;\theta)$ et utiliser experience replay + target network.

Perte (DQN) :
$$L(\theta)=E_{(s,a,r,s')\sim D}\left[\big(r+\gamma \max_{a'} Q(s',a';\theta^-) - Q(s,a;\theta)\big)^2\right].$$

Pratiques :

- Replay buffer (taille, échantillonnage uniforme vs prioritized).
- Target network périodiquement mis à jour.
- Gradient clipping, batch normalization, reward clipping selon contexte.

## 2. Proximal Policy Optimization (PPO)

PPO est une méthode *on-policy* moderne qui stabilise le gradient de la policy via une borne sur la mise à jour (clipping).

La fonction objective tronquée (sur un pas) :
$$L^{CLIP}(\theta)=E\left[ \min\left( r(\theta) \hat{A}, \operatorname{clip}(r(\theta),1-ε,1+ε) \hat{A} \right) \right]$$
avec $r(\theta)=\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}$ et $\hat{A}$ l'avantage estimé.

## 3. Architecture & practical tips

- Choix du réseau (MLP, CNN pour observations images, RNN pour partial observability).
- Standardiser/normaliser les états et rewards.
- Sauvegarder et monitorer (tensorboard, weights & biases).

## 4. Petit schéma de pipeline

```text
Choose algorithm (DQN / DDPG / PPO / A3C)
Design network architecture
Set up replay / data pipeline (if needed)
Train with monitoring and checkpoints
Analyse courbes: return moyen, variance, fréquence d'exploration
```