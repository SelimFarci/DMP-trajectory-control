#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 08:38:05 2024

@author: loicmenoret & Selim Farci
"""

import numpy as np
import matplotlib.pyplot as plt

class DMP3D:
    def __init__(self, n_bases=10, alpha=25.0, beta=None, tau=1.0):
        """
        Initialisation des paramètres pour les DMP 3D.
        n_bases : Nombre de fonctions de base
        alpha : Gain de convergence vers la cible
        beta : Coefficient de stabilisation (par défaut alpha/4)
        tau : Facteur de temps (ajuste la vitesse de mouvement)
        """
        self.n_bases = n_bases
        self.alpha = alpha
        self.beta = beta if beta is not None else alpha / 4.0
        self.tau = tau
        
        # Initialisation des poids pour chaque dimension (x, y, z)
        self.weights = {
            'x': np.zeros(n_bases),
            'y': np.zeros(n_bases),
            'z': np.zeros(n_bases)
        }

        # Fonctions de base partagées
        self.centers = np.linspace(0, 1, n_bases)
        self.widths = 1.0 / (np.diff(self.centers)**2)
        self.widths = np.append(self.widths, self.widths[-1])

    def basis_functions(self, phase):
        """
        Calcul des fonctions de base pour une phase donnée.
        """
        return np.exp(-self.widths * (phase - self.centers)**2)

    def phase_variable(self, t):
        """
        Phase variable pour contrôler l'évolution dans le temps (linéaire ici).
        """
        return np.exp(-self.alpha * t / self.tau)

    def learn(self, trajectory_3d, dt=0.01):
        """
        Apprentissage des poids à partir d'une trajectoire 3D donnée.
        trajectory_3d : Dictionnaire contenant les trajectoires x, y, z
        """
        time = np.linspace(0, 1, len(trajectory_3d['x']))
        phase = self.phase_variable(time)

        for dim in ['x', 'y', 'z']:
            trajectory = trajectory_3d[dim]
            velocities = np.gradient(trajectory, dt)
            accelerations = np.gradient(velocities, dt)

            # Calcul du terme de forçage désiré
            forcing_term = (self.tau**2 * accelerations - 
                            self.alpha * (self.beta * (trajectory[-1] - trajectory) - self.tau * velocities))

            # Résolution pour obtenir les poids
            psi = np.array([self.basis_functions(p) for p in phase])
            for i in range(self.n_bases):
                self.weights[dim][i] = np.sum(forcing_term * psi[:, i]) / (np.sum(psi[:, i]) + 1e-10)

    def generate(self, x0_3d, goal_3d, dt=0.01, duration=1.0):
        """
        Générer une trajectoire 3D en fonction des poids appris.
        """
        time = np.linspace(0, duration, int(duration / dt))
        phase = self.phase_variable(time)

        trajectory_3d = {'x': [], 'y': [], 'z': []}
        velocities_3d = {'x': [], 'y': [], 'z': []}
        accelerations_3d = {'x': [], 'y': [], 'z': []}

        state = {'x': x0_3d[0], 'y': x0_3d[1], 'z': x0_3d[2]}
        velocity = {'x': 0.0, 'y': 0.0, 'z': 0.0}

        for p in phase:
            psi = self.basis_functions(p)
            
            for dim in ['x', 'y', 'z']:
                forcing_term = np.dot(self.weights[dim], psi) / (np.sum(psi) + 1e-10)

                # Équations différentielles
                accel = (self.alpha * (self.beta * (goal_3d[dim] - state[dim]) - velocity[dim]) + forcing_term) / self.tau**2
                velocity[dim] += accel * dt
                state[dim] += velocity[dim] * dt

                # Stocker les valeurs
                trajectory_3d[dim].append(state[dim])
                velocities_3d[dim].append(velocity[dim])
                accelerations_3d[dim].append(accel)

        return trajectory_3d, velocities_3d, accelerations_3d


# Exemple d'utilisation
if __name__ == "__main__":
    # Trajectoire de consigne (3D)
    time = np.linspace(0, 1, 100)
    train_trajectory_3d = {
        'x': np.sin(2 * np.pi * time),
        'y': np.cos(2 * np.pi * time),
        'z': 0.5 * np.sin(4 * np.pi * time)
    }

    # Initialisation du modèle DMP3D
    dmp = DMP3D(n_bases=20, alpha=48.0) # Point de l'espace
    
    # Apprentissage
    dmp.learn(train_trajectory_3d)

    # Génération d'une nouvelle trajectoire
    x0_3d = [0, 1, 0]  # Point de départ initial (x, y, z)
    goal_3d = {'x': 1, 'y': 0, 'z': 0.5}  # Cible en 3D
    trajectory_3d, velocities_3d, accelerations_3d = dmp.generate(x0_3d, goal_3d, duration=1.0)

    # Tracé des résultats
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(train_trajectory_3d['x'], train_trajectory_3d['y'], train_trajectory_3d['z'], label="Trajectoire d'apprentissage", linestyle="--")
    ax.plot(trajectory_3d['x'], trajectory_3d['y'], trajectory_3d['z'], label="Trajectoire générée")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title("Dynamic Motion Primitives - 3D")
    plt.show()
