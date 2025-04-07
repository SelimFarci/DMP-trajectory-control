import numpy as np
import matplotlib.pyplot as plt

class DMP(object):
    def __init__(self, pastor_mod=False):
        self.pastor_mod = pastor_mod
        # Paramètres du système de transformation
        self.alpha = 25.0             # = D = 20.0
        self.beta = self.alpha / 4.0  # = K / D = 100.0 / 20.0 = 5.0
        # Paramètres du système canonique
        self.alpha_t = self.alpha / 3.0
        # Paramètres pour l'évitement d'obstacles
        self.gamma_o = 1000.0
        self.beta_o = 20.0 / np.pi

    def spring_damper(self, x0, g, tau, dt, alpha, beta, observe_for):
        X = [x0]
        xd = 0.0
        xdd = 0.0
        t = 0.0
        while t < observe_for:
            X.append(X[-1] + xd * dt)
            x = X[-1]
            xd += xdd * dt
            xdd = alpha / (tau ** 2) * (beta * (g - x) - tau * xd)
            t += dt
        return X
    def forcing_term(self, x0, g, tau, w, s, X, scale=False):
        """The forcing term shapes the movement based on the weights."""
        n_features = w.shape[1]
        f = np.dot(w, self._features(tau, n_features, s))
        if scale:
            f *= g - x0

        if X.ndim == 3:
            F = np.empty_like(X)
            F[:, :] = f
            return F
        else:
            return f
    def _features(self, tau, n_features, s):
        if n_features == 0:
            return np.array([])
        elif n_features == 1:
            return np.array([1.0])
        c = self.phase(n_features)
        h = np.diff(c)
        h = np.hstack((h, [h[-1]]))
        phi = np.exp(-h * (s - c) ** 2)
        return s * phi / phi.sum()
    

    def trajectory(self, w, x0, g, tau, dt, o=None, shape=True, avoidance=False):
        x = x0.copy()
        xd = np.zeros_like(x, dtype=np.float64)
        xdd = np.zeros_like(x, dtype=np.float64)
        X = [x0.copy()]
        Xd = [xd.copy()]
        Xdd = [xdd.copy()]

        internal_dt = min(0.001, dt)
        n_internal_steps = int(tau / internal_dt)
        steps_between_measurement = int(dt / internal_dt)
        t = 0.5 * internal_dt
        ti = 0
        S = self.phase(n_internal_steps + 1)

        while t < tau:
            t += internal_dt
            ti += 1
            s = S[ti]

            x += internal_dt * xd
            xd += internal_dt * xdd

            sd = self.spring_damper(x0, g, tau, internal_dt, self.alpha, self.beta, tau)
            f = self.forcing_term(x0, g, tau, w, s, x) if shape else 0.0
            C = self.obstacle(o, x, xd) if avoidance else 0.0
            xdd = sd + f + C

            if ti % steps_between_measurement == 0:
                X.append(x.copy())
                Xd.append(xd.copy())
                Xdd.append(xdd.copy())

        return np.array(X), np.array(Xd), np.array(Xdd)

    def potential_field(self, dmp, t_max, Td, w, x0, g, tau, dt, o, x_range, y_range, n_tics):
        # Simulation de champ de potentiel
        # Calcul des forces (champ de potentiel, forçage, etc.)
        # Code à définir selon votre logique spécifique.
        pass

    def potential_trajectory(self, dmp, t_max, dt, shape, avoidance):
        T, Td, _ = dmp.trajectory(w, x0, g, tau, dt, o, shape, avoidance)
        X, Y, sd, f, C, acc = self.potential_field(
            dmp, t_max, Td[t_max - 1],
            w, x0, g, tau, dt, o, x_range, y_range, n_tics)
        if not avoidance:
            acc -= C
        return T[:t_max], X, Y, sd, f, C, acc

    def phase(self, n_steps, t=None):
        phases = np.exp(-self.alpha_t * np.linspace(0, 1, n_steps))
        if t is None:
            return phases
        else:
            return phases[t]

    

    def obstacle(self, o, X, Xd):
        if X.ndim == 1:
            X = X[np.newaxis, np.newaxis, :]
        if Xd.ndim == 1:
            Xd = Xd[np.newaxis, np.newaxis, :]

        C = np.zeros_like(X)
        R = np.array([[np.cos(np.pi / 2.0), -np.sin(np.pi / 2.0)],
                      [np.sin(np.pi / 2.0), np.cos(np.pi / 2.0)]])
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                obstacle_diff = o - X[i, j]
                theta = (np.arccos(obstacle_diff.dot(Xd[i, j]) /
                                   (np.linalg.norm(obstacle_diff) *
                                    np.linalg.norm(Xd[i, j]) + 1e-10)))
                C[i, j] = (self.gamma_o * R.dot(Xd[i, j]) * theta *
                           np.exp(-self.beta_o * theta))

        return np.squeeze(C)

# Test et Affichage des résultats
if __name__ == "__main__":
    x0, g = np.array([0.0, 0.0]), np.array([1.0, 1.0])
    tau = 1.0
    observe_for = 2.0 * tau
    dt = 0.01
    w = np.array([[-50.0, 100.0, 300.0], [-200.0, -200.0, -200.0]])  # Exemples de poids
    o = np.array([0.5, 0.5])  # Position de l'obstacle
    x_range = (-0.2, 1.2)
    y_range = (-0.2, 1.2)
    n_tics = 10

    dmp = DMP()

    # Paramètres pour les différentes valeurs de t_max
    fig = plt.figure(figsize=(10, 10))
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.01, hspace=0.01)

    for i, t in enumerate([1, 5, 10, 15, 20, 25, 30, 50, 99]):
        ax = plt.subplot(3, 3, 1 + i, aspect="equal")

        T, X, Y, sd, f, C, acc = dmp.potential_trajectory(
            dmp, t, dt, shape=True, avoidance=True)

        plt.plot(T[:, 0], T[:, 1], lw=5, color="black")

        quiver_scale = np.abs(acc).max() * n_tics
        plt.quiver(X, Y, sd[:, :, 0], sd[:, :, 1], scale=quiver_scale, color="g")
        plt.quiver(X, Y, f[:, :, 0], f[:, :, 1], scale=quiver_scale, color="r")
        plt.quiver(X, Y, C[:, :, 0], C[:, :, 1], scale=quiver_scale, color="y")
        plt.quiver(X, Y, acc[:, :, 0], acc[:, :, 1], scale=quiver_scale, color="black")

        plt.plot(x0[0], x0[1], "o", color="black", markersize=12)
        plt.plot(g[0], g[1], "x", color="green", markersize=12)
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
