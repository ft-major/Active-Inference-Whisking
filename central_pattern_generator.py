import numpy as np
import matplotlib.pyplot as plt
from CPG_GM import GM
rng = np.random.RandomState(42)

def sech(x):
    return 1/np.cosh(x)


def tanh(x):
    return np.tanh(x)

# %%

# Generative Process Class
class GP:

    def __init__(self, dt, omega2_GP=0.5, alpha=[1,1]):

        # Harmonic oscillator angular frequency (both x_0 and x_2)
        self.omega2 = omega2_GP
        # Variable representing the central pattern generator
        self.cpg = np.array([0., 1.])
        # Parameter that regulates whiskers amplitude oscillation
        self.a = np.array(alpha)
        # Whiskers base angles
        self.x = np.array([0, 0])
        # Array storing proprioceptive sensory inputs (whiskers angular velocity)
        self.s_p = np.array([0. ,0.])
        # Array storing touch sensory inputs
        self.s_t = np.array([0., 0.])
        # Variance of the Gaussian noise that gives proprioceptive sensory inputs
        self.Sigma_s_p = np.array([0.1, 0.1])
        # Variance of the Gaussian noise that gives touch sensort inputs
        self.Sigma_s_t = np.array([0.1, 0.1])
        # Size of a simulation step
        self.dt = dt
        # Time variable
        self.t = 0
        # Object position (when is present) with respect to Whiskers angles
        self.object_position = [0.5, 5]
        # Time interval in which the object appears
        self.platform_interval = [15, 90]

    # Function that regulates object position
    def obj_pos(self, t):
        obj_interval = [15, 90]
        if t > obj_interval[0] and t < obj_interval[1]:
            return np.array([0.5, 2])
        else:
            return np.array([10, 10])


    # Function that return if a whisker has touched
    def touch(self, x, object):
        if x>=object:
            return 1
        else:
            return 0
    """
    def touch_cont(self, x, platform_position, prec=100):
        return 0.5 * (tanh(prec*(x-platform_position)) + 1)"""

    # Function that implement dynamics of the process.
    def update(self, action):
        # Action argument (double) is the variable that comes from the GM that modifies alpha
        # variable affecting the amplitude of the oscillation.

        # Increment of time variable
        self.t += self.dt
        # Increment of alpha variable (that changes the amplitude) given by agent's action
        self.a += action
        # GP dynamics implementation
        self.cpg[0] += self.dt*(self.cpg[1])
        self.cpg[1] += self.dt*(-self.omega2*self.cpg[0])
        self.x += self.dt*(self.a*self.cpg[0] - self.x)

        # object Action on touch sensory inputs
        for i in arange(len(self.x)):
            self.s_t[i] = self.touch(self.x[i], self.obj_pos(self.t)[i]) #+ self.Sigma_s_t[i]*rng.randn()
            self.x[i] = min(self.x[i], self.obj_pos)
            if self.s_t[i] == 1:
                self.s_p[i] = 0
            else:
                self.s_p[i] = self.a[i]*self.cpg[0] - self.x[i]
            self.s_p[i] += self.Sigma_s_p[i]*rng.randn()

        # Proprioceptive sensor inputs
        self.s_p = self.x + self.Sigma_s_p*rng.randn()

# %%

if __name__ == "__main__":
    dt = 0.005
    n_steps = 20000+5000
    gp = GP(dt=dt, omega2_GP=0.5, alpha=1)
    gm = GM(dt=dt, x=gp.x[0], eta=0.001, eta_d=1., eta_a=0.06, eta_nu=0.01, omega2_GM=0.5, nu=1)

    data_GP = []
    data_GM = []
    a = 0.
    for step in np.arange(n_steps):
        a = gm.update(gp.s, gp.x[0])
        gp.update(a)
        data_GP.append([gp.x[2], gp.a, gp.s[0], gp.s[1], gp.x[0]])
        data_GM.append([gm.mu, gm.nu, 0, 0, gm.PE_mu,
                        gm.PE_s[0], gm.PE_s[1], gm.dmu, 0, 0, 0, 0, gm.g_touch(x=gm.mu, v=gm.dmu) ])
    data_GP = np.vstack(data_GP)
    data_GM = np.vstack(data_GM)
    platform = gp.platform_for_graph()
    # %%
    plt.figure(figsize=(20, 10))
    plt.subplot(211)
    plt.plot(np.arange(0, n_steps*dt, dt), data_GP[:, 0], c="red", lw=2, ls="dashed", label=r"$x_2$")
    #plt.plot(np.arange(0, n_steps*dt, dt), data_GP[:, 4], c="blue", lw=2, ls="dashed", label=r"$x_0$")
    plt.plot(np.arange(0, n_steps*dt, dt), data_GP[:, 1], c="#aa6666", lw=4, label=r"\alpha")
    plt.plot(platform[:,0], platform[:,1], c="black", lw=2, label="platform")
    plt.ylim(bottom=-1, top=1.40)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.subplot(212)
    plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 0],
    c="green", lw=2, ls="dashed", label=r"$\mu_2$")
    plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 1], c="#66aa66", lw=3, label=r"\nu")
    #plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 8], c="blue", lw=2, ls="dashed", label=r"$\mu_0$")
    plt.ylim(bottom=-1, top=1.40)
    plt.plot(platform[:,0], platform[:,1], c="black", lw=2, label="platform")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig("simulation_results1")
    plt.show()

    #%%

    plt.figure(figsize=(20, 10))
    plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 0], c="green", lw=2, ls="dashed", label=r"$\mu_2$")
    plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 7], c="red", lw=2, ls="dashed", label=r"d$\mu_2$")
    #plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 11], c="purple", lw=2, ls="dashed", label=r"d$\mu_2$")
    #plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 12], c="grey", lw=2, ls="dashed", label=r"d$\mu_2$")
    #plt.plot(np.arange(0, n_steps*dt, dt), gm.g_touch(x=data_GM[:, 0], v=data_GM[:, 7]), label=r"touch")
    #plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 1], c="#66aa66", lw=3, label=r"\nu")
    #plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 8], c="blue", lw=2, ls="dashed", label=r"$\mu_0$")
    #plt.plot(platform[:,0], platform[:,1], c="black", lw=2, label="platform")
    #plt.ylim(bottom=-1.8, top=4)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()


    # %%
    plt.figure(figsize=(20, 10))
    plt.subplot(211)
    plt.plot(np.arange(0, n_steps*dt, dt), data_GP[:, 0], c="red", lw=2, ls="dashed", label=r"$x_2$")
    plt.plot(np.arange(0, n_steps*dt, dt), data_GP[:, 4], c="blue", lw=2, ls="dashed", label=r"$x_0$")
    plt.plot(np.arange(0, n_steps*dt, dt), data_GP[:, 1], c="#aa6666", lw=4, label=r"\alpha")
    plt.plot(platform[:,0], platform[:,1], c="black", lw=2, label="platform")
    #plt.ylim(bottom=-1.8, top=4)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.subplot(212)
    plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 0],
             c="green", lw=2, ls="dashed", label=r"$\mu_2$")
    plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 1], c="#66aa66", lw=3, label=r"\nu")
    plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 8], c="blue", lw=2, ls="dashed", label=r"$\mu_0$")
    plt.plot(platform[:,0], platform[:,1], c="black", lw=2, label="platform")
    #plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 10]*gm.Sigma_s[1], c="orange", label=r"$\, \frac{ d\varepsilon_{s_1} }{ da }$")
    #plt.ylim(bottom=-1.8, top=4)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()

    #%%
    #plt.figure(figsize=(20, 10))
    #plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 9], label=r"$\frac{ 1 }{ \Sigma_{s_0} } \, \frac{ d\varepsilon_{s_0} }{ da }$")
    #plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 10], c = "orange", label=r"$\frac{ 1 }{ \Sigma_{s_1} } \, \frac{ d\varepsilon_{s_1} }{ da }$")
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    # %%
    plt.figure(figsize=(12, 8))
    plt.subplot(211)
    plt.plot(np.arange(0, n_steps*dt, dt), data_GP[:, 0], c="red", lw=1, ls="dashed", label=r"$x_2$")
    plt.plot(np.arange(0, n_steps*dt, dt), data_GP[:, 1], c="#aa6666", lw=3, label=r"\alpha")
    plt.plot(platform[:,0], platform[:,1], c="black", lw=0.5, label="platform")
    # plt.ylim(bottom=0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.subplot(212)
    plt.plot(np.arange(0, n_steps*dt, dt), data_GP[:, 2], label="Proprioceptive sensory input")
    plt.plot(np.arange(0, n_steps*dt, dt), data_GP[:, 3], label="Touch sensory input")
    #plt.plot(platform[:,0], platform[:,1], c="black", lw=0.5, label="platform")
    # plt.ylim(bottom=0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()

    # %%
    plt.figure(figsize=(12, 8))
    plt.subplot(311)
    plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 2], label=r"$PE_{\mu 0}$")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.subplot(312)
    plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 3], label=r"$PE_{\mu 1}$")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.subplot(313)
    plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 4], label=r"$PE_{\mu 2}$")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()
    # %%
    plt.figure(figsize=(12, 8))
    plt.subplot(211)
    plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 5], label=r"$PE_{s 0}$")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.subplot(212)
    plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 6], label=r"$PE_{s 1}$")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()
    #%%
