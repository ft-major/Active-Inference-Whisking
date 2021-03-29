#%%
# Modules
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.RandomState(42)

def sech(x):
    return 1/np.cosh(x)


def tanh(x):
    return np.tanh(x)

#%%
# Generative process class
class GenProc:
    def __init__(self, dt, omega2_GP=0.5):

        # Generative process parameters

        # Generative process s variance
        self.Sigma_s = np.array([0.1, 0.1])
        # Two dimensional array storing angle and angle velocity initialized with his initial conditions
        self.x = np.array([0., 1.])
        # Harmonic oscillator angular frequency square (omega^2)
        self.omega2 = omega2_GP
        # Costant that quantify the amount of energy (friction?) that the agent can insert in the system
        self.u = 1
        # Array storing respectively proprioceptive sensory input (initialized with the real value x_0) and touch sensory input
        self.s = np.array([0, 1., 0.])
        # Size of a simulation step
        self.dt = dt
        # Time variable
        self.t = 0
        # Platform position (when is present) with respect to x_0 variable
        self.platform_position = 0.5
        # Time interval in which the platform appears
        self.platform_interval = [20, 80]

    # Step of generative process dynamic
    def update(self, action):
        # Increment of time variable
        self.t += self.dt
        # GP dynamics implementation
        self.x[0] += self.x[1]*dt
        self.x[1] += -self.omega2*self.x[0]*dt + self.u*np.tanh(action)*dt*self.x[1]

    # Funciton that create agent's sensory input (two dimensional array)
    def genS(self):
        # Platform Action
        if self.t > self.platform_interval[0] and self.t < self.platform_interval[1]:
            if self.x[0] > self.platform_position:
                self.s[2] = 1.
                #self.s[0] = self.platform_position
            else:
                #self.s[0] = self.x[0]
                self.s[2] = 0.
        else:
            #self.s[0] = self.x[0]
            self.s[2] = 0.
        self.s[0] = self.x[0] + self.Sigma_s[0]*rng.randn()
        self.s[1] = self.x[1] + self.Sigma_s[1]*rng.randn()
        return self.s

    # Function that generates the array to graph the platform
    def platform_for_graph(self):
        plat = []
        for t in np.arange(0., self.t, self.dt):
            if t in self.platform_interval:
                plat.append([t, self.platform_position])
        return np.vstack(plat)

#%%
# Generative model class
class GenMod:
    def __init__(self, dt, omega2_GM=0.5, k_mu=0.001, k_dmu=0.1, k_a=0.1):

        # Generative process parameters

        # Harmonic oscillator angular frequency (omega^2). We're assuming is equal to the one of the GP
        self.omega2 = omega2_GM
        # Vector \vec{\mu}={\mu_0, \mu_1} initialized with the GP initial conditions
        self.mu = np.array([0., 1.])
        # Vector \dot{\vec{\mu}}={\dot{\mu_0}, \dot{\mu_1}} inizialized with the right ones
        self.dmu = np.array([0., -self.omega2])
        # Internal variables precisions
        self.Sigma_mu = np.array([0.01, 0.01])
        # Variances (inverse of precisions) of sensory input (the first one proprioceptive and the second one touch)
        self.Sigma_s = np.array([0.01, 1000.01, 0.01])
        # Action variable
        self.a = 0
        # Costant that quantify the amount of energy (friction?) that the agent can insert in the system
        self.u = 1
        # Gradient descent inference parameters
        self.k_mu = k_mu
        self.k_dmu = k_dmu
        # Gradient descent action parameter
        self.k_a = k_a
        self.dt =dt

    # Touch function
    def g_touch(self, x, v, prec=50):
        return sech(prec*v)*(0.5*tanh(prec*x)+0.5)

    # Derivative of the touch function with respect to \mu_0
    def dg_dv(self, x, v, prec=50):
        return -prec*sech(prec*v)*tanh(prec*v)*(0.5 * tanh(prec*x) + 0.5)

    # Derivative of the touch function with respect to \mu_2
    def dg_dx(self, x, v, prec=50):
        return sech(prec*v)*prec*0.5*(sech(prec*x))**2

    # Function that implement the update of internal variables.
    def update(self, sensory_states):
        # sensory_states argument (two dimensional array) come from GP and store proprioceptive
        # and somatosensory perception
        # Returns action increment

        self.s = sensory_states

        self.PE_mu = np.array([
            self.dmu[0]-self.mu[1],
            self.dmu[1]+self.omega2*self.mu[0]
        ])
        self.PE_s = np.array([
            self.s[0]-self.mu[0],
            self.s[1]-self.mu[1],
            self.s[2]-self.g_touch(x=self.mu[0], v=self.mu[1])  # v=self.dmu[0]?
        ])

        self.dF_dmu = np.array([
            self.omega2*self.PE_mu[1]/self.Sigma_mu[1] - self.PE_s[0]/self.Sigma_s[0] \
                - self.dg_dx(x=self.mu[0], v=self.mu[1])*self.PE_s[2]/self.Sigma_s[2],
            -self.PE_mu[0]/self.Sigma_mu[0] - self.PE_s[1]/self.Sigma_s[1] \
                - self.dg_dv(x=self.mu[0], v=self.mu[1])*self.PE_s[2]/self.Sigma_s[2]
        ])

        self.dF_d_dmu = np.array([
            self.PE_mu[0]/self.Sigma_mu[0],
            self.PE_mu[1]/self.Sigma_mu[1]
        ])


        # Action update
        # case with dg/da = 1
        self.a = -self.dt*self.k_a*( self.PE_s[1]/self.Sigma_s[1] + self.PE_s[2]/self.Sigma_s[2] )
        # case with real dg/da
        #self.da = -self.dt*self.k_a*x*self.dg_dv(x=self.mu, v=self.dmu)*self.PE_s[1]/self.Sigma_s[1]

        # Internal variables update
        self.mu += self.dt*(self.dmu - self.k_mu*self.dF_dmu)
        self.dmu += -self.dt*self.k_dmu*self.dF_d_dmu

        return self.a

#%%
if __name__ == "__main__":
    dt=0.005
    n_steps = 20000+5000
    gp = GenProc(dt=dt, omega2_GP=0.5)
    gm = GenMod(dt=dt, k_mu=0.01, k_dmu=0.1, k_a=3, omega2_GM=0.5)

    data_GP = []
    data_GM = []
    a = 0.
    for step in np.arange(n_steps):
        a = gm.update(gp.genS())
        gp.update(a)
        data_GP.append([gp.x[0], gp.u*np.tanh(a)*dt*gp.x[1], gp.s[0], gp.s[1], a])
        data_GM.append([gm.mu[0], 0, 0, 0, gm.PE_mu[0],
                        gm.PE_s[0], gm.PE_s[1], gm.dmu[0], 0, 0, 0, 0, gm.g_touch(x=gm.mu[0], v=gm.mu[1]) ])
    data_GP = np.vstack(data_GP)
    data_GM = np.vstack(data_GM)
    platform = gp.platform_for_graph()

    # %%
    plt.figure(figsize=(20, 10))
    plt.subplot(311)
    plt.plot(np.arange(0, n_steps*dt, dt), data_GP[:, 0], c="red", lw=2, ls="dashed", label=r"$x_2$")
    #plt.plot(np.arange(0, n_steps*dt, dt), data_GP[:, 4], c="blue", lw=2, ls="dashed", label=r"$x_0$")
    #plt.plot(np.arange(0, n_steps*dt, dt), data_GP[:, 1], c="#aa6666", lw=4, label=r"\alpha")
    plt.plot(platform[:,0], platform[:,1], c="black", lw=2, label="platform")
    #plt.ylim(bottom=-1, top=1.25)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.subplot(312)
    plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 0], c="green", lw=2, ls="dashed", label=r"$\mu_2$")
    #plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 1], c="#66aa66", lw=3, label=r"\nu")
    #plt.plot(np.arange(0, n_steps*dt, dt), data_GM[:, 8], c="blue", lw=2, ls="dashed", label=r"$\mu_0$")
    #plt.ylim(bottom=-1, top=1.25)
    plt.plot(platform[:,0], platform[:,1], c="black", lw=2, label="platform")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.subplot(313)
    plt.plot(np.arange(0, n_steps*dt, dt), data_GP[:, -1], c="blue", lw=2, label="action")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    #plt.savefig("simulation_results2")
    plt.show()

    #%%
    plt.figure(figsize=(20, 10))
    plt.plot(np.arange(0, n_steps*dt, dt), data_GP[:, -1], c="blue", lw=2, label="action")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()
