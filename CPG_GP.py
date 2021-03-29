import numpy as np
rng = np.random.RandomState(42)


# %%

# Generative Process Class
class GP:

    def __init__(self, dt, omega2_GP=0.5, alpha=[1.,1.]):

        # Harmonic oscillator angular frequency (both x_0 and x_2)
        self.omega2 = omega2_GP
        # Variable representing the central pattern generator
        self.cpg = np.array([0., 1.])
        # Parameter that regulates whiskers amplitude oscillation
        self.a = np.array(alpha)
        # Whiskers base angles
        self.x = np.array([0., 0.])
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
        self.t = 0.
        # Object position (when is present) with respect to Whiskers angles
        self.object_position = [0.5, 5.]
        # Time interval in which the object appears
        self.platform_interval = [15, 90]

    # Function that regulates object position
    def obj_pos(self, t):
        obj_interval = [15, 90]
        if t > obj_interval[0] and t < obj_interval[1]:
            return np.array([0.5, 2.])
        else:
            return np.array([10, 10])


    # Function that return if a whisker has touched
    def touch(self, x, object):
        if x>=object:
            return 1.
        else:
            return 0.
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
        for i in range(len(self.x)):
            self.s_t[i] = self.touch(self.x[i], self.obj_pos(self.t)[i]) #+ self.Sigma_s_t[i]*rng.randn()
            self.x[i] = min(self.x[i], self.obj_pos(self.t)[i])
            if self.s_t[i] == 1:
                self.s_p[i] = 0
            else:
                self.s_p[i] = self.a[i]*self.cpg[0] - self.x[i]
            self.s_p[i] += self.Sigma_s_p[i]*rng.randn()

        # Proprioceptive sensor inputs
        self.s_p = self.x + self.Sigma_s_p*rng.randn()
