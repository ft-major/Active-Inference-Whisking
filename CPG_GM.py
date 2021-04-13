import numpy as np

def sech(x):
    return 1/np.cosh(x)

def tanh(x):
    return np.tanh(x)

#%%
#Generative model class
class GM:

    def __init__(self, dt, eta=0.1, eta_d=0.1, eta_a=1., eta_nu=1., nu=[1., 1.], Sigma_mu=[0.01, 0.01], Sigma_s_p=[0.01, 0.01], Sigma_s_t=[0.07, 0.07]):

        # Parameter that regulates whiskers amplitude oscillation
        self.nu = np.array(nu)
        # Vector \vec{\mu} initialized with the GP initial conditions
        self.mu = np.zeros(len(nu))
        # Vector \dot{\vec{\mu}}
        self.dmu = np.zeros(len(nu))
        # Variances (inverse of precisions) of sensory proprioceptive inputs
        self.Sigma_s_p = np.ones(len(nu))*0.05
        # Variances (inverse of precisions) of sensory touch inputs
        self.Sigma_s_t = np.ones(len(nu))*0.001 #np.array(Sigma_s_t)
        # Internal variables precisions
        self.Sigma_mu = np.ones(len(nu))*0.01 #np.array(Sigma_mu)

        # Action variable (in this case the action is intended as the increment of the variable that the agent is allowed to modified)
        self.da = 0.
        # Size of a simulation step
        self.dt = dt
        # Gradient descent weights
        self.eta = np.array([eta, eta_d, eta_a, eta_nu])

    # Touch function
    def g_touch(self, x, v, prec=10):
        return sech(prec*v)*(0.5*tanh(prec*x)+0.5)

    # Derivative of the touch function with respect to v
    def dg_dv(self, x, v, prec=10):
        return -prec*sech(prec*v)*tanh(prec*v)*(0.5 * tanh(prec*x) + 0.5)

    # Derivative of the touch function with respect to x
    def dg_dx(self, x, v, prec=10):
        return sech(prec*v)*0.5*prec*(sech(prec*x))**2


    # Function that implement the update of internal variables.
    def update(self, touch_sensory_states, proprioceptive_sensory_states, x):
        # touch_sensory_states  and proprioceptive_sensory_states arguments come from GP (both arrays have dimension equal to the number of whiskers)
        # Returns action increment

        #x = np.array(x)    # ?needed?
        self.s_p = proprioceptive_sensory_states
        self.s_t = touch_sensory_states
        eta, eta_d, eta_a, eta_nu = (self.eta[0], self.eta[1], self.eta[2], self.eta[3])

        self.PE_mu = self.dmu - (self.nu*x - self.mu)
        self.PE_s_p = self.s_p-self.dmu
        self.PE_s_t = self.s_t-self.g_touch(x=self.mu, v=self.dmu)

        self.dF_dmu = self.PE_mu/self.Sigma_mu \
                    - self.dg_dx(x=self.mu, v=self.dmu)*self.PE_s_t/self.Sigma_s_t

        self.dF_d_dmu = self.PE_mu/self.Sigma_mu \
                        - self.PE_s_p/self.Sigma_s_p \
                        - self.dg_dv(x=self.mu, v=self.dmu)*self.PE_s_t/self.Sigma_s_t

        # Action update
        # case with dg/da = 1
        self.da = -self.dt*eta_a*( x*self.PE_s_p/self.Sigma_s_p + self.PE_s_t/self.Sigma_s_t )
        # case with real dg/da
        #self.da = -self.dt*eta_a*x*self.dg_dv(x=self.mu, v=self.dmu)*self.PE_s[1]/self.Sigma_s[1]

        # Learning internal parameter nu
        self.nu += -self.dt*eta_nu*(-x*self.PE_mu/self.Sigma_mu)  # - 0*x*self.dg_dv(x=self.mu, v=self.dmu)*self.PE_s[1]/self.Sigma_s[1])

        self.mu += self.dt*(self.dmu - eta*self.dF_dmu)
        self.dmu += -self.dt*eta_d*self.dF_d_dmu

        # Efference copy
        #self.nu += self.da

        return self.da
