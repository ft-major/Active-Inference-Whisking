import numpy as np
import matplotlib.pyplot as plt
from CPG_GM import GM
from CPG_GP import GP

# %%

if __name__ == "__main__":
    dt = 1/600/2
    n_steps = 100*6*4*2
    gp = GP(dt=dt, omega2_GP=(6*2*np.pi)**2, alpha=[1.])
    gm = GM(dt=dt, eta=0.01, eta_d=1., eta_nu=0.001, eta_a=0.001, nu=gp.a)#, x=gp.x[0], eta=0.001, eta_d=1., eta_a=0.06, eta_nu=0.01, omega2_GM=0.5, nu=1)

    cpg = []
    obj_position = []

    gp_x = []
    gp_a = []

    sensory_state = []

    gm_mu = []
    gm_dmu = []
    gm_nu = []
    a = 0.
    gp.update(0)
    for step in np.arange(n_steps):
        #print(gp.x)
        sensory_state.append([gp.s_p[0], gp.s_t[0]])

        a = gm.update(touch_sensory_states=gp.s_t, proprioceptive_sensory_states=gp.s_p, x=gp.cpg[0])
        gp.update(0)
        cpg.append( gp.cpg[0] )
        obj_position.append( gp.object_position[0])

        gp_x.append( [gp.x[0]])#, gp.x[1]] )
        gp_a.append( [gp.a[0]])#, gp.a[1]] )

        gm_mu.append( [gm.mu[0]])#, gm.mu[1]] )

        gm_dmu.append( [gm.dmu[0]])#, gm.dmu[1]] )
        gm_nu.append( [gm.nu[0]])#, gm.nu[1]] )
        #data_GP.append([gp.x, gp.a, gp.object_position])
        #data_GM.append([gm.mu, gm.nu, 0, 0, gm.PE_mu,
        #                gm.PE_s[0], gm.PE_s[1], gm.dmu, 0, 0, 0, 0, gm.g_touch(x=gm.mu, v=gm.dmu) ])
    #print(gpx[0])
    sensory_state = np.vstack(sensory_state)
    gp_x = np.vstack(gp_x)
    gp_a = np.vstack(gp_a)

    gm_mu = np.vstack(gm_mu)
    gm_dmu = np.vstack(gm_dmu)
    gm_nu = np.vstack(gm_nu)
    #data_GP = np.array(data_GP)
    #data_GM = np.vstack(data_GM)

    #%%
    time = np.arange(0, n_steps*dt, dt)
    plt.figure(figsize=(20, 10))
    plt.subplot(211)
    plt.plot(time, gp_x[:,0], label=r"$x_0$")
    #plt.plot(time, sensory_state[:,0], label=r"$s_p$")
    #plt.plot(time, sensory_state[:,1], label=r"$s_t$")
    plt.plot(time, gp_a[:,0], label=r"$\alpha_0$")
    #plt.plot(time, obj_position, c="black", label=r"object")
    #plt.plot(time, cpg, label=r"cpg")
    #plt.ylim(-1.5,1.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.subplot(212)
    plt.plot(time, gm_mu[:,0], c="green", label=r"$\mu_0$")
    plt.plot(time, gm_nu[:,0], c="#66aa66", label=r"$\nu_0$")
    plt.plot(time, obj_position, c="black", label=r"object")
    #plt.plot(time, gm_dmu[:,0], label=r"$\mu_0'$")
    #plt.plot(time, cpg, label=r"$cpg$")
    #plt.plot(time, gm_nu[:,0], label=r"$\nu_0$")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.ylim(-1.5,1.5)
    plt.show()

    #%%
    """
    plt.figure(figsize=(20, 10))
    plt.subplot(211)
    plt.plot(time, gp_x[:,0], label=r"$x_0$")
    #plt.plot(time, cpg, label=r"cpg")
    plt.plot(time, gp_a[:,0], label=r"$\alpha_0$")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.subplot(212)
    plt.plot(time, gp_x[:,1], label=r"$x_1$")
    #plt.plot(time, cpg, label=r"$cpg$")
    plt.plot(time, gp_a[:,1], label=r"$\alpha_1$")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()
    """
