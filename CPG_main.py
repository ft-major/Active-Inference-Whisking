import numpy as np
import matplotlib.pyplot as plt
from CPG_GM import GM
from CPG_GP import GP

# %%

if __name__ == "__main__":
    dt = 0.005
    n_steps = 10 #20000+5000
    gp = GP(dt=dt)
    gm = GM(dt=dt)#, x=gp.x[0], eta=0.001, eta_d=1., eta_a=0.06, eta_nu=0.01, omega2_GM=0.5, nu=1)

    data_GP = []
    data_GM = []
    a = 0.
    for step in np.arange(n_steps):
        #print(gp.x)
        a = gm.update(touch_sensory_states=gp.s_t, proprioceptive_sensory_states=gp.s_p, x=gp.cpg[0])
        gp.update(a)
        data_GP.append([gp.x, gp.a, gp.object_position])
        #data_GM.append([gm.mu, gm.nu, 0, 0, gm.PE_mu,
        #                gm.PE_s[0], gm.PE_s[1], gm.dmu, 0, 0, 0, 0, gm.g_touch(x=gm.mu, v=gm.dmu) ])
    #data_GP = np.vstack(data_GP)
    #data_GM = np.vstack(data_GM)
    # %%
    print(data_GP)
    plt.figure(figsize=(20, 10))
    plt.subplot(211)
    plt.plot(np.arange(0, n_steps*dt, dt), data_GP[:, 0][0], c="red", lw=2, ls="dashed", label=r"$x_2$")
    #plt.plot(np.arange(0, n_steps*dt, dt), data_GP[:, 4], c="blue", lw=2, ls="dashed", label=r"$x_0$")
    plt.plot(np.arange(0, n_steps*dt, dt), data_GP[:, 1][0], c="#aa6666", lw=4, label=r"\alpha")
    plt.plot(np.arange(0, n_steps*dt, dt), data_GP[:, 2][0], c="#aa6666", lw=4, label=r"platform")
    #plt.ylim(bottom=-1, top=1.40)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.subplot(212)
    plt.plot(np.arange(0, n_steps*dt, dt), data_GP[:, 0][1], c="red", lw=2, ls="dashed", label=r"$x_2$")
    #plt.plot(np.arange(0, n_steps*dt, dt), data_GP[:, 4], c="blue", lw=2, ls="dashed", label=r"$x_0$")
    plt.plot(np.arange(0, n_steps*dt, dt), data_GP[:, 1][1], c="#aa6666", lw=4, label=r"\alpha")
    plt.plot(np.arange(0, n_steps*dt, dt), data_GP[:, 2][1], c="#aa6666", lw=4, label=r"platform")
#%%
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
