import numpy as np
import matplotlib.pyplot as plt

def smooth_biphasic_pulse(t):

    tb=500 # ms (beginning time)
    ts=507  # ms (switching time)
    te=510 # ms (end time)
    # total pulse duration: 10 ms

    tau=0.1
    k_t=1/tau

    step_b=0.5*(1+np.tanh(k_t*(t-tb)))
    step_s=0.5*(1+np.tanh(k_t*(t-ts)))
    step_e=0.5*(1+np.tanh(k_t*(t-te)))

    E0=100 # cm^(-1), multiply by 0.1 V, to get V/cm.

    Et=E0*(step_b-step_s)-E0*(step_s-step_e)
    return Et

if __name__=="__main__":
    tt=np.linspace(0,700,701)
    Et=smooth_biphasic_pulse(tt)
    plt.plot(tt,Et)
    plt.show()
    
