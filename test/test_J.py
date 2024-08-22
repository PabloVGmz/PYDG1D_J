import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pytest
import time


from maxwell.driver import *
from maxwell.dg.mesh1d import *
from maxwell.dg.dg1d import *
from maxwell.fd.fd1d import *


from nodepy import runge_kutta_method as rk

def test_pec_dielectrico_upwind_J():
#planificar el test a mano
#numero elementos
    #calcular amplitudes coeficientes
    epsilon_1=1
    epsilon_2=2
    mu_1=1
    mu_2=1
    z_1=np.sqrt(mu_1/epsilon_1)
    z_2=np.sqrt(mu_2/epsilon_2)
    v = np.zeros(100)
    rho=1
    epsilons = epsilon_1*np.ones(100)  
    epsilons[50:99]=epsilon_2

    v[0:49]=(epsilons[0:49]*mu_1)**-2
    v[50:99]=(epsilons[50:99]*mu_2)**-2
    
    sp = DG1D(
        n_order=3,
        mesh=Mesh1D(-5.0, 5.0, 100, boundary_label="PEC"),
        epsilon=epsilons
    )
    driver = MaxwellDriver(sp)


    #Coeficientes T y R
    T_E=2*z_2/(z_1+z_2)
    R_E=(z_2-z_1)/(z_2+z_1)

    final_time = 6
    s0 = 0.50
    initialFieldE = np.exp(-(sp.x+2)**2/(2*s0**2))
    initialFieldH = initialFieldE
    initialFieldJ = v*rho
    # initialFieldJ = np.zeros(initialFieldE.shape)

    driver['E'][:] = initialFieldE[:]
    driver['H'][:] = initialFieldH[:]
    driver['J'][:] = initialFieldJ[:]
        
    driver.run_until(final_time)

    max_electric_field_2 = np.max(driver['E'][2][50:99])
    # assert np.isclose(max_electric_field_2, T_E, atol=0.1)

    min_electric_field_1 = np.min(driver['E'][2][0:49])
    # assert np.isclose(min_electric_field_1, R_E, atol=0.1)

    driver['E'][:] = initialFieldE[:]
    driver['H'][:] = initialFieldH[:]
    driver['J'][:] = initialFieldJ[:]
    for _ in range(300):
        driver.step()
        plt.plot(sp.x, driver['E'],'b')
        plt.plot(sp.x, driver['H'],'r')
        plt.ylim(-1, 1.5)
        plt.grid(which='both')
        plt.pause(0.01)
        plt.cla()
