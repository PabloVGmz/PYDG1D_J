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
    rho=1 # Ideally, we want conductivity (or resistivity) to also be a vector, similar to epsilon or mu.
    # The idea behind it is that each element can be made of a different material, with different properties.
    # We should still have epsilon and mu be consistent with the vector position of the material properties.
    # Where we have vacuum, rho = 0 and thus the J term will not affect the free space fields.
    # Where we have a conductive element, rho =/= 0 and thus the J term will affect the equations.
    # You can adjust epsilon and mu to the material properties, but for the easiest cases, 
    # inside the conductive material, you can set them to 1.
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
    initialFieldJ = v*rho #This one confuses me a bit, as you mention this variable is equal to the speed of the wave times conductivity.
    # initialFieldJ = np.zeros(initialFieldE.shape). Leaving rho aside, J itself should be (A m−2), and from this equation it'd be (m s-1),
    # which sadly doesn't make much sense. We do not need to have an initial field for J in almost any setup, as we should always avoid for
    # the initial field to be inside the material if possible.

    driver['E'][:] = initialFieldE[:]
    driver['H'][:] = initialFieldH[:]
    driver['J'][:] = initialFieldJ[:] # Same as before.
        
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
