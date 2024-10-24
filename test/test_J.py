import numpy as np
import matplotlib.pyplot as plt

from maxwell.driver import *
from maxwell.dg.mesh1d import *
from maxwell.dg.dg1d import *
from maxwell.fd.fd1d import *

def test_pec_dielectrico_upwind_J():
    
    Z_0 = 376.73
    
    # Defining material properties
    epsilon_1 = 1.0
    epsilon_2 = 1.0
    sigma_1=0.0
    sigma_2=20.0

    # Defining mesh properties
    sigmas = sigma_1*np.zeros(100)
    sigmas[50:99] = sigma_2 / Z_0
    epsilons = epsilon_1 * np.ones(100)
    epsilons[50:99] = epsilon_2 


    # Setting up DG1D simulation
    sp = DG1D(
        n_order=3,
        mesh=Mesh1D(-5.0, 5.0, 100, boundary_label="PEC"),
        epsilon=epsilons,
        sigma=sigmas
    )
    driver = MaxwellDriver(sp)

    # Initial conditions
    final_time = 6.0
    s0 = 0.50
    x_0 = -2.0
    initialFieldE = np.exp(-(sp.x - x_0) ** 2.0 / (2.0 * s0 ** 2.0))
    initialFieldH = initialFieldE

    # Initialize fields in driver
    driver['E'][:] = initialFieldE[:]
    driver['H'][:] = initialFieldH[:]
    
    # Run the simulation until the final time
    driver.run_until(final_time)

    # Animation loop
    driver['E'][:] = initialFieldE[:]
    driver['H'][:] = initialFieldH[:]

    for _ in range(300):
        driver.step()
        plt.plot(sp.x, driver['E'],'b')
        plt.plot(sp.x, driver['H'],'r')
        plt.ylim(-1.0, 1.5)
        plt.grid(which='both')
        plt.pause(0.01)
        plt.cla()