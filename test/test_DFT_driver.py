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

def dft(x,freq,time):
    X=[]
    for f in range(len(freq)):
        summatory=0.
        for t in range(len(time)):
            summatory=summatory + x[t] * np.exp(-2j*np.pi*freq[f]*time[t])
        X.append(summatory)
    return X

def test_DFT_driver():

    #Material distribution
    epsilon_1=1
    # epsilon_2=1
    L1=-5.0
    L2=5.0
    elements=100
    epsilons = epsilon_1*np.ones(elements)
    sigmas=np.zeros(elements) 
    sigmas[45:55]=2.
    # epsilons[int(elements/2):elements-1]=epsilon_2

    sp = DG1D(
        n_order=3,
        mesh=Mesh1D(L1, L2, elements, boundary_label="PEC"),
        epsilon=epsilons,
        sigma=sigmas
    )
    driver = MaxwellDriver(sp)

    #Type of wave
    s0 = 0.50
    x0=-2
    final_time = 1
    steps = 1000
    time_vector = np.linspace(0, final_time, steps)
    freq_vector = np.logspace(6, 9, 31)

    initialFieldE = np.exp(-(sp.x-x0)**2/(2*s0**2))
    initialFieldH = initialFieldE

    #Driver operates
    driver['E'][:] = initialFieldE[:]
    driver['H'][:] = initialFieldH[:]
    E_vector_1 = []
    E_vector_2=[]

    #DFT calculations
    for t in range(len(time_vector)):
        driver.run_until(time_vector[t])
        E_vector_1.append(driver['E'][2][0:45])
        E_vector_2.append(driver['E'][2][55:100])


    dft_E_1=dft(E_vector_1,freq_vector,time_vector)
    dft_E_2=dft(E_vector_2,freq_vector,time_vector)

    # Plot the DFTs
    plt.figure(figsize=(10, 6))
    plt.plot(freq_vector, np.abs(dft_E_1), label='', color='blue')
    plt.plot(freq_vector, np.abs(dft_E_2), label='', color='red')
    plt.xscale('log')  
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('DFT of Electric Fields')
    plt.legend()
    plt.grid(True)
    plt.show()

def test_DFT_point():

    #Material distribution
    epsilon_1=1
    # epsilon_2=1
    L1=-5.0
    L2=5.0
    elements=100
    epsilons = epsilon_1*np.ones(elements)
    sigmas=np.zeros(elements) 
    sigmas[45:55]=2.
    # epsilons[int(elements/2):elements-1]=epsilon_2

    sp = DG1D(
        n_order=3,
        mesh=Mesh1D(L1, L2, elements, boundary_label="SMA"),
        epsilon=epsilons,
        sigma=sigmas
    )
    driver = MaxwellDriver(sp)

    #Type of wave
    s0 = 0.50
    x0=-2
    final_time = 5
    steps = 100
    time_vector = np.linspace(0, final_time, steps)
    freq_vector = np.logspace(6, 9, 31)

    initialFieldE = np.exp(-(sp.x-x0)**2/(2*s0**2))
    initialFieldH = initialFieldE

    #Driver operates
    driver['E'][:] = initialFieldE[:]
    driver['H'][:] = initialFieldH[:]
    E_vector_R = []
    E_vector_T=[]
    E_vector_0=[]

    #DFT calculations
    for t in range(len(time_vector)):
        driver.run_until(time_vector[t])
        func=np.exp(-(time_vector[t]-x0)**2/(2*s0**2))
        E_vector_0.append(func)
        E_vector_R.append(driver['E'][2][5])
        E_vector_T.append(driver['E'][2][60])


    dft_E_R=dft(E_vector_R,freq_vector,time_vector)
    dft_E_T=dft(E_vector_T,freq_vector,time_vector)
    dft_0=dft(E_vector_0,freq_vector,time_vector)

    T = np.abs(np.array(dft_E_T)) / np.abs(np.array(dft_0))
    R = np.abs(np.array(dft_E_R)) / np.abs(np.array(dft_0))


    # Plot results
    fig, axes = plt.subplots(4, 1, figsize=(10, 16))
    
    axes[0].plot(freq_vector, np.abs(dft_E_R), label='DFT E_R', color='b')
    axes[0].plot(freq_vector, np.abs(dft_E_T), label='DFT E_T', color='r')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_title('DFT of E_R and E_T')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Magnitude')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(freq_vector, T, label='Transmission Coefficient', color='purple')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_title('Transmission Coefficient (T)')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude')
    axes[1].legend()
    axes[1].grid(True)
    
    axes[2].plot(freq_vector, R, label='Reflection Coefficient', color='orange')
    axes[2].set_xscale('log')
    axes[2].set_yscale('log')
    axes[2].set_title('Reflection Coefficient (R)')
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Magnitude')
    axes[2].legend()
    axes[2].grid(True)
    
    axes[3].plot(freq_vector, np.abs(dft_0), label='DFT E_0', color='g', linestyle='dashed')
    axes[3].set_xscale('log')
    axes[3].set_yscale('log')
    axes[3].set_title('DFT of E_0')
    axes[3].set_xlabel('Frequency (Hz)')
    axes[3].set_ylabel('Magnitude')
    axes[3].legend()
    axes[3].grid(True)
    
    plt.tight_layout()
    plt.show()
