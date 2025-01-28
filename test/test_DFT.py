import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pytest
import time



# Función para realizar la DFT manualmente
def dft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for i in range(N):
        for n in range(N):
            X[i]=X[i]+ x[n] * np.exp(-2j*np.pi*i*n/N)
    return X

# Monto la gaussiana
x = np.linspace(-5, 5, 100) 
s0 = 0.50
gaussiana = np.exp(-(x) ** 2 / (2 * s0 **2))  # Gaussiana

# FFT y DFT
fft_g = np.fft.fft(gaussiana)

dft_g = dft(gaussiana)



# Grafico
plt.figure(figsize=(12, 6))
# Ejes de frecuencia para FFT y DFT
freq = np.fft.fftfreq(len(x), x[1] - x[0])

# Original
plt.subplot(1, 3, 1)
plt.plot(x, gaussiana, label='Gaussiana')
plt.title('Gaussiana')
plt.xlabel('x')
plt.ylabel('Amplitud')
plt.grid()
plt.legend()

# DFT
plt.subplot(1, 3, 2)
plt.plot(freq, np.abs(fft_g), label='FFT')
plt.title('Transformada FFT')
plt.xlabel('Frecuencia')
plt.ylabel('Amplitud')
plt.grid()
plt.legend()

# FFT
plt.subplot(1, 3, 3)
plt.plot(freq, np.abs(dft_g), label='DFT', linestyle='--')
plt.title('Transformada DFT')
plt.xlabel('Frecuencia')
plt.ylabel('Amplitud')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
