import matplotlib.pyplot as plt
import numpy as np

AP = [0.10,0.10, 0.849, 0.844, 0.853,0.853,0.0]
OVERLAP = [0.0, 0.2, 0.4, 0.6, 0.8,0.9,1.0]

plt.plot(OVERLAP, AP, label='r=4, l=500')

AP = [0.119,0.119, 0.849, 0.844, 0.853,0.853,0.0]
OVERLAP = [0.0, 0.2, 0.4, 0.6, 0.8,0.9,1.0]

plt.plot(OVERLAP, AP, label='r=4, l=250')

plt.legend()
plt.show()