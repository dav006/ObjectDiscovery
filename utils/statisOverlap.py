import matplotlib.pyplot as plt
import numpy as np

AP = [0.88,0.83,0.74,0.7,0.67,0.66,0.57,0.56,0.5,0.39]
OVERLAP = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18,0.20]

plt.plot(OVERLAP, AP,'r', label='r=2, vocab = 1M')

AP = [0.86,0.89,0.89, 0.89, 0.87, 0.85,0.83,0.78,0.75]
OVERLAP = [0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18,0.20]

plt.plot(OVERLAP, AP,'r--', label='r=3, vocab = 1M')

AP = [0.92,0.89,0.83,0.8,0.74,0.7,0.68,0.57,0.45,0.46]
OVERLAP = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18,0.20]

plt.plot(OVERLAP, AP,'b', label='r=2, vocab = 500mil')

AP = [0.78,0.86,0.92,0.925,0.9,0.9,0.87,0.83,0.81]
OVERLAP = [ 0.04,0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18,0.20]

plt.plot(OVERLAP, AP,'b--', label='r=3, vocab = 500mil')

AP = [0.966,0.959,0.93,0.9,0.84,0.77,0.58,0.57,0.43,0.30]
OVERLAP = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18,0.20]

plt.plot(OVERLAP, AP,'g', label='r=2, vocab = 150mil')


AP = [0.92,0.964,0.947,0.94,0.92,0.9,0.87,0.84,0.79]
OVERLAP = [ 0.04,0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18,0.20]

plt.plot(OVERLAP, AP,'g--', label='r=3, vocab = 150mil')


plt.xticks([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18,0.20])
plt.xlabel("$\eta$")
plt.ylabel('MPPM')
plt.legend()
plt.show()


AP = [0.87,0.92,0.9,0.89,0.86,0.82,0.74,0.72,0.69,0.65]
OVERLAP = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18,0.20]

plt.plot(OVERLAP, AP,'r', label='r=2, vocab = 1M')

AP = [0.77,0.81,0.87,0.89,0.89,0.91,0.89,0.88,0.86]
OVERLAP = [0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18,0.20]

plt.plot(OVERLAP, AP,'r--', label='r=3, vocab = 1M')

AP = [0.89,0.9,0.9,0.87,0.83,0.8,0.69,0.7,0.63,0.48]
OVERLAP = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18,0.20]

plt.plot(OVERLAP, AP,'b', label='r=2, vocab = 500mil')

AP = [0.83,0.87,0.89,0.91,0.88,0.87,0.85,0.82,0.81]
OVERLAP = [ 0.04,0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18,0.20]

plt.plot(OVERLAP, AP,'b--', label='r=3, vocab = 500mil')

AP = [0.71,0.643,0.489,0.489,0.412,0.43,0.258,0.26,0.15,0.09]
OVERLAP = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18,0.20]

plt.plot(OVERLAP, AP,'g', label='r=2, vocab = 150mil')


AP = [0.7,0.68,0.559,0.56,0.534,0.516,0.5,0.489,0.45]
OVERLAP = [ 0.04,0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18,0.20]

plt.plot(OVERLAP, AP,'g--', label='r=3, vocab = 150mil')

plt.xticks([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18,0.20])
plt.xlabel("$\eta$")
plt.ylabel('MPPM')
plt.legend()
plt.show()

AP = [0.0187,0.007,0.003]
OVERLAP = [0.02, 0.04, 0.06]

plt.plot(OVERLAP, AP,'r', label='r=2, vocab = 250mil')

AP = [0.002, 0.0008,0.0003]
OVERLAP = [0.02, 0.04, 0.06]

plt.plot(OVERLAP, AP,'b', label='r=2, vocab = 1M')

plt.xticks([0.02, 0.04, 0.06])
plt.xlabel("$\eta$")
plt.ylabel('MPPM')
plt.legend()
plt.show()

AP = [ 0.876721918034,0.86900497623,0.841066090403,0.834314468,0.802461031]
Landmark_Number = [11, 20, 40,80,160]

plt.plot(Landmark_Number, AP,'r', label='r=2, vocab = 150mil, l=1733')

plt.xticks([11, 20, 40,80,160])
plt.xlabel("# Landmarks")
plt.ylabel('MPPM')
plt.legend()
plt.show()


AP = [0.841066090403 ,0.834499911858 ,0.826280263164]
Data_Landark = [500, 1000, 2000]

plt.plot(Data_Landark, AP,'r', label='r=2, vocab = 150mil, l=1733, Landmarks = 40')

plt.xticks([500, 1000, 2000])
plt.xlabel("# Datos por Landmark")
plt.ylabel('MPPM')
plt.legend()
plt.show()

AP = [0.93335118, 0.96028993, 0.95582548, 0.95357094, 0.95270414, 0.95650254, 0.9529572, 0.95180994, 0.95255141, 0.94997491, 0.95574393, 0.95549736, 0.95665352, 0.95006933, 0.94488219 ]
Data_Landark = range(1,16)

plt.plot(Data_Landark, AP,'r', label='r=2, vocab = 25mil, l=1733, Landmarks = 40')


AP = [0.92426702, 0.93970354, 0.93487948, 0.93255031, 0.93039048, 0.93610562, 0.93133564, 0.93354285, 0.93943454, 0.93100161, 0.93544558, 0.93220242, 0.93609067, 0.93082225, 0.92773195 ]
plt.plot(Data_Landark, AP,'b', label='r=2, vocab = 10mil, l=1733, Landmarks = 40')

plt.xticks(Data_Landark)
plt.xlabel("# Epochs")
plt.ylabel('Accuracy')
plt.legend()
plt.show()

AP = [0.447230350097, 0.404687632193, 0.458237573921, 0.476294308065, 0.464634537689, 0.475380515578, 0.482560172841, 0.463802565256,  0.458003264948, 0.466841523415, 0.429461013107, 0.479016540466, 0.407604326987, 0.461815583589, 0.465546477975, 0.427378318172 ]
Data_Landark = range(0,16)

plt.plot(Data_Landark, AP,'r', label='r=2, vocab = 2,048, l=433, Landmarks = 40')

AP = [0.486377500657, 0.505750051552, 0.487009722883, 0.516706444054, 0.505782393247, 0.517173994333, 0.536100966173, 0.50068639019,  0.512980673555, 0.473410289035, 0.521437577442, 0.521897141064, 0.478840738738, 0.521933699805, 0.52757512809,0.52757512809 ]
plt.plot(Data_Landark, AP,'b', label='r=2, vocab = 10mil, l=433, Landmarks = 40')

plt.xticks(Data_Landark)
plt.xlabel("# Epochs")
plt.ylabel('MPPM')
plt.legend()
plt.show()