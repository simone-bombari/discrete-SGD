import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import utils
import losses

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


N = 5000000
sigma = 0.0001
dim = 1
positions = np.zeros((N, dim))
a = 1
eta = 1
cont_maker = 1
j = 0
dim = 1

#  Just one dimensional by now


for i in range(1, N):
    noise = eta * np.random.normal(0, sigma)
    positions[i] = positions[i - 1] + noise - eta * losses.grad_single_well_1d(positions[i - 1], a)


x = np.linspace(-3 * sigma * np.sqrt(eta / a), 3 * sigma * np.sqrt(eta / a), 1000)
p = np.zeros(len(x))
for i in range(len(x)):
    p[i] = utils.pdf_single_well_1d(x[i], sigma, eta, a)


fig, ax = plt.subplots()
fig.set_size_inches(3, 2)
bins = 500
ax.hist(positions, bins=bins, density=True)
ax.plot(x, p, 'r')
fig.savefig('prova.pdf', format='pdf')
plt.show()
