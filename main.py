import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import utils
import losses
from datetime import datetime

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def dynamics_and_plot(param, N):
    eta, a, sigma = param
    positions = np.zeros(N)
    for i in range(1, N):
        noise = eta * np.random.normal(0, sigma)
        positions[i] = positions[i - 1] + noise - eta * losses.grad_single_well_1d(positions[i - 1], a)

    x = np.linspace(-6 * sigma * np.sqrt(eta / a), 6 * sigma * np.sqrt(eta / a), 1000)
    y = losses.single_well_1d(x, a)
    p = utils.pdf_single_well_1d(x, sigma, eta, a)
    fig, ax = plt.subplots()
    fig.set_size_inches(3, 2)
    bins = 500
    ax.hist(positions, bins=bins, density=True, label=r'$p_d(x)$')
    ax.plot(x, p, 'r', label=r'$p_c(x)$')
    ax.plot(x, y, 'k', linestyle='--', label=r'$L(x)$')
    ax.legend(loc="upper right")
    ax.set_xlabel(r'$x$')
    ax.set_ylim([0, 5 * max(p) / 4])
    ax.set_xlim([-4 * sigma * np.sqrt(eta / a), 4 * sigma * np.sqrt(eta / a)])
    plt.tick_params(
        axis='both',
        which='both',
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False)
    anchored_text = AnchoredText(r'$\eta={}$'.format(eta) + '\n' + r'$a={}$'.format(a) +
                                 '\n' + r'$\sigma={}$'.format(sigma), loc='upper left')
    ax.add_artist(anchored_text)
    fig.savefig('{}_single_well_1d_{}.pdf'.format(datetime.now(), param), format='pdf')
    plt.show()

    return 0

dim = 1
N = 10000000
eta = [np.sqrt(1.5)]
a = [np.sqrt(1.5)]
sigma = [1]

params = []
for i in range(len(eta)):
    params.append([eta[i], a[i], sigma[i]])

for param in params:
    dynamics_and_plot(param, N)
