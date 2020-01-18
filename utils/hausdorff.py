import numpy as np
import matplotlib.pyplot as plt
import os
from oct2py import octave


class HausDim:
    @staticmethod
    def get_dim(loop):
        octave.eval('pkg load image')
        octave.addpath(os.path.abspath('matlab/hausdorff'))

        categories = ['H0', 'HDE', 'HRE', 'QE', 'IAE', 'ISE', '\u03C3 Huber',
                      '\u03B2 Laplace', '\u03B3', '\u03C3 Gauss', '\u03B1',
                      'H3', 'H2', 'H1']
        values = [loop.h0, np.abs(loop.minHde / loop.hde), loop.minHre / loop.hre,
                  loop.minQe / loop.qe, loop.minIae / loop.iae,
                  loop.minIse / loop.ise, loop.minRsig / loop.rsig,
                  loop.minLb / loop.lb, loop.minSgam / loop.sgam,
                  loop.minGsig / loop.gsig, loop.salf - 1, loop.h3, loop.h2,
                  loop.h1, loop.h0]
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        ax = plt.subplot(polar=True)
        ax.set_rlabel_position(0)
        ax.spines['polar'].set_visible(False)
        ax.grid(visible=False)
        plt.xticks(visible=False)
        plt.yticks(visible=False)
        plt.ylim(0, 1)
        ax.set_theta_zero_location('N')
        ax.plot(angles, values, color='black')
        ax.fill(angles, values, color='black')

        filename = f'radar_{loop.id}.png'
        plt.savefig(filename, format='png')
        plt.close()
        result = octave.hausDim(filename)
        avg = np.average([float(item) for item in values[:-1]])
        os.remove(filename)

        return result, result*avg
