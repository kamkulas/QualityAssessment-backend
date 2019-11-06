import numpy as np
import os
from math import pow, isnan, log, log10, floor, gamma, pi
from scipy.stats import norm
from statistics import mean, stdev
from oct2py import octave
from decimal import Decimal
import sys

LAMBDA = Decimal(0.2)
TP = 5


class Calculator:
    def __init__(self, loop):
        self.loop = loop
        self.error = []
        for i in range(len(self.loop.cv)):
            self.error.append(self.loop.cv[i] - self.loop.spa[i])

    def int_indexes(self, mv, error):
        n = len(error)
        ise = 0
        iae = 0
        qe = 0
        for i in range(n):
            ise += Decimal(pow(error[i], 2))
            iae += Decimal(abs(error[i]))
            qe += Decimal(pow(error[i], 2)) + LAMBDA*Decimal(pow(mv[i], 2))

        ise /= n
        iae /= n
        qe /= n
        return {'ise': ise, 'iae': iae, 'qe': qe}

    def stat_indexes(self, loop_error):
        octave.addpath(os.path.abspath('matlab/FitFunc'))
        octave.addpath(os.path.abspath('matlab/stbl'))
        octave.addpath(os.path.abspath('matlab/matlab/LIBRA'))
        octave.eval('pkg load statistics')

        error = [float(err) for err in loop_error]

        nbins = 140
        d_std = 3.0
        mu = float(mean(loop_error))
        st = float(stdev(loop_error))

        # dopasowanie rozkladu Gaussa dla uchybu
        xmin = mu - d_std * st
        xmax = mu + d_std * st
        stt = (xmax - xmin) / nbins
        ax = np.arange(xmin, xmax, stt)
        ax1 = np.arange(xmin, xmax, stt / 10)
        n, xbin = np.histogram(error, bins=nbins)
        n[0] = 0
        n[1] = 0
        n[nbins - 2] = 0
        n[nbins - 3] = 0
        rr1 = norm.pdf(ax, mu, st)
        rangex = xmax - xmin
        binwidth = rangex / nbins
        row = 0
        for item in loop_error:
            row += item > 0
        rr1n = row * (rr1 * binwidth)
        rr1 = norm.pdf(ax1, mu, st)
        rr1n = row * (rr1 * binwidth)
        gauss = rr1n[0::10]

        ppp = octave.stblfit(error, 'percentile')
        Salf = ppp[0][0]
        beta = ppp[1][0]
        Sgam = ppp[2][0]
        mu = ppp[3][0]
        rr3 = octave.stblpdf(ax1, Salf, beta, Sgam, mu)[0]
        rr3n = row * (rr3 * binwidth)
        levy = rr3n[0::10]

        # dopasowanie rozkladu Laplace'a dla uchybu regulacji
        params = octave.fit_ML_laplace(error)
        u = params['u']
        Lb = params['b']
        rr4 = (1 / (2 * Lb)) * (np.exp(-abs(ax1 - u) / Lb))
        rr4n = row * (rr4 * binwidth)
        laplace = rr4n[0::10]

        # Huber
        x0 = octave.mloclogist(error)
        Rsig = octave.mscalelogist(error)
        rr5 = norm.pdf(ax1, x0, Rsig)
        rr5n = row * (rr5 * binwidth)
        huber = rr5n[0::10]

        # funkcje pdf
        nn, xx = np.histogram(error, bins=ax)
        histX = xx
        histY = nn
        KK = len(nn)

        return {'gsig': st,
                'salf': Salf,
                'sgam': Sgam,
                'lb': Lb,
                'rsig': Rsig,
                'gauss': [Decimal(g) for g in gauss],
                'levy': [Decimal(l) for l in levy],
                'laplace': [Decimal(l) for l in laplace],
                'huber': [Decimal(h) for h in huber],
                'histX': [Decimal(float(x)) for x in histX],
                'histY': [Decimal(float(y)) for y in histY]}

    def entrophy(self, loop_error):
        nbins = 400
        error = [float(err) for err in loop_error]
        xmin = min(error)
        xmax = max(error)
        dx = (xmax - xmin) / nbins
        XX = np.arange(xmin, xmax+1, dx)
        histX, histY = np.histogram(error, bins=XX)
        KK = len(histX)
        hre = 0
        hde = 0
        for i in range(0, KK - 1):
            if histY[i] > 0:
                hre += histY[i] * log10(histY[i] / (1 + histY[i])) * dx
                hde += histY[i] * log(histY[i]) * dx
        hre *= -1
        hde *= -1
        return {'hre': Decimal(hre),
                'hde': Decimal(hde)}

    def _divisors(self, n, n0):
        tmp = n0
        d = []
        while tmp < n:
            if n % tmp == 0:
                d.append(int(tmp))
            tmp += 1
        return d

    def _RScalc(self, Z, n):
        Z = np.array(Z)
        m = int(len(Z)/n)
        Y = np.reshape(Z, (m, n))
        E = np.mean(Y, axis=1)
        S = np.std(Y, axis=1, ddof=1)
        for i in range(0, m):
            Y[i, :] = Y[i, :] - E[i]
        Y = np.cumsum(Y, axis=1)
        MM = np.max(Y, axis=1) - np.min(Y, axis=1)
        CS = MM/S
        return np.mean(CS)

    def hurst(self, loop_error):
        octave.addpath(os.path.abspath('matlab/hurst'))
        d = 10
        dmin = d
        N = len(loop_error)
        N0 = int(floor(0.99*N))
        dv = np.zeros((N-N0+1, 1))
        for i in range(N0, N+1):
            dv[i - N0] = len(self._divisors(i, dmin))
        OptN = N0 + np.argmax(dv)
        x = [float(item) for item in loop_error[0:OptN]]
        d = self._divisors(OptN, dmin)
        N = len(d)
        RSe = np.zeros((N, 1))
        ERS = np.zeros((N, 1))

        for i in range(0, N):
            RSe[i] = self._RScalc(x, d[i])

        for i in range(0, N):
            n = d[i]
            K = list(range(1, n))
            ratio = (n-0.5)/n * np.sum(np.sqrt((np.ones((1, n-1))*n-K)/K))
            if n > 340:
                ERS[i] = ratio/np.sqrt(0.5*pi*n)
            else:
                ERS[i] = (gamma(0.5*(n-1))*ratio) / (gamma(0.5*n)*np.sqrt(pi))
        
        QQ = len(RSe) - 1
        for i in range(QQ, 0, -1):
            if isnan(RSe[i]):
                RSe[i] = RSe[i+1]
        
        xx = np.log10(d)
        
        ERSal = np.sqrt(np.multiply(0.5*pi, d))
        ERSal = np.reshape(ERSal, (len(ERSal), 1))
        Pal = np.polyfit(xx, np.log10(RSe - ERS + ERSal), 1)
        Hal = Pal[0]
        Haa = np.polyval(Pal, xx)
        Pe = np.polyfit(xx, np.log10(RSe), 1)
        He = Pe[0]
        P = np.polyfit(xx, np.log10(ERS), 1)
        Ht = P[0]
        
        L = np.log2(OptN)
        
        pval95 = [0.5-np.exp(-7.33*np.log(np.log(L))+4.21), np.exp(-7.20*np.log(np.log(L))+4.04)+0.5]
        C = np.array([0.5-np.exp(-7.35*np.log(np.log(L))+4.06), np.exp(-7.07*np.log(np.log(L))+3.75)+0.5, .90])
        C = np.vstack([C, np.concatenate([np.array(pval95), np.array([.95])])])
        C = np.vstack([C, np.array([0.5-np.exp(-7.19*np.log(np.log(L))+4.34), np.exp(-7.51*np.log(np.log(L))+4.58)+0.5, .99])])
        Htcorr = 0.5
        
        w = np.array([Ht, Hal])
        yy = np.log10(RSe-ERS+ERSal)
        
        xx = np.reshape(xx, (len(xx), 1))
        result = octave.MultiFit3(xx, yy)
        H1 = result['H1']
        H2 = result['H2']
        H3 = result['H3']
        cr1 = int(result['cr1'])
        cr2 = int(result['cr2'])
        P1 = result['P1'][0]
        P1 = np.reshape(P1, (len(P1), 1))
        P3 = result['P3'][0]
        P3 = np.reshape(P3, (len(P3), 1))
        
        br3 = np.array([pow(10, xx[cr1-1])*TP, pow(10, xx[cr2-1])*TP])
        crossX1 = xx[cr1-1]
        crossY1 = yy[cr1-1]
        crossX2 = xx[cr2-1]
        crossY2 = yy[cr2-1]
        H0 = Hal[0]
        cr1 = br3[0]
        cr2 = br3[1]
        nnn = len(xx)
        xp = np.array([xx[0], crossX1, crossX2, xx[-1]])
        yp = np.array([np.polyval(P1, xp[0]), crossY1, crossY2, np.polyval(P3, xp[-1])])

        return {
            'h0': H0,
            'h1': H1,
            'h2': H2,
            'h3': H3,
            'cr1': cr1,
            'cr2': cr2,
            'xx': xx,
            'yy': yy,
            'haa': Haa,
            'crossx1': crossX1,
            'crossy1': crossY1,
            'crossx2': crossX2,
            'crossy2': crossY2,
            'xp': xp,
            'yp': yp,
        }

    def calculate_all(self):
        print('Calculating int indexes...')
        int_indexes = self.int_indexes(self.loop.mv, self.error)
        print('Int indexes done.')
        print('Calculating stat indexes...')
        stat_indexes = self.stat_indexes(self.error)
        print('Stat indexes done.')
        print('Calculating entrophy...')
        entrophy = self.entrophy(self.error)
        print('Entrophy done.')
        print('Calculating Hurst indices...')
        hurst = self.hurst(self.error)
        print('Hurst done.')
        result = {
            **int_indexes,
            **stat_indexes,
            **entrophy,
            **hurst
        }

        if len(self.loop.cv) >= 4000:
            print('Will calculate minimum values...')
            min_values = {
              'minIse': sys.maxsize,
              'minIae': sys.maxsize,
              'minQe': sys.maxsize,
              'minHre': sys.maxsize,
              'minHde': sys.maxsize,
              'minCr1': sys.maxsize,
              'minCr2': sys.maxsize,
              'minGsig': sys.maxsize,
              'minSgam': sys.maxsize,
              'minLb': sys.maxsize,
              'minRsig': sys.maxsize,
            }
            iterations = len(self.loop.cv) // 2000
            for i in range(0, iterations):
                print(f'Iteration {i+1} of {iterations}')
                tmp_err = self.error[2000*i:2000*(i+1)]
                tmp_mv = self.loop.mv[2000*i:2000*(i+1)]

                partial_int_indexes = self.int_indexes(tmp_mv, tmp_err)
                tmp_ise = partial_int_indexes['ise']
                tmp_iae = partial_int_indexes['iae']
                tmp_qe = partial_int_indexes['qe']
                print('Int indexes', tmp_ise, tmp_iae, tmp_qe)
                if tmp_ise < min_values['minIse']:
                    min_values['minIse'] = tmp_ise
                if tmp_iae < min_values['minIae']:
                    min_values['minIae'] = tmp_iae
                if tmp_qe < min_values['minQe']:
                    min_values['minQe'] = tmp_qe

                partial_stat_indexes = self.stat_indexes(tmp_err)
                tmp_gsig = partial_stat_indexes['gsig']
                tmp_sgam = partial_stat_indexes['sgam']
                tmp_lb = partial_stat_indexes['lb']
                tmp_rsig = partial_stat_indexes['rsig']
                print('Stat indexes', tmp_gsig, tmp_sgam, tmp_lb, tmp_rsig)
                if tmp_gsig < min_values['minGsig']:
                    min_values['minGsig'] = tmp_gsig
                if tmp_sgam < min_values['minSgam']:
                    min_values['minSgam'] = tmp_sgam
                if tmp_lb < min_values['minLb']:
                    min_values['minLb'] = tmp_lb
                if tmp_rsig < min_values['minRsig']:
                    min_values['minRsig'] = tmp_rsig

                partial_entrophy = self.entrophy(tmp_err)
                tmp_hre = partial_entrophy['hre']
                tmp_hde = partial_entrophy['hde']
                print('Entrophy', tmp_hre, tmp_hde)
                if tmp_hre < min_values['minHre']:
                    min_values['minHre'] = tmp_hre
                if tmp_hde < min_values['minHde']:
                    min_values['minHde'] = tmp_hde

                partial_hurst = self.hurst(tmp_err)
                tmp_cr1 = partial_hurst['cr1']
                tmp_cr2 = partial_hurst['cr2']
                print('Hurst', tmp_cr1, tmp_cr2)
                if tmp_cr1 < min_values['minCr1']:
                    min_values['minCr1'] = tmp_cr1
                if tmp_cr2 < min_values['minCr2']:
                    min_values['minCr2'] = tmp_cr2
            print('Minimal values found: ', min_values)
            result['min_values'] = min_values
        print('All done!')
        return result
