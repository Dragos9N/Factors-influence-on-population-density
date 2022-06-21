import numpy as np
from pandas import DataFrame


class ACP():
    def __init__(self, tabel, variabile_observate=None):
        if variabile_observate is None:
            self.v = list(tabel)
        else:
            self.v = variabile_observate
        self.__x = tabel[self.v].values

    def fit(self, std=True, nlib=0):
        # centrarea
        x_ = self.__x - np.mean(self.__x, axis=0)
        # standardizarea
        if std:
            x_ = x_ / np.std(self.__x, axis=0)
        n, m = np.shape(x_)  # n=numar instante, m=numar variabile
        matrice_cor_cov = (1 / (n - nlib)) * x_.T @ x_
        valori_proprii, vectori_proprii = np.linalg.eig(matrice_cor_cov)  # eig=calculeaza vectorii si valorile proprii
        sortare_valori_proprii = np.flip(np.argsort(valori_proprii))
        self.__alpha = valori_proprii[sortare_valori_proprii]
        self.__valori_proprii = vectori_proprii[:, sortare_valori_proprii]
        self.__componente = x_ @ self.__valori_proprii

        vector_denumiri = ["Val1", "Val2", "Val3", "Val4", "Val5"]
        self.etichete_componente = ["Componenta" + str(i + 1) for i in range(m)]
        if std:
            self.nrcomp_k = len(np.where(self.__alpha >= 1)[0])
        else:
            self.nrcomp_k = None
        pondere = np.cumsum(self.alpha / sum(self.alpha))
        self.nrcomp_p = len(np.where(pondere < 0.8)[0]) + 1
        eps = self.__alpha[:(m - 1)] - self.alpha[1:]
        sigma = eps[:(m - 2)] - eps[1:]
        negative = sigma < 0
        exista_negative = any(negative)

        if exista_negative:
            sortare_valori_proprii = np.where(negative)
            self.nrcomp_c = sortare_valori_proprii[0][0] + 2
        else:
            self.nrcomp_c = None
        if std:
            self.r_xc = self.valori_proprii * np.sqrt(self.alpha)
        else:
            self.r_xc = np.corrcoef(self.__x, self.componenta, rowvar=False)[:m, m:]

    def tabelare_varianta(self):
        procent_varianta = self.__alpha * 100 / sum(self.alpha)
        tabel_varianta = DataFrame(
            data={
                "Varianta": self.__alpha,
                "Procent varianta": procent_varianta,
                "Varianta cumulata": np.cumsum(self.__alpha),
                "Procent cumulat": np.cumsum(procent_varianta)
            }, index=self.etichete_componente
        )
        return tabel_varianta

    @property
    def alpha(self):
        return self.__alpha

    @property
    def valori_proprii(self):
        return self.__valori_proprii

    @property
    def componenta(self):
        return self.__componente