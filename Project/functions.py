import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype
import matplotlib.pyplot as plt
import seaborn as sb
from Principal_component_analysis import  ACP

def inlocuire_valori_nule(t):
    assert isinstance(t, pd.DataFrame)
    variabile = list(t.columns)
    for variabila in variabile:
        if any(t[variabila].isna()):
            if is_numeric_dtype(t[variabila]):
                t[variabila].fillna(t[variabila].mean(), inplace=True)
            else:
                modulul = t[variabila].mode()[0]
                t[variabila].fillna(modulul, inplace=True)


def csv_save(X, nume_linie, nume_coloana, nume_fisier="out.csv"):
    fisier = open(nume_fisier, "w")
    if nume_coloana is not None:
        if nume_linie is not None:
            fisier.write(",")
        fisier.write(",".join(nume_coloana) + "\n")
        n = np.shape(X)[0]
        for i in range(n):
            if nume_linie is not None:
                fisier.write(nume_linie[i] + ",")
            fisier.write(",".join([str(v) for v in X[i, :]]) + "\n")
        fisier.close()


def tabelare_matrice(x, nume_linii=None, nume_coloane=None, out=None):
    t = pd.DataFrame(x, nume_linii, nume_coloane)
    if out is not None:
        t.to_csv(out)
    return t



#GRAFICE

def corelograma(table, vmin=-1, vmax=1, titlu="Corelogramă corelații factoriale"):
    figura = plt.figure(figsize=(17, 10), facecolor="#ffccd5")
    assert isinstance(figura, plt.Figure)
    axa = figura.add_subplot(1, 1, 1)
    assert isinstance(axa, plt.Axes)
    axa.set_title(titlu, fontsize=20, color='#370617')
    ax_ = sb.heatmap(table, vmin=vmin, vmax=vmax, cmap="RdYlBu", annot=True, ax=axa)
    ax_.set_xticklabels(table.columns, rotation=30, ha="right", color="#370617")
    plt.show()


def plot_corelatii(tabel, variabila1, variabila2, titlu="Corelatii factoriale", aspect='auto'):
    figura = plt.figure(figsize=(17, 10), facecolor="#ffccd5")
    assert isinstance(figura, plt.Figure)
    axa = figura.add_subplot(1, 1, 1)
    assert isinstance(axa, plt.Axes)
    axa.set_title(titlu, fontdict={"fontsize": 16, "color": "#370617"})
    axa.set_xlabel(variabila1, fontdict={"fontsize": 12, "color": "#370617"})
    axa.set_ylabel(variabila2, fontdict={"fontsize": 12, "color": "#370617"})
    axa.set_aspect(aspect)
    u = np.arange(0,np.pi*2,0.01)
    axa.plot(np.cos(u),np.sin(u))
    axa.axvline(0,c='#fcbf49')
    axa.axhline(0,c='#fcbf49')
    axa.scatter(tabel[variabila1], tabel[variabila2], c="#d00000")
    for i in range(len(tabel)):
        axa.text(tabel[variabila1].iloc[i], tabel[variabila2].iloc[i], tabel.index[i])
    plt.show()


def plot_varianta(model_acp, titlu="Varianta componente"):
    assert isinstance(model_acp, ACP)
    fig = plt.figure(figsize=(13, 7),facecolor="#ffccd5")
    assert isinstance(fig, plt.Figure)
    ax = fig.add_subplot(1, 1, 1)
    assert isinstance(ax, plt.Axes)
    ax.set_title(titlu, fontdict={"fontsize": 16, "color": '#f72585'})
    ax.set_xlabel("Componente", fontdict={"fontsize": 12, "color": '#f72585'})
    ax.set_ylabel("Varianta", fontdict={"fontsize": 12, "color": '#f72585'})
    m = len(model_acp.alpha)
    x = np.arange(1, m + 1)
    ax.set_xticks(x)
    ax.plot(x, model_acp.alpha, color='#a3c4f3')
    ax.scatter(x, model_acp.alpha, c='#f72585')
    if model_acp.nrcomp_k is not None:
        ax.axhline(1, c='g', label="Kaiser")
    if model_acp.nrcomp_c is not None:
        ax.axhline(model_acp.alpha[model_acp.nrcomp_c - 1], c='#6a994e', label="Cattell")
    ax.axhline(model_acp.alpha[model_acp.nrcomp_p - 1], c='#ccff33', label="Procent acoperire > 80%")
    ax.legend()
    plt.show()



def plot_instante(tabel, variabila1, variabila2, titlu="Plot instante", aspect='auto'):
    figura = plt.figure(figsize=(10, 8),facecolor="#ffccd5")
    assert isinstance(figura, plt.Figure)
    axa = figura.add_subplot(1, 1, 1)
    assert isinstance(axa, plt.Axes)
    axa.set_title(titlu, fontdict={"fontsize": 16, "color": "#a3c4f3"})
    axa.set_xlabel(variabila1, fontdict={"fontsize": 12, "color": "#a3c4f3"})
    axa.set_ylabel(variabila2, fontdict={"fontsize": 12, "color": "#a3c4f3"})
    axa.set_aspect(aspect)
    axa.scatter(tabel[variabila1], tabel[variabila2], c="#ccff33")
    for i in range(len(tabel)):
        axa.text(tabel[variabila1].iloc[i], tabel[variabila2].iloc[i], tabel.index[i])
    plt.show()


