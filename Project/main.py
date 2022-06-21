import numpy as np
import pandas as pd
from functions import inlocuire_valori_nule,csv_save,tabelare_matrice
from functions import plot_varianta, corelograma,plot_instante,plot_corelatii
from Principal_component_analysis import ACP
from tkinter import *


#PRELUAREA SETURILOR DE DATE

tabel_densitate=pd.read_csv("date/Population density.csv", index_col=0)
inlocuire_valori_nule(tabel_densitate)

tabel_schimburi=pd.read_csv("date/Merchandise trade.csv", index_col=0)
inlocuire_valori_nule(tabel_schimburi)

tabel_paduri=pd.read_csv("date/Forest area.csv", index_col=0)
inlocuire_valori_nule(tabel_paduri)

tabel_cheltuieli_militare=pd.read_csv("date/Military expenditure.csv", index_col=0)
inlocuire_valori_nule(tabel_cheltuieli_militare)

tabel_acces_electricitate=pd.read_csv("date/Electricity access.csv", index_col=0)
inlocuire_valori_nule(tabel_acces_electricitate)


tabel_combinat1=tabel_densitate.merge(right=tabel_schimburi,left_on="Country Name",right_index=True)
tabel_combinat2=tabel_combinat1.merge(right=tabel_paduri,left_on="Country Name",right_index=True)
tabel_combinat3=tabel_combinat2.merge(right=tabel_cheltuieli_militare,left_on="Country Name",right_index=True)
tabel_combinat=tabel_combinat3.merge(right=tabel_acces_electricitate,left_on="Country Name",right_index=True)

nume_coloane=list(tabel_combinat.columns)#obtinerea numelor coloanelor
x=tabel_combinat[nume_coloane].values#obtinerea valorilor din tabel


#TABELE

numar_instante,numar_variabile=x.shape
corelatie=np.corrcoef(x,rowvar=False)
csv_save(corelatie, nume_coloane, nume_coloane, "Tables/Corelatie.csv")


covarianta=np.cov(x,rowvar=False,ddof=0)
csv_save(covarianta, nume_coloane, nume_coloane, "Tables/Covarianta.csv")


analiza_componente_principala=ACP(tabel_combinat,list(tabel_combinat))
analiza_componente_principala.fit()
tabel_varianta=analiza_componente_principala.tabelare_varianta()
tabel_varianta.to_csv("Tables/Varianta.csv")


t_scoruri = tabelare_matrice(
    analiza_componente_principala.componenta / np.sqrt(analiza_componente_principala.alpha),
    tabel_combinat.index, analiza_componente_principala.etichete_componente,
    "Tables/Scoruri.csv"
)


componenta_patrat = analiza_componente_principala.componenta * analiza_componente_principala.componenta
cosin = (componenta_patrat.T / np.sum(componenta_patrat, axis=1)).T#impartirea fiecarei linii pe coloane
tabel_cosinusuri = tabelare_matrice(cosin, tabel_combinat.index,
                                    analiza_componente_principala.etichete_componente,
                           "Tables/Cosinusuri.csv")



beta = componenta_patrat * 100 / np.sum(componenta_patrat, axis=0)
contributii = tabelare_matrice(beta, tabel_combinat.index,
                          analiza_componente_principala.etichete_componente,
                          "Tables/Contributii.csv")




#varianta explicata in comun
r2 = analiza_componente_principala.r_xc * analiza_componente_principala.r_xc
comunalitate = np.cumsum(r2, axis=1)
comunalitati = tabelare_matrice(comunalitate, analiza_componente_principala.v,
                                analiza_componente_principala.etichete_componente,
                          "Tables/Comunalitati.csv")





#GRAFICE
matrice_tabelata = tabelare_matrice(analiza_componente_principala.r_xc,
                          analiza_componente_principala.v,
                          analiza_componente_principala.etichete_componente)


def corelograma_grafic():
    corelograma(matrice_tabelata)


def cerc_corelatii_1_2():
    plot_corelatii(matrice_tabelata,"Componenta1","Componenta2",titlu="Cercul corelațiilor componenta 1-componenta 2",aspect=1)

def cerc_corelatii_1_3():
    plot_corelatii(matrice_tabelata,"Componenta1","Componenta3",titlu="Cercul corelațiilor componenta 1-componenta 3",aspect=1)

def cerc_corelatii_1_4():
    plot_corelatii(matrice_tabelata,"Componenta1","Componenta4",titlu="Cercul corelațiilor componenta 1- componenta 4",aspect=1)

def cerc_corelatii_1_5():
    plot_corelatii(matrice_tabelata,"Componenta1","Componenta5",titlu="Cercul corelațiilor componenta 1-componenta 5",aspect=1)


def corelograma_comunalitati():
    corelograma(comunalitati,vmin=0,titlu="Corelogramă de comunalități")

def grafic_varianta():
    plot_varianta(analiza_componente_principala,titlu="Varianța componentelor")


def scoruri_comp_1_2():
    plot_instante(t_scoruri,"Componenta1","Componenta2",aspect=1,titlu="Plot scoruri componenta 1-componenta 2")

def scoruri_comp_1_3():
    plot_instante(t_scoruri,"Componenta1","Componenta3",aspect=1,titlu="Plot scoruri componenta 1-componenta 3")

def scoruri_comp_1_4():
    plot_instante(t_scoruri,"Componenta1","Componenta4",aspect=1,titlu="Plot scoruri componenta 1-componenta 4")

def scoruri_comp_1_5():
    plot_instante(t_scoruri,"Componenta1","Componenta5",aspect=1,titlu="Plot scoruri componenta 1-componenta 5")











#DEZVOLTAREA GRAFICA A APLICATIEI

window=Tk()
window.geometry("400x600")

grafic_varianta_btn=Button(window,text="Varianța")
grafic_varianta_btn.config(command=grafic_varianta)
grafic_varianta_btn.pack()

corelograma_btn=Button(window,text="Corelograma")
corelograma_btn.config(command=corelograma_grafic)
corelograma_btn.pack()

scoruri1_btn=Button(window,text="Scoruri între componentele 1 și 2")
scoruri1_btn.config(command=scoruri_comp_1_2)
scoruri1_btn.pack()

scoruri2_btn=Button(window,text="Scoruri între componentele 1 și 3")
scoruri2_btn.config(command=scoruri_comp_1_3)
scoruri2_btn.pack()

scoruri3_btn=Button(window,text="Scoruri între componentele 1 și 4")
scoruri3_btn.config(command=scoruri_comp_1_4)
scoruri3_btn.pack()

scoruri4_btn=Button(window,text="Scoruri între componentele 1 și 5")
scoruri4_btn.config(command=scoruri_comp_1_5)
scoruri4_btn.pack()

cercuri1_btn=Button(window,text="Cercul corelațiilor între componentele 1 și 2")
cercuri1_btn.config(command=cerc_corelatii_1_2)
cercuri1_btn.pack()

cercuri2_btn=Button(window,text="Cercul corelațiilor între componentele 1 și 3")
cercuri2_btn.config(command=cerc_corelatii_1_3)
cercuri2_btn.pack()

cercuri3_btn=Button(window,text="Cercul corelațiilor între componentele 1 și 4")
cercuri3_btn.config(command=cerc_corelatii_1_4)
cercuri3_btn.pack()

cercuri4_btn=Button(window,text="Cercul corelațiilor între componentele 1 și 5")
cercuri4_btn.config(command=cerc_corelatii_1_5)
cercuri4_btn.pack()

corelograma_comunalitati_btn=Button(window,text="Corelograma comunalităților")
corelograma_comunalitati_btn.config(command=corelograma_comunalitati)
corelograma_comunalitati_btn.pack()


#pozitionare și culori

grafic_varianta_btn.config(font=('Ink Free',20,'bold'))
grafic_varianta_btn.place(x=0,y=0)
grafic_varianta_btn.config(bg="#ffc6ff")
grafic_varianta_btn.config(fg="#1e6091")

corelograma_btn.config(font=('Ink Free',20,'bold'))
corelograma_btn.place(x=225,y=0)
corelograma_btn.config(bg="#bdb2ff")
corelograma_btn.config(fg="#1e6091")

scoruri1_btn.config(font=('Ink Free',10,'bold'))
scoruri1_btn.place(x=70,y=100)
scoruri1_btn.config(bg="#a0c4ff")
scoruri1_btn.config(fg="#1e6091")
scoruri2_btn.config(font=('Ink Free',10,'bold'))
scoruri2_btn.place(x=70,y=130)
scoruri2_btn.config(bg="#a0c4ff")
scoruri2_btn.config(fg="#1e6091")
scoruri3_btn.config(font=('Ink Free',10,'bold'))
scoruri3_btn.place(x=70,y=160)
scoruri3_btn.config(bg="#a0c4ff")
scoruri3_btn.config(fg="#1e6091")
scoruri4_btn.config(font=('Ink Free',10,'bold'))
scoruri4_btn.place(x=70,y=190)
scoruri4_btn.config(bg="#a0c4ff")
scoruri4_btn.config(fg="#1e6091")

cercuri1_btn.config(font=('Ink Free',10,'bold'))
cercuri1_btn.place(x=40,y=250)
cercuri1_btn.config(bg="#caffbf")
cercuri1_btn.config(fg="#1e6091")
cercuri2_btn.config(font=('Ink Free',10,'bold'))
cercuri2_btn.place(x=40,y=280)
cercuri2_btn.config(bg="#caffbf")
cercuri2_btn.config(fg="#1e6091")
cercuri3_btn.config(font=('Ink Free',10,'bold'))
cercuri3_btn.place(x=40,y=310)
cercuri3_btn.config(bg="#caffbf")
cercuri3_btn.config(fg="#1e6091")
cercuri4_btn.config(font=('Ink Free',10,'bold'))
cercuri4_btn.place(x=40,y=340)
cercuri4_btn.config(bg="#caffbf")
cercuri4_btn.config(fg="#1e6091")

corelograma_comunalitati_btn.config(font=('Ink Free',20,'bold'))
corelograma_comunalitati_btn.place(x=15,y=400)
corelograma_comunalitati_btn.config(bg="#ffadad")
corelograma_comunalitati_btn.config(fg="#1e6091")

label=Label(window,text="Influența factorilor asupra densității populației")
label.pack()
label.place(x=10,y=568)
label.config(font=('Ink Free',12,'bold'))
label.config(bg="#d8e2dc")

window.config(bg="#ffe5d9")
window.mainloop()