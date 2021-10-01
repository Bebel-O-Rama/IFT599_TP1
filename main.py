import numpy as np
import pandas as panda
import matplotlib.pyplot as plot
from dataclasses import dataclass

SETOSA = "setosa"
VERSICOLOR = "versicolor"
VIRGINICA = "virginica"

SEPAL_LENGTH = "sepal_length"
SEPAL_WIDTH = "sepal_width"
PETAL_LENGTH = "petal_length"
PETAL_WIDTH = "petal_width"

SPECIES = "species"

@dataclass
class IRISClass:
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    species: str


def getCentreClasse(dataset, nomClasse):
    classe = dataset[dataset[SPECIES] == nomClasse]
    classe = classe[[SEPAL_LENGTH, SEPAL_WIDTH, PETAL_LENGTH, PETAL_WIDTH]]
    sepalLengthMean = round(classe[SEPAL_LENGTH].mean(), 1)
    sepalWidthMean = round(classe[SEPAL_WIDTH].mean(), 1)
    petalLengthMean = round(classe[PETAL_LENGTH].mean(), 1)
    petalWidthMean = round(classe[PETAL_WIDTH].mean(), 1)
    return IRISClass(sepalLengthMean, sepalWidthMean, petalLengthMean, petalWidthMean, nomClasse)


def getEcartTypeClasse(dataset, nomClasse):
    classe = dataset[dataset[SPECIES] == nomClasse]
    classe = classe[[SEPAL_LENGTH, SEPAL_WIDTH, PETAL_LENGTH, PETAL_WIDTH]]
    sepalLengthStd = round(classe[SEPAL_LENGTH].std(), 1)
    sepalWidthStd = round(classe[SEPAL_WIDTH].std(), 1)
    petalLengthStd = round(classe[PETAL_LENGTH].std(), 1)
    petalWidthStd = round(classe[PETAL_WIDTH].std(), 1)
    return IRISClass(sepalLengthStd, sepalWidthStd, petalLengthStd, petalWidthStd, nomClasse)


# Methode 1a

# TODO: Rajouter des attributs avec la distance intra et interclasse
def distanceEuclide(classe, centreClasse):
    minDistance = float('inf')
    maxDistance = 0
    for i in classe.index:
        distanceSepalL = (classe[SEPAL_LENGTH].get(i) - centreClasse.sepal_length)
        distanceSepalW = (classe[SEPAL_WIDTH].get(i) - centreClasse.sepal_width)
        distancePetalL = (classe[PETAL_LENGTH].get(i) - centreClasse.petal_length)
        distancePetalW = (classe[PETAL_WIDTH].get(i) - centreClasse.petal_width)

        distance = np.sqrt(
            np.power(distanceSepalL, 2) + np.power(distanceSepalW, 2) + np.power(distancePetalL, 2) + np.power(
                distancePetalW, 2))
        if (distance > maxDistance):
            maxDistance = distance
        if (distance < minDistance):
            minDistance = distance

    return (minDistance, maxDistance)


def distanceMahalanobis(classeA, classeB, centreClasseB):
    minDistance = float('inf')
    maxDistance = 0
    matriceCov = np.zeros((4, 4), dtype=float)
    centre = np.array(
        [centreClasseB.sepal_length, centreClasseB.sepal_width, centreClasseB.petal_length, centreClasseB.petal_width])
    for i in classeB.index:
        currentPos = np.array(
            [classeB[SEPAL_LENGTH].get(i), classeB[SEPAL_WIDTH].get(i), classeB[PETAL_LENGTH].get(i),
             classeB[PETAL_WIDTH].get(i)])
        a = np.array(currentPos - centre)[np.newaxis]
        b = a * a.T
        matriceCov += b
    matriceCov = matriceCov / (len(classeB.index) - 1)
    matriceCov = np.linalg.inv(matriceCov)
    for i in classeA.index:
        currentPos = np.array(
            [classeA[SEPAL_LENGTH].get(i), classeA[SEPAL_WIDTH].get(i), classeA[PETAL_LENGTH].get(i),
             classeA[PETAL_WIDTH].get(i)])
        a = np.array(currentPos - centre)[np.newaxis]
        b = a.dot(matriceCov)
        c = b.dot(a.T)
        distance = np.sqrt(c)
        if (distance > maxDistance):
            maxDistance = distance
        if (distance < minDistance):
            minDistance = distance
    return (float(minDistance[0]), float(maxDistance[0]))


def methodeUnA(iris):
    setosaCentre = getCentreClasse(iris, SETOSA)
    versicolorCentre = getCentreClasse(iris, VERSICOLOR)
    virginicaCentre = getCentreClasse(iris, VIRGINICA)

    # On commence par trouver les distances euclidienne intraclasse et interclasse pour nos trois classes
    # Intraclasse
    intraClasseSetosaEuclid = distanceEuclide(iris[iris[SPECIES] == SETOSA], setosaCentre)
    intraClasseVersicolorEuclid = distanceEuclide(iris[iris[SPECIES] == VERSICOLOR], versicolorCentre)
    intraClasseVirginicaEuclid = distanceEuclide(iris[iris[SPECIES] == VIRGINICA], virginicaCentre)

    # Interclasse Setosa
    interClasseSetosaVersicolorEuclid = distanceEuclide(iris[iris[SPECIES] == SETOSA], versicolorCentre)
    interClasseSetosaVirginicaEuclid = distanceEuclide(iris[iris[SPECIES] == SETOSA], virginicaCentre)

    # Interclasse Versicolor
    interClasseVersicolorSetosaEuclid = distanceEuclide(iris[iris[SPECIES] == VERSICOLOR], setosaCentre)
    interClasseVersicolorVirginicaEuclid = distanceEuclide(iris[iris[SPECIES] == VERSICOLOR], virginicaCentre)

    # Interclasse Virginica
    interClasseVirginicaSetosaEuclid = distanceEuclide(iris[iris[SPECIES] == VIRGINICA], setosaCentre)
    interClasseVirginicaVersicolorEuclid = distanceEuclide(iris[iris[SPECIES] == VIRGINICA], versicolorCentre)

    # On va trouver les distances Mahalanobis

    # Intraclasse
    intraClasseSetosaMahal = distanceMahalanobis(iris[iris[SPECIES] == SETOSA], iris[iris[SPECIES] == SETOSA],
                                                 setosaCentre)
    intraClasseVersicolorMahal = distanceMahalanobis(iris[iris[SPECIES] == VERSICOLOR],
                                                     iris[iris[SPECIES] == VERSICOLOR], versicolorCentre)
    intraClasseVirginicaMahal = distanceMahalanobis(iris[iris[SPECIES] == VIRGINICA],
                                                    iris[iris[SPECIES] == VIRGINICA], virginicaCentre)

    # Interclasse Setosa
    interClasseSetosaVersicolorMahal = distanceMahalanobis(iris[iris[SPECIES] == SETOSA],
                                                           iris[iris[SPECIES] == VERSICOLOR], versicolorCentre)
    interClasseSetosaVirginicaMahal = distanceMahalanobis(iris[iris[SPECIES] == SETOSA],
                                                          iris[iris[SPECIES] == VIRGINICA], virginicaCentre)

    # Interclasse Versicolor
    interClasseVersicolorSetosaMahal = distanceMahalanobis(iris[iris[SPECIES] == VERSICOLOR],
                                                           iris[iris[SPECIES] == SETOSA], setosaCentre)
    interClasseVersicolorVirginicaMahal = distanceMahalanobis(iris[iris[SPECIES] == VERSICOLOR],
                                                              iris[iris[SPECIES] == VIRGINICA], virginicaCentre)

    # Interclasse Virginica
    interClasseVirginicaSetosaMahal = distanceMahalanobis(iris[iris[SPECIES] == VIRGINICA],
                                                          iris[iris[SPECIES] == SETOSA], setosaCentre)
    interClasseVirginicaVersicolorMahal = distanceMahalanobis(iris[iris[SPECIES] == VIRGINICA],
                                                              iris[iris[SPECIES] == VERSICOLOR], versicolorCentre)

    # On prépare ici le tableau des distances euclidienne
    dataEuclid = [["À " + SETOSA, "À " + VERSICOLOR, "À " + VIRGINICA],
                  ["De " + SETOSA, round(intraClasseSetosaEuclid[1], 4), round(interClasseSetosaVersicolorEuclid[0], 4),
                   round(interClasseSetosaVirginicaEuclid[0], 4)],
                  ["De " + VERSICOLOR, round(interClasseVersicolorSetosaEuclid[0], 4),
                   round(intraClasseVersicolorEuclid[1], 4), round(interClasseVersicolorVirginicaEuclid[0], 4)],
                  ["De " + VIRGINICA, round(interClasseVirginicaSetosaEuclid[0], 4),
                   round(interClasseVirginicaVersicolorEuclid[0], 4), round(intraClasseVirginicaEuclid[1], 4)]]

    # On prépare ici le tableau des distances Mahalanobis
    dataMahal = [["À " + SETOSA, "À " + VERSICOLOR, "À " + VIRGINICA],
                 ["De " + SETOSA, round(intraClasseSetosaMahal[1], 4), round(interClasseSetosaVersicolorMahal[0], 4),
                  round(interClasseSetosaVirginicaMahal[0], 4)],
                 ["De " + VERSICOLOR, round(interClasseVersicolorSetosaMahal[0], 4),
                  round(intraClasseVersicolorMahal[1], 4), round(interClasseVersicolorVirginicaMahal[0], 4)],
                 ["De " + VIRGINICA, round(interClasseVirginicaSetosaMahal[0], 4),
                  round(interClasseVirginicaVersicolorMahal[0], 4), round(intraClasseVirginicaMahal[1], 4)]]

    # On sauvegarde les deux tableaux en format .png
    saveTable("Methode1_euclid_4var.png", "Méthode 1 : Distance euclidienne (4 variables sans transformation)", dataEuclid)
    saveTable("Methode1_mahal_4var.png", "Méthode 1 : Distance Mahalanobis (4 variables sans transformation)", dataMahal)

    return

def methodeDeuxA(iris):
    histogramme("Methode2a_VersiSetoPW.png", iris, VERSICOLOR, SETOSA, PETAL_WIDTH)
    # histogramme("Methode2a_VersiSetoPL.png", iris, VERSICOLOR, SETOSA, PETAL_LENGTH)
    # histogramme("Methode2a_VersiSetoSW.png", iris, VERSICOLOR, SETOSA, SEPAL_WIDTH)
    # histogramme("Methode2a_VersiSetoSL.png", iris, VERSICOLOR, SETOSA, SEPAL_LENGTH)
    # histogramme("Methode2a_VirgiSetoPW.png", iris, VIRGINICA, SETOSA, PETAL_WIDTH)
    histogramme("Methode2a_VirgiSetoPL.png", iris, VIRGINICA, SETOSA, PETAL_LENGTH)
    # histogramme("Methode2a_VirgiSetoSW.png", iris, VIRGINICA, SETOSA, SEPAL_WIDTH)
    # histogramme("Methode2a_VirgiSetoSL.png", iris, VIRGINICA, SETOSA, SEPAL_LENGTH)
    histogramme("Methode2a_VersiVirgiPW.png", iris, VERSICOLOR, VIRGINICA, PETAL_WIDTH)
    # histogramme("Methode2a_VersiVirgiPL.png", iris, VERSICOLOR, VIRGINICA, PETAL_LENGTH)
    # histogramme("Methode2a_VersiVirgiSW.png", iris, VERSICOLOR, VIRGINICA, SEPAL_WIDTH)
    # histogramme("Methode2a_VersiVirgiSL.png", iris, VERSICOLOR, VIRGINICA, SEPAL_LENGTH)

# CHANGER LES CONST POUR DES VAR!!!!!
def histogramme(nomFichier, iris, nomDataA, nomDataB, nomVariable):
    plot.clf()
    plot.figure(1)

    versicolor = iris[iris[SPECIES] == nomDataA]
    setosa = iris[iris[SPECIES] == nomDataB]
    plot.hist((versicolor[nomVariable], setosa[nomVariable]))
    plot.title("Variable " + nomVariable + " de " + nomDataA + " en comparaison à " + nomDataB)
    plot.ylabel("Fréquence")
    plot.xlabel(nomVariable)
    plot.savefig("output/" + nomFichier)


# CHANGER LES STRING POUR DES VAR
def nuagePoints(iris):
    plot.clf()
    plot.figure(1)

    setosa = iris[iris[SPECIES] == "setosa"]
    versicolor = iris[iris[SPECIES] == "versicolor"]
    plot.scatter(x=setosa[SEPAL_LENGTH], y=setosa["petal_length"])
    plot.scatter(x=versicolor[SEPAL_LENGTH], y=versicolor["petal_length"])
    plot.title("Longueurs des sépales comparément à la longueur des pétales")
    plot.ylabel("Longueur des pétales")
    plot.xlabel("Longueur des sépales")
    plot.savefig("output/2b.png")


def getEigenValues(iris):
    classe = iris[iris[SPECIES] == "setosa"]
    centreClasse = getCentreClasse(iris, "setosa")

    matriceCov = np.zeros((4, 4), dtype=float)
    centre = np.array(
        [centreClasse.sepal_length, centreClasse.sepal_width, centreClasse.petal_length, centreClasse.petal_width])
    for i in classe.index:
        currentPos = np.array(
            [classe["sepal_length"].get(i), classe["sepal_width"].get(i), classe["petal_length"].get(i),
             classe["petal_width"].get(i)])
        a = np.array(currentPos - centre)[np.newaxis]
        b = a * a.T
        matriceCov += b
    matriceCov = matriceCov / (len(classe.index) - 1)

    eigenResults = np.linalg.eig(matriceCov)
    eigenValues = eigenResults[0]
    eigenVectors = eigenResults[1]
    DimensionsImportance = (
        eigenValues[0] / sum(eigenValues), eigenValues[1] / sum(eigenValues), eigenValues[2] / sum(eigenValues),
        eigenValues[3] / sum(eigenValues))


def printDistance(nomClasse, tupleDistance, estIntra):
    indiceTuple = 0
    if estIntra:
        indiceTuple = 1
    print(nomClasse + " : " + str(tupleDistance[indiceTuple]))


# Notre code pour la génération de talbeau avec matplotlib est basé sur celui
# offert sur le site https://towardsdatascience.com/simple-little-tables-with-matplotlib-9780ef5d0bc4
def saveTable(nomFichier, titre, data):
    plot.clf()
    plot.figure(2)

    title_text = titre
    footer_text = '1er octobre 2021'
    fig_background_color = 'skyblue'
    fig_border = 'steelblue'

    # Pop the headers from the data array
    column_headers = data.pop(0)
    row_headers = [x.pop(0) for x in data]

    # Get some lists of color specs for row and column headers
    rcolors = plot.cm.BuPu(np.full(len(row_headers), 0.1))
    ccolors = plot.cm.BuPu(np.full(len(column_headers), 0.1))
    # Create the figure. Setting a small pad on tight_layout
    # seems to better regulate white space. Sometimes experimenting
    # with an explicit figsize here can produce better outcome.
    plot.figure(linewidth=2,
                edgecolor=fig_border,
                facecolor=fig_background_color,
                tight_layout={'pad': 1},
                figsize=(5, 2)
                )
    # Add a table at the bottom of the axes
    the_table = plot.table(cellText=data,
                           rowLabels=row_headers,
                           rowColours=rcolors,
                           rowLoc='right',
                           colColours=ccolors,
                           colLabels=column_headers,
                           loc='center')
    # Scaling is the only influence we have over top and bottom cell padding.
    # Make the rows taller (i.e., make cell y scale larger).
    the_table.scale(1, 1.5)
    # Hide axes
    ax = plot.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Hide axes border
    plot.box(on=None)
    # Add title
    plot.suptitle(title_text)
    # Add footer
    plot.figtext(0.95, 0.05, footer_text, horizontalalignment='right', size=6, weight='light')
    # Force the figure to update, so backends center objects correctly within the figure.
    # Without plt.draw() here, the title will center on the axes and not the figure.
    plot.draw()
    # Create image. plt.savefig ignores figure edge and face colors, so map them.
    fig = plot.gcf()
    plot.savefig("output/"+nomFichier,
                 bbox_inches='tight',
                 edgecolor=fig.get_edgecolor(),
                 facecolor=fig.get_facecolor(),
                 dpi=150
                 )


def main():
    iris = panda.read_csv("data/iris.csv")
    methodeUnA(iris)
    methodeDeuxA(iris)
    nuagePoints(iris)
    # normalizeIRIS(iris)
    getEigenValues(iris)


if __name__ == '__main__':
    main()
