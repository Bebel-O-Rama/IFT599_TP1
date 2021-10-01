import numpy as np
import pandas as panda
import matplotlib.pyplot as plot
from dataclasses import dataclass

SETOSA = "setosa"
VERSICOLOR = "versicolor"
VIRGINICA = "virginica"


@dataclass
class IRISClass:
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    species: str


def getCentreClasse(dataset, nomClasse):
    classe = dataset[dataset["species"] == nomClasse]
    classe = classe[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    sepalLengthMean = round(classe["sepal_length"].mean(), 1)
    sepalWidthMean = round(classe["sepal_width"].mean(), 1)
    petalLengthMean = round(classe["petal_length"].mean(), 1)
    petalWidthMean = round(classe["petal_width"].mean(), 1)
    return IRISClass(sepalLengthMean, sepalWidthMean, petalLengthMean, petalWidthMean, nomClasse)


def getEcartTypeClasse(dataset, nomClasse):
    classe = dataset[dataset["species"] == nomClasse]
    classe = classe[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    sepalLengthStd = round(classe["sepal_length"].std(), 1)
    sepalWidthStd = round(classe["sepal_width"].std(), 1)
    petalLengthStd = round(classe["petal_length"].std(), 1)
    petalWidthStd = round(classe["petal_width"].std(), 1)
    return IRISClass(sepalLengthStd, sepalWidthStd, petalLengthStd, petalWidthStd, nomClasse)


# Methode 1a

# TODO: Rajouter des attributs avec la distance intra et interclasse
def distanceEuclide(classe, centreClasse):
    minDistance = float('inf')
    maxDistance = 0
    for i in classe.index:
        distanceSepalL = (classe["sepal_length"].get(i) - centreClasse.sepal_length)
        distanceSepalW = (classe["sepal_width"].get(i) - centreClasse.sepal_width)
        distancePetalL = (classe["petal_length"].get(i) - centreClasse.petal_length)
        distancePetalW = (classe["petal_width"].get(i) - centreClasse.petal_width)

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
            [classeB["sepal_length"].get(i), classeB["sepal_width"].get(i), classeB["petal_length"].get(i),
             classeB["petal_width"].get(i)])
        a = np.array(currentPos - centre)[np.newaxis]
        b = a * a.T
        matriceCov += b
    matriceCov = matriceCov / (len(classeB.index) - 1)
    matriceCov = np.linalg.inv(matriceCov)
    for i in classeA.index:
        currentPos = np.array(
            [classeA["sepal_length"].get(i), classeA["sepal_width"].get(i), classeA["petal_length"].get(i),
             classeA["petal_width"].get(i)])
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
    intraClasseSetosaEuclid = distanceEuclide(iris[iris["species"] == SETOSA], setosaCentre)
    intraClasseVersicolorEuclid = distanceEuclide(iris[iris["species"] == VERSICOLOR], versicolorCentre)
    intraClasseVirginicaEuclid = distanceEuclide(iris[iris["species"] == VIRGINICA], virginicaCentre)

    # Interclasse Setosa
    interClasseSetosaVersicolorEuclid = distanceEuclide(iris[iris["species"] == SETOSA], versicolorCentre)
    interClasseSetosaVirginicaEuclid = distanceEuclide(iris[iris["species"] == SETOSA], virginicaCentre)

    # Interclasse Versicolor
    interClasseVersicolorSetosaEuclid = distanceEuclide(iris[iris["species"] == VERSICOLOR], setosaCentre)
    interClasseVersicolorVirginicaEuclid = distanceEuclide(iris[iris["species"] == VERSICOLOR], virginicaCentre)

    # Interclasse Virginica
    interClasseVirginicaSetosaEuclid = distanceEuclide(iris[iris["species"] == VIRGINICA], setosaCentre)
    interClasseVirginicaVersicolorEuclid = distanceEuclide(iris[iris["species"] == VIRGINICA], versicolorCentre)

    # On va trouver les distances Mahalanobis

    # Intraclasse
    intraClasseSetosaMahal = distanceMahalanobis(iris[iris["species"] == SETOSA], iris[iris["species"] == SETOSA],
                                                 setosaCentre)
    intraClasseVersicolorMahal = distanceMahalanobis(iris[iris["species"] == VERSICOLOR],
                                                     iris[iris["species"] == VERSICOLOR], versicolorCentre)
    intraClasseVirginicaMahal = distanceMahalanobis(iris[iris["species"] == VIRGINICA],
                                                    iris[iris["species"] == VIRGINICA], virginicaCentre)

    # Interclasse Setosa
    interClasseSetosaVersicolorMahal = distanceMahalanobis(iris[iris["species"] == SETOSA],
                                                           iris[iris["species"] == VERSICOLOR], versicolorCentre)
    interClasseSetosaVirginicaMahal = distanceMahalanobis(iris[iris["species"] == SETOSA],
                                                          iris[iris["species"] == VIRGINICA], virginicaCentre)

    # Interclasse Versicolor
    interClasseVersicolorSetosaMahal = distanceMahalanobis(iris[iris["species"] == VERSICOLOR],
                                                           iris[iris["species"] == SETOSA], setosaCentre)
    interClasseVersicolorVirginicaMahal = distanceMahalanobis(iris[iris["species"] == VERSICOLOR],
                                                              iris[iris["species"] == VIRGINICA], virginicaCentre)

    # Interclasse Virginica
    interClasseVirginicaSetosaMahal = distanceMahalanobis(iris[iris["species"] == VIRGINICA],
                                                          iris[iris["species"] == SETOSA], setosaCentre)
    interClasseVirginicaVersicolorMahal = distanceMahalanobis(iris[iris["species"] == VIRGINICA],
                                                              iris[iris["species"] == VERSICOLOR], versicolorCentre)

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
    saveTable("Methode1_euclid1.png", "Méthode 1 : Distance euclidienne avec les 4 variables", dataEuclid)
    saveTable("Methode1_mahal1.png", "Méthode 1 : Distance Mahalanobis avec les 4 variables", dataMahal)

    # # À titre de test, on print les résultats des distances euclidiennes et Mahalanobis
    # print("--- DISTANCE EUCLIDIENNE INTRACLASSE ---")
    # printDistance(SETOSA, intraClasseSetosaEuclid, True)
    # printDistance(VERSICOLOR, intraClasseVersicolorEuclid, True)
    # printDistance(VIRGINICA, intraClasseVirginicaEuclid, True)
    # print("--- DISTANCE EUCLIDIENNE INTERCLASSE SETOSA ---")
    # printDistance(VERSICOLOR, interClasseSetosaVersicolorEuclid, False)
    # printDistance(VIRGINICA, interClasseSetosaVirginicaEuclid, False)
    # print("--- DISTANCE EUCLIDIENNE INTERCLASSE VERSICOLOR ---")
    # printDistance(SETOSA, interClasseVersicolorSetosaEuclid, False)
    # printDistance(VIRGINICA, interClasseVersicolorVirginicaEuclid, False)
    # print("--- DISTANCE EUCLIDIENNE INTERCLASSE VIRGINICA ---")
    # printDistance(SETOSA, interClasseVirginicaSetosaEuclid, False)
    # printDistance(VERSICOLOR, interClasseVirginicaVersicolorEuclid, False)
    # print("\n--- DISTANCE MAHALANOBIS INTRACLASSE ---")
    # printDistance(SETOSA, intraClasseSetosaMahal, True)
    # printDistance(VERSICOLOR, intraClasseVersicolorMahal, True)
    # printDistance(VIRGINICA, intraClasseVirginicaMahal, True)
    # print("--- DISTANCE MAHALANOBIS INTERCLASSE SETOSA ---")
    # printDistance(VERSICOLOR, interClasseSetosaVersicolorMahal, False)
    # printDistance(VIRGINICA, interClasseSetosaVirginicaMahal, False)
    # print("--- DISTANCE MAHALANOBIS INTERCLASSE VERSICOLOR ---")
    # printDistance(SETOSA, interClasseVersicolorSetosaMahal, False)
    # printDistance(VIRGINICA, interClasseVersicolorVirginicaMahal, False)
    # print("--- DISTANCE MAHALANOBIS INTERCLASSE VIRGINICA ---")
    # printDistance(SETOSA, interClasseVirginicaSetosaMahal, False)
    # printDistance(VERSICOLOR, interClasseVirginicaVersicolorMahal, False)

    return


def histogramme(iris):
    plot.figure(1)
    versicolor = iris[iris["species"] == "versicolor"]
    setosa = iris[iris["species"] == "setosa"]
    plot.hist((versicolor["sepal_length"], setosa["sepal_length"]))
    plot.title("Longueurs des sépales de versicolor en comparaison à setosa")
    plot.ylabel("Fréquence")
    plot.xlabel("Longueur des sépales")
    plot.savefig("output/2a.png")


def histogrammeNormalise(irisNormalise, eigenVector):
    plot.figure(5)

    setosaNorm = irisNormalise[irisNormalise["species"] == "setosa"]
    setosaNorm = setosaNorm[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    setosaNormalisees = np.matmul(eigenVector, setosaNorm.to_numpy().transpose())

    versicolorNorm = irisNormalise[irisNormalise["species"] == "versicolor"]
    versicolorNorm = versicolorNorm[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    versicolorNormalisees = np.matmul(eigenVector, versicolorNorm.to_numpy().transpose())

    virginicaNorm = irisNormalise[irisNormalise["species"] == "virginica"]
    virginicaNorm = virginicaNorm[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    virginicaNormalisees = np.matmul(eigenVector, virginicaNorm.to_numpy().transpose())

    plot.hist((virginicaNormalisees[0], versicolorNormalisees[0]))
    plot.title("Séparation des classes de la banque de données IRIS")
    plot.ylabel("Fréquence")
    plot.xlabel("Valeur Z")
    plot.savefig("output/2ac.png")


def nuagePoints(iris):
    plot.figure(2)
    setosa = iris[iris["species"] == "setosa"]
    versicolor = iris[iris["species"] == "versicolor"]
    plot.scatter(x=setosa["sepal_length"], y=setosa["petal_length"])
    plot.scatter(x=versicolor["sepal_length"], y=versicolor["petal_length"])
    plot.title("Longueurs des sépales comparément à la longueur des pétales")
    plot.ylabel("Longueur des pétales")
    plot.xlabel("Longueur des sépales")


def nuagePointsNormalise(irisNormalise, eigenVectors):
    plot.figure(3)
    setosaNorm = irisNormalise[irisNormalise["species"] == "setosa"]
    setosaNorm = setosaNorm[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    setosaNormalisees = np.matmul(eigenVectors, setosaNorm.to_numpy().transpose())

    versicolorNorm = irisNormalise[irisNormalise["species"] == "versicolor"]
    versicolorNorm = versicolorNorm[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    versicolorNormalisees = np.matmul(eigenVectors, versicolorNorm.to_numpy().transpose())

    virginicaNorm = irisNormalise[irisNormalise["species"] == "virginica"]
    virginicaNorm = virginicaNorm[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    virginicaNormalisees = np.matmul(eigenVectors, virginicaNorm.to_numpy().transpose())

    plot.scatter(x=setosaNormalisees[0], y=setosaNormalisees[1])
    plot.scatter(x=versicolorNormalisees[0], y=versicolorNormalisees[1])
    plot.scatter(x=virginicaNormalisees[0], y=virginicaNormalisees[1])
    plot.title("Séparation des différentes classes de la banque de données IRIS")
    plot.savefig("output/2bc.png")
    plot.show()


def getEigenValues(iris, nbAxis):
    classe = iris[iris["species"] == "setosa"]
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

    # from https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
    mostImportantColumns = np.argpartition(DimensionsImportance, -nbAxis)[-nbAxis:]
    mostImportantColumns = np.sort(mostImportantColumns)

    return eigenVectors[mostImportantColumns]


def normalizeIRIS(iris):
    centreSetosa = getCentreClasse(iris, "setosa")
    centreVersicolor = getCentreClasse(iris, "versicolor")
    centreVirginica = getCentreClasse(iris, "virginica")
    ecartTypeSetosa = getEcartTypeClasse(iris, "setosa")
    ecartTypeVersicolor = getEcartTypeClasse(iris, "versicolor")
    ecartTypeVirginica = getEcartTypeClasse(iris, "virginica")

    setosaSepalL = (iris[iris["species"] == "setosa"][
                        ["sepal_length"]] - round(iris["sepal_length"].mean(), 1)) / round(iris["sepal_length"].std(), 1)
    setosaSepalW = (iris[iris["species"] == "setosa"][
                        ["sepal_width"]] - round(iris["sepal_width"].mean(), 1)) / round(iris["sepal_width"].std(), 1)
    setosaPetalL = (iris[iris["species"] == "setosa"][
                        ["petal_length"]] - round(iris["petal_length"].mean(), 1)) / round(iris["petal_length"].std(), 1)
    setosaPetalW = (iris[iris["species"] == "setosa"][
                        ["petal_width"]] - round(iris["petal_width"].mean(), 1)) / round(iris["petal_width"].std(), 1)

    versicolorSepalL = (iris[iris["species"] == "versicolor"][
                            ["sepal_length"]] - round(iris["sepal_length"].mean(), 1)) / round(iris["sepal_length"].std(), 1)
    versicolorSepalW = (iris[iris["species"] == "versicolor"][
                            ["sepal_width"]] - round(iris["sepal_width"].mean(), 1)) / round(iris["sepal_width"].std(), 1)
    versicolorPetalL = (iris[iris["species"] == "versicolor"][
                            ["petal_length"]] - round(iris["petal_length"].mean(), 1)) / round(iris["petal_length"].std(), 1)
    versicolorPetalW = (iris[iris["species"] == "versicolor"][
                            ["petal_width"]] - round(iris["petal_width"].mean(), 1)) / round(iris["petal_width"].std(), 1)

    virginicaSepalL = (iris[iris["species"] == "virginica"][
                           ["sepal_length"]] - round(iris["sepal_length"].mean(), 1)) / round(iris["sepal_length"].std(), 1)
    virginicaSepalW = (iris[iris["species"] == "virginica"][
                           ["sepal_width"]] - round(iris["sepal_width"].mean(), 1)) / round(iris["sepal_width"].std(), 1)
    virginicaPetalL = (iris[iris["species"] == "virginica"][
                           ["petal_length"]] - round(iris["petal_length"].mean(), 1)) / round(iris["petal_length"].std(), 1)
    virginicaPetalW = (iris[iris["species"] == "virginica"][
                           ["petal_width"]] - round(iris["petal_width"].mean(), 1)) / round(iris["petal_width"].std(), 1)

    sepalLength = panda.concat([setosaSepalL, versicolorSepalL, virginicaSepalL])
    sepalWidth = panda.concat([setosaSepalW, versicolorSepalW, virginicaSepalW])
    petalLength = panda.concat([setosaPetalL, versicolorPetalL, virginicaPetalL])
    petalWidth = panda.concat([setosaPetalW, versicolorPetalW, virginicaPetalW])

    normalizedDataframe = panda.DataFrame()

    # Order is reversed to keep the dataframe in the same order as the original
    normalizedDataframe.insert(0, 'species', iris["species"].squeeze(axis=0))
    normalizedDataframe.insert(0, 'petal_width', petalWidth.squeeze(axis=0))
    normalizedDataframe.insert(0, 'petal_length', petalLength.squeeze(axis=0))
    normalizedDataframe.insert(0, 'sepal_width', sepalWidth.squeeze(axis=0))
    normalizedDataframe.insert(0, 'sepal_length', sepalLength.squeeze(axis=0))

    return normalizedDataframe


def printDistance(nomClasse, tupleDistance, estIntra):
    indiceTuple = 0
    if estIntra:
        indiceTuple = 1
    print(nomClasse + " : " + str(tupleDistance[indiceTuple]))


# Notre code pour la génération de talbeau avec matplotlib est basé sur celui
# offert sur le site https://towardsdatascience.com/simple-little-tables-with-matplotlib-9780ef5d0bc4
def saveTable(nomFichier, titre, data):
    plot.figure(9)
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
    histogramme(iris)
    nuagePoints(iris)
    normalizedIRIS = normalizeIRIS(iris)
    histogrammeNormalise(normalizedIRIS, getEigenValues(normalizedIRIS, 1))
    nuagePointsNormalise(normalizedIRIS, getEigenValues(normalizedIRIS, 2))


if __name__ == '__main__':
    main()
