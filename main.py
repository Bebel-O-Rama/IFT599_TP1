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


def methodeUn(iris):
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
    saveTable("Methode1_euclid_4var.png", "Méthode 1 : Distance euclidienne (4 variables sans transformation)",
              dataEuclid)
    saveTable("Methode1_mahal_4var.png", "Méthode 1 : Distance Mahalanobis (4 variables sans transformation)",
              dataMahal)

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


def methodeDeuxB(iris):
    normalizedIRIS = normalizeIRIS(iris)
    histogrammeNormalise("histo_norm_SetosaVersi.png", normalizedIRIS, SETOSA, VERSICOLOR)
    histogrammeNormalise("histo_norm_SetosaVirgi.png", normalizedIRIS, SETOSA, VIRGINICA)
    histogrammeNormalise("histo_norm_VersiVirgi.png", normalizedIRIS, VERSICOLOR, VIRGINICA)

    return


def methodeDeuxC(iris):
    nuagePoints("np_SetoVersi_PW_SL.png", iris, SETOSA, VERSICOLOR, PETAL_WIDTH, SEPAL_LENGTH)
    nuagePoints("np_SetoVirgi_PW_PL.png", iris, SETOSA, VIRGINICA, PETAL_WIDTH, PETAL_LENGTH)
    nuagePoints("np_VersiVirgi_SW_PW.png", iris, VERSICOLOR, VIRGINICA, SEPAL_WIDTH, PETAL_WIDTH)

    return


def methodeDeuxD(iris):
    normalizedIRIS = normalizeIRIS(iris)
    nuagePointsNormalise("np_norm_SetosaVersi.png", normalizedIRIS, SETOSA, VERSICOLOR)
    nuagePointsNormalise("np_norm_SetosaVirgi.png", normalizedIRIS, SETOSA, VIRGINICA)
    nuagePointsNormalise("np_norm_VersiVirgi.png", normalizedIRIS, VERSICOLOR, VIRGINICA)

    return


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


def histogrammeNormalise(nomFichier, irisNormalise, nomDataA, nomDataB):
    plot.clf()
    plot.figure(1)

    eigenVecA = getEigenValues(irisNormalise, nomDataA, nomDataB, 1)
    eigenVecB = getEigenValues(irisNormalise, nomDataB, nomDataB, 1)

    dataNormA = irisNormalise[irisNormalise[SPECIES] == nomDataA]
    dataNormA = dataNormA[[SEPAL_LENGTH, SEPAL_WIDTH, PETAL_LENGTH, PETAL_WIDTH]]
    dataNormA = np.matmul(eigenVecA, dataNormA.to_numpy().transpose())

    dataNormB = irisNormalise[irisNormalise[SPECIES] == nomDataB]
    dataNormB = dataNormB[[SEPAL_LENGTH, SEPAL_WIDTH, PETAL_LENGTH, PETAL_WIDTH]]
    dataNormB = np.matmul(eigenVecB, dataNormB.to_numpy().transpose())

    plot.hist((dataNormA[0], dataNormB[0]))
    plot.title("Séparation des classes " + nomDataA + " et de " + nomDataB)
    plot.ylabel("Fréquence")
    plot.xlabel("Valeur Z")
    plot.savefig("output/" + nomFichier)


def nuagePoints(nomFichier, iris, nomDataA, nomDataB, nomVariableA, nomVariableB):
    plot.clf()
    plot.figure(1)

    dataA = iris[iris[SPECIES] == nomDataA]
    dataB = iris[iris[SPECIES] == nomDataB]
    plot.scatter(x=dataA[nomVariableA], y=dataA[nomVariableB])
    plot.scatter(x=dataB[nomVariableA], y=dataB[nomVariableB])
    plot.title(
        "Classes " + nomDataA + " et " + nomDataB + " avec les variables " + nomVariableA + " et " + nomVariableB)
    plot.ylabel(nomVariableB)
    plot.xlabel(nomVariableA)
    plot.savefig("output/" + nomFichier)


def nuagePointsNormalise(nomFichier, irisNormalise, nomDataA, nomDataB):
    plot.clf()
    plot.figure(1)

    eigenVecA = getEigenValues(irisNormalise, nomDataA, nomDataB, 2)
    eigenVecB = getEigenValues(irisNormalise, nomDataB, nomDataB, 2)

    dataNormA = irisNormalise[irisNormalise[SPECIES] == nomDataA]
    dataNormA = dataNormA[[SEPAL_LENGTH, SEPAL_WIDTH, PETAL_LENGTH, PETAL_WIDTH]]
    dataNormA = np.matmul(eigenVecA, dataNormA.to_numpy().transpose())

    dataNormB = irisNormalise[irisNormalise[SPECIES] == nomDataB]
    dataNormB = dataNormB[[SEPAL_LENGTH, SEPAL_WIDTH, PETAL_LENGTH, PETAL_WIDTH]]
    dataNormB = np.matmul(eigenVecB, dataNormB.to_numpy().transpose())

    plot.scatter(x=dataNormA[0], y=dataNormA[1])
    plot.scatter(x=dataNormB[0], y=dataNormB[1])
    plot.title("Séparation des classes " + nomDataA + " et " + nomDataB)
    plot.savefig("output/" + nomFichier)


def getEigenValues(iris, dataNameA, dataNameB, nbAxis):
    classe = iris[iris[SPECIES] == dataNameA]
    centreClasse = getCentreClasse(iris, dataNameB)

    matriceCov = np.zeros((4, 4), dtype=float)
    centre = np.array(
        [centreClasse.sepal_length, centreClasse.sepal_width, centreClasse.petal_length, centreClasse.petal_width])
    for i in classe.index:
        currentPos = np.array(
            [classe[SEPAL_LENGTH].get(i), classe[SEPAL_WIDTH].get(i), classe[PETAL_LENGTH].get(i),
             classe[PETAL_WIDTH].get(i)])
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
                        ["sepal_length"]] - round(iris["sepal_length"].mean(), 1)) / round(iris["sepal_length"].std(),
                                                                                           1)
    setosaSepalW = (iris[iris["species"] == "setosa"][
                        ["sepal_width"]] - round(iris["sepal_width"].mean(), 1)) / round(iris["sepal_width"].std(), 1)
    setosaPetalL = (iris[iris["species"] == "setosa"][
                        ["petal_length"]] - round(iris["petal_length"].mean(), 1)) / round(iris["petal_length"].std(),
                                                                                           1)
    setosaPetalW = (iris[iris["species"] == "setosa"][
                        ["petal_width"]] - round(iris["petal_width"].mean(), 1)) / round(iris["petal_width"].std(), 1)

    versicolorSepalL = (iris[iris["species"] == "versicolor"][
                            ["sepal_length"]] - round(iris["sepal_length"].mean(), 1)) / round(
        iris["sepal_length"].std(), 1)
    versicolorSepalW = (iris[iris["species"] == "versicolor"][
                            ["sepal_width"]] - round(iris["sepal_width"].mean(), 1)) / round(iris["sepal_width"].std(),
                                                                                             1)
    versicolorPetalL = (iris[iris["species"] == "versicolor"][
                            ["petal_length"]] - round(iris["petal_length"].mean(), 1)) / round(
        iris["petal_length"].std(), 1)
    versicolorPetalW = (iris[iris["species"] == "versicolor"][
                            ["petal_width"]] - round(iris["petal_width"].mean(), 1)) / round(iris["petal_width"].std(),
                                                                                             1)

    virginicaSepalL = (iris[iris["species"] == "virginica"][
                           ["sepal_length"]] - round(iris["sepal_length"].mean(), 1)) / round(
        iris["sepal_length"].std(), 1)
    virginicaSepalW = (iris[iris["species"] == "virginica"][
                           ["sepal_width"]] - round(iris["sepal_width"].mean(), 1)) / round(iris["sepal_width"].std(),
                                                                                            1)
    virginicaPetalL = (iris[iris["species"] == "virginica"][
                           ["petal_length"]] - round(iris["petal_length"].mean(), 1)) / round(
        iris["petal_length"].std(), 1)
    virginicaPetalW = (iris[iris["species"] == "virginica"][
                           ["petal_width"]] - round(iris["petal_width"].mean(), 1)) / round(iris["petal_width"].std(),
                                                                                            1)

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
    plot.savefig("output/" + nomFichier,
                 bbox_inches='tight',
                 edgecolor=fig.get_edgecolor(),
                 facecolor=fig.get_facecolor(),
                 dpi=150
                 )


def main():
    # Ramasse les données du dataset Iris
    iris = panda.read_csv("data/iris.csv")

    # Méthode 1 (1.a + 1.b)
    methodeUn(iris)

    # Méthode 2a (2.a)
    methodeDeuxA(iris)

    # Méthode 2b (2.a + 2.c)
    methodeDeuxB(iris)

    # Méthode 2c (2.b)
    methodeDeuxC(iris)

    # Méthode 2d (2.b + 2.c)
    methodeDeuxD(iris)


if __name__ == '__main__':
    main()
