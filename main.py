import numpy as np
import pandas as panda
import matplotlib.pyplot as plot
from dataclasses import dataclass


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

    return maxDistance


def distanceMahalanobis(classe, centreClasse):
    minDistance = float('inf')
    minDistanceIndexPoint = -1
    maxDistance = 0
    maxDistanceIndexPoint = -1
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

    matriceCov = np.linalg.inv(matriceCov)
    for i in classe.index:
        currentPos = np.array(
            [classe["sepal_length"].get(i), classe["sepal_width"].get(i), classe["petal_length"].get(i),
             classe["petal_width"].get(i)])
        a = np.array(currentPos - centre)[np.newaxis]
        b = a.dot(matriceCov)
        c = b.dot(a.T)
        distance = np.sqrt(c)
        print(distance)
        if (distance > maxDistance):
            maxDistance = distance
            maxDistanceIndexPoint = i
        if (distance < minDistance):
            minDistance = distance
            minDistanceIndexPoint = i
    print(maxDistanceIndexPoint)
    print(maxDistance)
    print(minDistanceIndexPoint)
    print(minDistance)


def methodeUnA(iris):
    setosaCentre = getCentreClasse(iris, "setosa")
    versicolorCentre = getCentreClasse(iris, "versicolor")
    ### air_quality["intra_class_distance"] = (air_quality["station_paris"] / air_quality["station_antwerp"])

    intraClasseSetosa = distanceEuclide(iris[iris["species"] == "setosa"], setosaCentre)
    # Interclasse de versicolor -> Setosa
    interClasseVersiSetosa = distanceEuclide(iris[iris["species"] == "versicolor"], setosaCentre)
    distanceMahalanobis(iris[iris["species"] == "setosa"], setosaCentre)
    versicolor = iris[iris["species"] == "versicolor"]
    versicolor = versicolor[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    # print(versicolor)
    plot.savefig("output/test.pdf")
    # print(iris.to_string())
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
    mostImportantColumns = np.argpartition(DimensionsImportance, -NB_AXIS)[-NB_AXIS:]
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


def main():
    iris = panda.read_csv("data/iris.csv")
    methodeUnA(iris)
    # histogramme(iris)
    # nuagePoints(iris)
    getEigenValues(iris)
    normalizedIRIS = normalizeIRIS(iris)
    histogrammeNormalise(normalizedIRIS, getEigenValues(normalizedIRIS, 1))
    nuagePointsNormalise(normalizedIRIS, getEigenValues(normalizedIRIS, 2))


# Inspiré du tutoriel de plotting de Pandas:
# https://pandas.pydata.org/docs/getting_started/intro_tutorials/04_plotting.html

if __name__ == '__main__':
    main()
