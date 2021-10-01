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
    plot.hist((versicolor["petal_length"], setosa["petal_length"]))
    plot.title("Longueurs des pétales de versicolor en comparaison à setosa")
    plot.ylabel("Fréquence")
    plot.xlabel("Longueur des pétales")
    plot.savefig("output/2a.pdf")


def nuagePoints(iris):
    plot.figure(2)
    setosa = iris[iris["species"] == "setosa"]
    versicolor = iris[iris["species"] == "versicolor"]
    plot.scatter(x=setosa["sepal_length"], y=setosa["petal_length"])
    plot.scatter(x=versicolor["sepal_length"], y=versicolor["petal_length"])
    plot.title("Longueurs des sépales comparément à la longueur des pétales")
    plot.ylabel("Longueur des pétales")
    plot.xlabel("Longueur des sépales")
    plot.show()
    plot.savefig("output/2b.pdf")


def getEigenValues(iris):
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

def main():
    iris = panda.read_csv("data/iris.csv")
    methodeUnA(iris)
    # histogramme(iris)
    # nuagePoints(iris)
    normalizeIRIS(iris)
    getEigenValues(iris)


# Inspiré du tutoriel de plotting de Pandas:
# https://pandas.pydata.org/docs/getting_started/intro_tutorials/04_plotting.html

if __name__ == '__main__':
    main()
