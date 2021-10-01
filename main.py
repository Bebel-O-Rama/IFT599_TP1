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
    setosaCentre = getCentreClasse(iris, SETOSA)
    versicolorCentre = getCentreClasse(iris, VERSICOLOR)
    virginicaCentre = getCentreClasse(iris, VIRGINICA)

    # On commence par trouver les distances euclidienne intraclasse et interclasse pour nos trois classes

    # Intraclasse
    intraClasseSetosa = distanceEuclide(iris[iris["species"] == SETOSA], setosaCentre)
    intraClasseVersicolor = distanceEuclide(iris[iris["species"] == VERSICOLOR], versicolorCentre)
    intraClasseVirginica = distanceEuclide(iris[iris["species"] == VIRGINICA], virginicaCentre)

    # Interclasse Setosa
    interClasseSetosaVersicolor = distanceEuclide(iris[iris["species"] == SETOSA], versicolorCentre)
    interClasseSetosaVirginica = distanceEuclide(iris[iris["species"] == SETOSA], virginicaCentre)

    # Interclasse Versicolor
    interClasseVersicolorSetosa = distanceEuclide(iris[iris["species"] == VERSICOLOR], setosaCentre)
    interClasseVersicolorVirginica = distanceEuclide(iris[iris["species"] == VERSICOLOR], virginicaCentre)

    # Interclasse Virginica
    interClasseVirginicaSetosa = distanceEuclide(iris[iris["species"] == VIRGINICA], setosaCentre)
    interClasseVirginicaVersicolor = distanceEuclide(iris[iris["species"] == VIRGINICA], versicolorCentre)

    print("--- DISTANCE INTRACLASSE ---")
    printDistance(SETOSA, intraClasseSetosa, True)
    printDistance(VERSICOLOR, intraClasseVersicolor, True)
    printDistance(VIRGINICA, intraClasseVirginica, True)

    print("--- DISTANCE INTERCLASSE SETOSA ---")
    printDistance(VERSICOLOR, interClasseSetosaVersicolor, False)
    printDistance(VIRGINICA, interClasseSetosaVirginica, False)

    print("--- DISTANCE INTERCLASSE VERSICOLOR ---")
    printDistance(SETOSA, interClasseVersicolorSetosa, False)
    printDistance(VIRGINICA, interClasseVersicolorVirginica, False)

    print("--- DISTANCE INTERCLASSE VIRGINICA ---")
    printDistance(SETOSA, interClasseVirginicaSetosa, False)
    printDistance(VERSICOLOR, interClasseVirginicaVersicolor, False)

    # distanceMahalanobis(iris[iris["species"] == "setosa"], setosaCentre)
    # versicolor = iris[iris["species"] == "versicolor"]
    # versicolor = versicolor[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    # versicolor.plot.scatter(x="sepal_length", y="petal_length", alpha=0.5)
    # # print(versicolor)
    # plot.savefig("output/test.pdf")
    # print(iris.to_string())
    return

def printDistance(nomClasse, tupleDistance, estIntra):
    indiceTuple = 0
    if estIntra:
        indiceTuple = 1
    print(nomClasse + " : " + str(tupleDistance[indiceTuple]))

def main():
    iris = panda.read_csv("data/iris.csv")
    methodeUnA(iris)


# Inspiré du tutoriel de plotting de Pandas:
# https://pandas.pydata.org/docs/getting_started/intro_tutorials/04_plotting.html

if __name__ == '__main__':
    main()
