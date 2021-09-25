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


# Methode 1a

# TODO: Rajouter des attributs avec la distance intra et interclasse
def distanceEuclide(classe, centreClasse):
    maxDistance = 0
    for i in classe.index:
        distanceSepalL = (classe["sepal_length"].get(i) - centreClasse.sepal_length)
        distanceSepalW = (classe["sepal_width"].get(i) - centreClasse.sepal_width)
        distancePetalL = (classe["petal_length"].get(i) - centreClasse.petal_length)
        distancePetalW = (classe["petal_width"].get(i) - centreClasse.petal_width)

        distance = np.sqrt(np.power(distanceSepalL, 2) + np.power(distanceSepalW, 2) + np.power(distancePetalL, 2) + np.power(distancePetalW, 2))
        if(distance > maxDistance):
            maxDistance = distance

    return maxDistance


def methodeUnA(iris):
    setosaCentre = getCentreClasse(iris, "setosa")
    versicolorCentre = getCentreClasse(iris, "versicolor")
    ### air_quality["intra_class_distance"] = (air_quality["station_paris"] / air_quality["station_antwerp"])

    intraClasseSetosa = distanceEuclide(iris[iris["species"] == "setosa"], setosaCentre, "setosa")
    # Interclasse de versicolor -> Setosa
    interClasseVersiSetosa = distanceEuclide(iris[iris["species"] == "versicolor"], setosaCentre, "versicolor")


    versicolor = iris[iris["species"] == "versicolor"]
    versicolor = versicolor[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    versicolor.plot.scatter(x="sepal_length",y="petal_length",alpha=0.5)
    print(versicolor)
    plot.savefig("output/test.pdf")
    # print(iris.to_string())
    return

def main():
    iris = panda.read_csv("data/iris.csv")
    methodeUnA(iris)


# Inspiré du tutoriel de plotting de Pandas:
# https://pandas.pydata.org/docs/getting_started/intro_tutorials/04_plotting.html

if __name__ == '__main__':
    main()
