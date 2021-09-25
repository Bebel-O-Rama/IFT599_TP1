import pandas as panda
import matplotlib.pyplot as plot
from dataclasses import dataclass

SEPAL_LENGTH = "sepal_length"
SEPAL_WIDTH = "sepal_width"
PETAL_LENGTH = "petal_length"
PETAL_WIDTH = "petal_width"


@dataclass
class IRISClass:
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    class_data: float
    species: str


def getDataClasse(dataset, nomClasse):
    classe = dataset[dataset["species"] == nomClasse]
    classe = classe[[SEPAL_LENGTH, SEPAL_WIDTH, PETAL_LENGTH, PETAL_WIDTH]]
    sepalLengthMean = round(classe[SEPAL_LENGTH].mean(), 1)
    sepalWidthMean = round(classe[SEPAL_WIDTH].mean(), 1)
    petalLengthMean = round(classe[PETAL_LENGTH].mean(), 1)
    petalWidthMean = round(classe[PETAL_WIDTH].mean(), 1)
    return IRISClass(sepalLengthMean, sepalWidthMean, petalLengthMean, petalWidthMean, classe, nomClasse)


# Methode 1a

# TODO: Rajouter des attributs avec la distance intra et interclasse
def methodeUnA(iris):
    setosa_data_centre = getDataClasse(iris, "setosa")
    versicolor_data_centre = getDataClasse(iris, "versicolor")
    virginica_data_centre = getDataClasse(iris, "virginica")

    setosa_data_centre = saveGraph(SEPAL_LENGTH, PETAL_LENGTH, "test1", "test_setosa", setosa_data_centre.class_data)
    versicolor_plot = saveGraph(SEPAL_LENGTH, PETAL_LENGTH, "test2", "test_versicolor", versicolor_data_centre.class_data)
    virginica_data_centre = saveGraph(SEPAL_LENGTH, PETAL_LENGTH, "test3", "test_virginica", virginica_data_centre.class_data)

    return


def saveGraph(x_axis, y_axis, plot_name, file_name, data):
    data.plot.scatter(x=x_axis, y=y_axis, alpha=0.5)
    plot.title(plot_name)
    plot.savefig("output/"+ file_name + ".pdf")
    return


def main():
    iris = panda.read_csv("data/iris.csv")
    methodeUnA(iris)


# Inspiré du tutoriel de plotting de Pandas:
# https://pandas.pydata.org/docs/getting_started/intro_tutorials/04_plotting.html

if __name__ == '__main__':
    main()
