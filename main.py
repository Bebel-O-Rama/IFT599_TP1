import numpy as np
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

    # setosa_data_plot = saveGraph(SEPAL_LENGTH, PETAL_LENGTH, "test1", "test_setosa", setosa_data_centre.class_data)
    # versicolor_data_plot = saveGraph(SEPAL_LENGTH, PETAL_LENGTH, "test2", "test_versicolor", versicolor_data_centre.class_data)
    # virginica_data_plot = saveGraph(SEPAL_LENGTH, PETAL_LENGTH, "test3", "test_virginica", virginica_data_centre.class_data)

    saveTableDistance(setosa_data_centre, versicolor_data_centre, virginica_data_centre)

    return


def saveTableDistance(setosa_data, versicolor_data, virginica_data):
    setosa_coordinates = setosa_data.class_data
    versicolor_coordinates = versicolor_data.class_data
    virginica_coordinates = virginica_data.class_data

    # newarray = np.apply_along_axis(lambda setosa_coordinates: setosa_coordinates[0] + '\t' + setosa_coordinates[1], 1)
    # for x in setosa_coordinates
    df = panda.DataFrame()
    # print(setosa_coordinates.iloc[0,0])
    # print(setosa_coordinates)
    setosa_list_coord = list()
    versicolor_list_coord = list()
    virginica_list_coord = list()
    setosa_list_euclid = list()
    versicolor_list_euclid = list()
    virginica_list_euclid = list()

    data = list()
    for index in range(49):
        setosa_pos = "#" + str(index + 1) + " à (" + str(
            setosa_coordinates.iloc[index, 0]) + ", " + str(
            setosa_coordinates.iloc[index, 1]) + ", " + str(setosa_coordinates.iloc[index, 2]) + ", " + str(
            setosa_coordinates.iloc[index, 3]) + ")"
        versicolor_pos = "#" + str(index + 1) + " à (" + str(
            versicolor_coordinates.iloc[index, 0]) + ", " + str(
            versicolor_coordinates.iloc[index, 1]) + ", " + str(versicolor_coordinates.iloc[index, 2]) + ", " + str(
            versicolor_coordinates.iloc[index, 3]) + ")"
        virginica_pos = "#" + str(index + 1) + " à (" + str(
            virginica_coordinates.iloc[index, 0]) + ", " + str(
            virginica_coordinates.iloc[index, 1]) + ", " + str(virginica_coordinates.iloc[index, 2]) + ", " + str(
            virginica_coordinates.iloc[index, 3]) + ")"

        data.append([setosa_pos, "missing euclid", "missing mahan", "missing euclid_versi", "missing mahan_versi",
                     "missing euclid_vergi", "missing mahan_vergi", versicolor_pos, "missing euclid", "missing mahan",
                     "missing euclid_setosa", "missing mahan_setosa", "missing euclid_vergi", "missing mahan_vergi",
                     virginica_pos, "missing euclid", "missing mahan", "missing euclid_setosa", "missing mahan_setosa",
                     "missing euclid_versi", "missing mahan_versi"])

    col_label = ["Point et position pour Setosa", "Distance Euclidienne", "Distance Mahalanobis",
                 "Distance Euclidienne de Versicolor", "Distance Mahalanobis de Versicolor",
                 "Distance Euclidienne de Virginica", "Distance Mahalanobis de Virginica",
                 "Point et position pour Versicolor", "Distance Euclidienne", "Distance Mahalanobis",
                 "Distance Euclidienne de Setosa", "Distance Mahalanobis de Setosa",
                 "Distance Euclidienne de Virginica",
                 "Distance Mahalanobis de Virginica", "Point et position pour Versicolor", "Distance Euclidienne",
                 "Distance Mahalanobis", "Distance Euclidienne de Setosa", "Distance Mahalanobis de Setosa",
                 "Distance Euclidienne de Versicolor", "Distance Mahalanobis de Versicolor"]
    nrows, ncols = len(data)+1, len(col_label)
    hcell, wcell = 0.1, 0.1
    hpad, wpad = 0.5, 0.5
    fig = plot.figure(figsize=(ncols * wcell + wpad, nrows * hcell + hpad))
    ax = fig.add_subplot(111)
    ax.axis('off')

    the_table = ax.table(cellText=data,
                         colLabels=col_label,
                         loc='center')
    plot.savefig('gugusse.png', dpi=1000)
        # setosa_list_coord.append(
        #     "Point #" + str(index + 1) + " at position (" + str(setosa_coordinates.iloc[index, 0]) + ", " + str(
        #         setosa_coordinates.iloc[index, 1]) + ", " + str(setosa_coordinates.iloc[index, 2]) + ", " + str(
        #         setosa_coordinates.iloc[index, 3]) + ")")
        # versicolor_list_coord.append(
        #     "Point #" + str(index + 1) + " at position (" + str(versicolor_coordinates.iloc[index, 0]) + ", " + str(
        #         versicolor_coordinates.iloc[index, 1]) + ", " + str(versicolor_coordinates.iloc[index, 2]) + ", " + str(
        #         versicolor_coordinates.iloc[index, 3]) + ")")
        # virginica_list_coord.append(
        #     "Point #" + str(index + 1) + " at position (" + str(virginica_coordinates.iloc[index, 0]) + ", " + str(
        #         virginica_coordinates.iloc[index, 1]) + ", " + str(virginica_coordinates.iloc[index, 2]) + ", " + str(
        #         virginica_coordinates.iloc[index, 3]) + ")")
        #
        # data.append()
        # df["Point et coordonnées"] = setosa_list_coord
        # # df["Distance Euclidienne"] = setosa_list_euclid
        # df["Point et coordonnées"] = versicolor_list_coord
        # # df["Distance Euclidienne"] = versicolor_list_euclid
        # df["Point et coordonnées"] = virginica_list_coord
        # # df["Distance Euclidienne"] = virginica_list_euclid

        # Inspiré du post de stack https://stackoverflow.com/questions/52632356/setting-row-edge-color-of-matplotlib-table

        # size = (np.array(df.shape[::-1]) + np.array([0, 1])) * np.array([3.0, 0.625])
        # fig, ax = plot.subplots()
        # fig.patch.set_visible(False)
        # ax.axis('off')
        # ax.axis('tight')
        # plot.auto_set_font_size(False)
        # plot.set_fontsize(24)

        # ax.table(cellText=df.values, colLabels=df.columns, loc='center')
        # fig.tight_layout()
        # mpl_table = ax.table(cellText=df.values, colLabels=df.columns)

        # plot.savefig("output/testos.pdf")
    print(data)


def saveGraph(x_axis, y_axis, plot_name, file_name, data):
    data.plot.scatter(x=x_axis, y=y_axis, alpha=0.5)
    plot.title(plot_name)
    plot.savefig("output/" + file_name + ".pdf")
    return


def main():
    iris = panda.read_csv("data/iris.csv")
    methodeUnA(iris)


# Inspiré du tutoriel de plotting de Pandas:
# https://pandas.pydata.org/docs/getting_started/intro_tutorials/04_plotting.html

if __name__ == '__main__':
    main()
