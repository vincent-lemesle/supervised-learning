from csv import reader
import math
import numpy as np
from graphviz import Graph
import matplotlib.pyplot as plt


def compute_dissimilarity(f, s):
    if int(f[0]) == int(s[0]):
        dissimilarity_meal = 0
    else:
        dissimilarity_meal = 15
    dissimilarity = math.sqrt(
        (float(f[1]) - float(s[1])) ** 2
        + 3 * (float(f[2]) - float(s[2])) ** 2
        + dissimilarity_meal
    )
    print(
        f"plyr 1 {int(f[0])}, plyr 2 {int(s[0])}, dissimilarity: {dissimilarity}"
    )
    return dissimilarity


with open('dataset.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    data = []
    for row in csv_reader:
        data.append(row)
    data = data[1:]
    f = data[0]
    s = data[1]
    nb_person = len(data)
    dissimilarity_matrix = np.zeros((nb_person, nb_person))
    print("compute dissimilarities")
    for person_1_id in range(nb_person):
        for person_2_id in range(nb_person):
            dissimilarity = compute_dissimilarity(data[person_1_id], data[person_2_id])
            dissimilarity_matrix[person_1_id, person_2_id] = dissimilarity
    print(dissimilarity_matrix)
    threshold = 15
    # build a graph from the dissimilarity
    dot = Graph(comment="Graph created from complex data", strict=True)
    for person_id in range(nb_person):
        dot.node(str(person_id))
    for person_1_id in range(nb_person):
        for person_2_id in range(nb_person):
            # no self loops
            if not person_1_id == person_2_id:
                if dissimilarity_matrix[person_1_id, person_2_id] > threshold:
                    dot.edge(
                        str(person_1_id),
                        str(person_2_id),
                        color="darkolivegreen4",
                        penwidth="1.1",
                    )

    # visualize the graph
    dot.attr(label=f"threshold {threshold}", fontsize="20")
    graph_name = f"images/data_threshold_{threshold}"
    dot.render(graph_name)
