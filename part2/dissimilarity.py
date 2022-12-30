from csv import reader
from graphviz import Graph

import math
import numpy as np


def compute_dissimilarity(f, s):
    if f[3] == s[3]:
        dissimilarity_job = 0
    else:
        dissimilarity_job = 10
    if f[4] == s[4]:
        dissimilarity_city = 0
    else:
        dissimilarity_city = 10
    if f[5] == s[5]:
        dissimilarity_music = 0
    else:
        dissimilarity_music = 10
    d = math.sqrt(
        (float(f[1]) - float(s[1])) ** 2
        + (float(f[2]) - float(s[2])) ** 2
        + dissimilarity_job
        + dissimilarity_city
        + dissimilarity_music
    )
    return d


with open('dataset.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    data = []
    for row in csv_reader:
        data.append(row)
    data = data[1:]
    nb_person = len(data)
    d_matrix = np.zeros((nb_person, nb_person))
    print("compute dissimilarities")
    for person_1_id in range(nb_person):
        for person_2_id in range(nb_person):
            dis = compute_dissimilarity(data[person_1_id], data[person_2_id])
            d_matrix[person_1_id, person_2_id] = dis
    print('saving dissimilarity matrix...')
    np.save('dissimilarity_matrix', d_matrix)
    threshold = 20
    for d in d_matrix:
        print(d)
    # build a graph from the dissimilarity
    dot = Graph(comment="Graph created from complex data", strict=True)
    for person_id in range(nb_person):
        dot.node(str(person_id))
    for person_1_id in range(nb_person):
        for person_2_id in range(nb_person):
            if not person_1_id == person_2_id:
                if d_matrix[person_1_id, person_2_id] > threshold:
                    dot.edge(
                        str(person_1_id),
                        str(person_2_id),
                        color="blue",
                        penwidth="1.1",
                    )
    # visualize the graph
    print('preparing the visualisation...')
    dot.attr(label=f"threshold {threshold}", fontsize="20")
    graph_name = f"result/dissimilarity_{threshold}"
    dot.render(graph_name)
