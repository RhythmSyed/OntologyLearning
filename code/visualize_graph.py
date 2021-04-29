from pyvis.network import Network
import pandas as pd
import sys
import csv
import webbrowser
import os
import io
import pydotplus
from IPython.display import display, Image
from rdflib.tools.rdf2dot import rdf2dot
from rdflib import Graph, URIRef, Literal


def visualize(g):
    stream = io.StringIO()
    rdf2dot(g, stream, opts = {display})
    dg = pydotplus.graph_from_dot_data(stream.getvalue())
    png = dg.create_png()

    with open("ontology.png", "wb") as file:
        file.write(png)


def main(ontology_path):
    dataset = pd.read_csv(ontology_path, delimiter=',', header=None, error_bad_lines=False, quoting=csv.QUOTE_ALL)
    dataset.columns = ['subject', 'relation', 'object']

    # g = Network(500, 1000, notebook=True)
    graph = Graph()
    for index, row in dataset.iterrows():
        s = URIRef('https://www.example.com/' + row['subject'].replace(' ', '-'))
        r = URIRef('https://www.example.com/' + row['relation'].replace(' ', '-'))
        o = URIRef('https://www.example.com/' + row['object'].replace(' ', '-'))
        graph.add((s, r, o))

    with open('ontology.ttl', 'w') as file:
        file.write(graph.serialize(format='turtle').decode('utf-8'))

    g = Graph()
    g.parse('./ontology.ttl', format='turtle')
    visualize(g)

        # g.add_node(row['subject'])
        # g.add_node(row['object'])
        # g.add_edge(row['subject'], row['object'], label=row['relation'])

    # g.show('starwars_ontology.html')
    webbrowser.open('file://' + os.path.realpath('ontology.png'))


if __name__ == '__main__':
    if len(sys.argv) < 1:
        sys.exit()
    ontology_path = str(sys.argv[1])
    # ontology_path = './ontology.txt'
    main(ontology_path)
