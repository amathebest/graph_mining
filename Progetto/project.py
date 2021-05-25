# libraries import
import os
import re
import json
import itertools
import networkx as nx
from scipy.stats import bernoulli

def main():
    # fetching all the files
    base_path = 'sequences/'
    files = os.listdir(base_path)

    # creating the Graph using networkx
    G = nx.Graph()
    node_index = 0

    # main loop for graph construction
    for file in files:
        # opening file
        with open(base_path + file) as in_file:
            data = json.load(in_file)

        # isolating the comment section
        if 'comment' in data['results'][0]:
            comments_section = data['results'][0]['comment']

            # isolating the authors
            authors = []
            for comment in comments_section:
                pattern = '(_([a-zA-Z]+\.?\s?)*_)'
                results = re.findall(pattern, comment)
                if results:
                    author = results[0][0].split('_')[1]
                    authors.append(author)

            # adding the nodes
            for author in authors:
                if author not in nx.get_node_attributes(G, 'name').values():
                    G.add_node(node_index, name = author)
                    node_index += 1

            # adding the edges
            for pair in itertools.combinations(authors, 2):
                # fetching node 1
                node_1_key = list(nx.get_node_attributes(G, 'name').values()).index(pair[0])
                # fetching node 2
                node_2_key = list(nx.get_node_attributes(G, 'name').values()).index(pair[1])
                # adding the edge if it isn't already in the graph
                if (node_1_key, node_2_key) not in list(G.edges):
                    G.add_edge(node_1_key, node_2_key)



if __name__ == '__main__':
    main()

# function that finds one MIS and returns it
# complexity: O(n+m)
def greedy_MIS(G):
    # inizialize:
    # - V_set as a copy of the vertex set
    # - E_set as a copy of the edge set
    # - maximal_independent_set that will be returned
    V_set = list(G.nodes).copy()
    maximal_independent_set = []
    # main loop
    while len(V_set) > 0:
        node = random.choice(V_set)
        maximal_independent_set.append(node)
        V_set.remove(node)
        for adj_node in list(G.adj[node]):
            if adj_node in V_set:
                V_set.remove(adj_node)
    return maximal_independent_set

# function that finds one MIS and returns it
# complexity: O(logn) when run in parallel
def luby_MIS(G):
    # inizialization of V_set as a copy of the vertex set, E_set as a copy of the edge set
    # and maximal_independent_set that will be returned
    V_set = list(G.nodes).copy()
    E_set = list(G.edges).copy()
    maximal_independent_set = []
    # main loop
    while len(V_set) > 0:
        # creating a probability distribution associated to each node and
        # marking each node with that probability (1/2*node's degree)
        probability_dist = [1/(2*G.degree[node]) if G.degree[node] > 0 else 1 for node in V_set]
        marks = bernoulli.rvs(probability_dist, size = len(V_set))
        zip_iter = zip(V_set, marks)
        marked_nodes = dict(zip_iter)
        # removing a mark if both ends of an edge are marked
        for edge in E_set:
            if marked_nodes[edge[0]] == 1 and marked_nodes[edge[1]] == 1:
                if G.degree[edge[0]] < G.degree[edge[1]]:
                    marked_nodes[edge[0]] = 0
                elif G.degree[edge[0]] > G.degree[edge[1]]:
                    marked_nodes[edge[1]] = 0
                else:
                    if edge[0] < edge[1]:
                        marked_nodes[edge[0]] = 0
                    else:
                        marked_nodes[edge[1]] = 0
        # adding every marked node to the MIS
        for idx, node in enumerate(V_set):
            if marked_nodes[idx] == 1:
                maximal_independent_set.append(node)
        # removing every added node and their neighbors from the V_set
        for node in maximal_independent_set:
            V_set.remove(node)
            for adj_node in G.adj[node]:
                if adj_node in V_set:
                    V_set.remove(adj_node)
        # removing every edge that doesn't exist anymore
        for edge in E_set:
            if not(edge[0] in V_set and edge[1] in V_set):
                E_set.remove(edge)
    return maximal_independent_set


def find_all_MIS(G):

    return
