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

# function that returns true if the set is independent (no 2 vertex build an edge of G)
def check_if_set_is_IS(G, IS):
    pairs = itertools.permutations(IS, 2)
    if any([pair in list(G.edges) for pair in pairs]):
        return False
    else:
        return True

# function that returns true if the independent set is maximal (non-extendible)
def check_if_set_is_MIS(G, IS):
    nodes_without_IS = [node for node in list(G.nodes) if node not in IS]
    if any([is_node_addable_to_IS(G, IS, node) for node in nodes_without_IS]):
        return False
    else:
        return True

# function that returns true if the node we're trying to add doesn't turn the set into a dependent set
def is_node_addable_to_IS(G, IS, new_node):
    pairs = [(new_node, node) for node in IS] + [(node, new_node) for node in IS]
    if any([pair in list(G.edges) for pair in pairs]):
        return False
    else:
        return True

# recursive function
def find_all_MIS_rec(G, node, V_set, adj_V_set, remaining_nodes, set_of_found_MIS):
    # exit condition: go back in the recursion tree when there are no remaining nodes available
    if set(remaining_nodes) == set():
        return
    # core of the recursion, we add the node to the set and we remove its adjacency list from the available nodes
    V_set.append(node)
    remaining_nodes.remove(node)
    for adj_node in G.adj[node]:
        if adj_node in remaining_nodes:
            remaining_nodes.remove(adj_node)
        if adj_node not in adj_V_set:
            adj_V_set.append(adj_node)
    # if the set is MIS, we store it and then we go up a level removing the last added node
    if check_if_set_is_MIS(G, V_set):
        new_MIS = V_set.copy()
        if set(new_MIS) not in [set(MIS) for MIS in set_of_found_MIS]:
            set_of_found_MIS.append(new_MIS)
        V_set.remove(node)
    # main recursion loop: we proceed further only on the addable nodes and on the ones that don't form an already found MIS with the current V_set
    if set(V_set + remaining_nodes) not in [set(MIS) for MIS in set_of_found_MIS]:
        for rem_node in remaining_nodes:
            if is_node_addable_to_IS(G, V_set, rem_node):
                find_all_MIS_rec(G, rem_node, V_set.copy(), adj_V_set.copy(), remaining_nodes.copy(), set_of_found_MIS)


def find_all_MIS(G):
    set_of_found_MIS = []
    V_set = []
    remaining_nodes = list(G.nodes).copy()
    # adding all nodes with degree 0
    for node in remaining_nodes:
        if G.degree[node] == 0:
            V_set.append(node)
    for node in V_set:
        remaining_nodes.remove(node)
    for node in remaining_nodes:
        adj_V_set = []
        if is_node_addable_to_IS(G, V_set, node):
            find_all_MIS_rec(G, node, V_set.copy(), adj_V_set.copy(), remaining_nodes.copy(), set_of_found_MIS)
    return set_of_found_MIS


def find_maximum_MIS(G):
    set_of_found_MIS = []
    V_set = []
    remaining_nodes = list(G.nodes).copy()
    # adding all nodes with degree 0
    for node in remaining_nodes:
        if G.degree[node] == 0:
            V_set.append(node)
    for node in V_set:
        remaining_nodes.remove(node)
    for node in remaining_nodes:
        adj_V_set = []
        if is_node_addable_to_IS(G, V_set, node):
            find_all_MIS_rec(G, node, V_set.copy(), adj_V_set.copy(), remaining_nodes.copy(), set_of_found_MIS)
    return max(set_of_found_MIS, key = len)
