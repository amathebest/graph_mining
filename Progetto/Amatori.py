# Python script that contains all the work for the APMD exam.
# This file contains all the implementation of the functions needed to compute:
# - a MIS of G
# - all the MIS of G
# - a MaximumIS of G
# I used both naive approaches and the approach explained in the paper:
# Alessio Conte, Roberto Grossi, Andrea Marino, Takeaki Uno, Luca Versari. Listing Maximal Independent Sets with Minimal Space and Bounded Delay. International Symposium on String Processing
# and Information Retrieval (SPIRE), Sep 2017, Palermo, Italy. pp.144-160, ff10.1007/978-3-319-67428-5_13ff. ffhal-01609012f
#
# Amatori Matteo

# libraries import
import os
import re
import json
import random
import itertools
import numpy as np
import networkx as nx
from scipy.stats import bernoulli

def main():
    # building the graph
    number_of_nodes = 30
    G = build_graph(number_of_nodes)

    print("Single MIS (naive):")
    print(greedy_MIS(G))

    print("Single MIS (Luby's algorithm):")
    print(luby_MIS(G))

    print("All MIS found:")
    all_MIS = wrap_iterative_spawn(G, 'all')
    print(all_MIS)

    print("Longest MIS:")
    print(max(all_MIS, key = len))


def build_graph(number_of_nodes):
    # fetching all the files
    base_path = '/home/heaven/Documents/graph_mining/Progetto/sequences/'
    files = os.listdir(base_path)
    # creating the Graph using networkx
    G = nx.Graph()
    node_index = 0
    # variables for progress output
    i = 1
    # main loop for graph construction
    print('Reading files and creating the graph...')
    for file in files:
        # looping until we reach the specified number of nodes (-1 to specify no limit)
        if node_index > number_of_nodes and not number_of_nodes == -1:
            break
        # filtering for only json files
        if '.json' in file:
            # opening current file
            with open(base_path + file) as in_file:
                data = json.load(in_file)
            # progress output
            if number_of_nodes == -1:
                print(str("%.3f" % (i*100/len(files))) + "%")
            else:
                print(str("%.3f" % (node_index*100/number_of_nodes)) + "%")
            i += 1
            # isolating the comment section
            if 'comment' in data['results'][0]:
                comments_section = data['results'][0]['comment']
                # isolating the authors
                authors = []
                for comment in comments_section:
                    pattern = '_[A-Z][\w+\s?\.?]+_'
                    results = re.findall(pattern, comment)
                    if results:
                        author = results[0].split('_')[1]
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
    return G

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

# function that implements the Luby's algorithm, it finds one MIS and returns it
# complexity: O(logn) when run in parallel
def luby_MIS(G):
    # inizialization of V_set as a copy of the vertex set, E_set as a copy of the edge set and maximal_independent_set that will be returned
    V_set = set(G.nodes).copy()
    E_set = list(G.edges).copy()
    maximal_independent_set = []
    # main loop
    while len(V_set) > 0:
        # creating a probability distribution associated to each node and marking each node with that probability (1/2*node's degree)
        probability_dist = [1/(2*G.degree[node]) if G.degree[node] > 0 else 1 for node in V_set]
        marks = bernoulli.rvs(probability_dist, size = len(V_set))
        marked_nodes = dict(zip(V_set, marks))
        # removing a mark if both ends of an edge are marked
        for edge in E_set:
            if edge[0] in marked_nodes and edge[1] in marked_nodes:
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
        for node in V_set:
            if marked_nodes[node] == 1:
                maximal_independent_set.append(node)
        # removing every added node and their neighbors from the V_set
        for node in maximal_independent_set:
            if node in V_set:
                V_set.remove(node)
            V_set = V_set.difference(set(G.adj[node]))
        # removing every edge that doesn't exist anymore
        for edge in E_set:
            if not(edge[0] in V_set and edge[1] in V_set):
                E_set.remove(edge)
    return maximal_independent_set

# function that returns true if the independent set is maximal (non-extendible)
def check_if_set_is_MIS(G, IS):
    nodes_without_IS = [node for node in list(G.nodes) if node not in IS]
    if any(is_node_addable_to_IS(G, IS, node) for node in nodes_without_IS):
        return False
    else:
        return True

# function that returns true if the node we're trying to add doesn't turn the set into a dependent set
def is_node_addable_to_IS(G, IS, new_node):
    if any(new_node in G.adj[node] for node in IS):
        return False
    else:
        return True

# recursive function of the naive approach to find all MIS in the given graph
# complexity: exponential with n
def find_all_MIS_rec(G, node, V_set, remaining_nodes, set_of_found_MIS):
    # exit condition: go back in the recursion tree when there are no remaining nodes available
    if set(remaining_nodes) == set():
        return
    # core of the recursion, we add the node to the set and we remove its adjacency list from the available nodes
    V_set.append(node)
    remaining_nodes.remove(node)
    for adj_node in G.adj[node]:
        if adj_node in remaining_nodes:
            remaining_nodes.remove(adj_node)
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
                find_all_MIS_rec(G, rem_node, V_set.copy(), remaining_nodes.copy(), set_of_found_MIS)

# wrap function of the naive approach to find all MIS (or the maximum IS, depending on the mode specified) in the given graph
def find_all_MIS(G, mode):
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
        if is_node_addable_to_IS(G, V_set, node):
            find_all_MIS_rec(G, node, V_set.copy(), remaining_nodes.copy(), set_of_found_MIS)
    if mode == 'all':
        return set_of_found_MIS
    else:
        return max(set_of_found_MIS, key = len)

# implementation of the functions described in paper cited in the title of the script for tasks 2 and 3 (all MIS and MaximumIS)
# function that returns a degeneracy ordering for the graph G
def get_degeneracy_ordering(G):
    deg_ordering = []
    graph = G.copy()
    degrees = dict(graph.degree)
    for i in range(graph.number_of_nodes()):
        min_degree_node = min(degrees, key = degrees.get)
        graph.remove_node(min_degree_node)
        del degrees[min_degree_node]
        deg_ordering.append(min_degree_node)
    return deg_ordering

# function that given a set A returns its minimum element according to the degeneracy ordering provided
def get_minimum_elem_in_deg_ord(deg_ord, A):
    if None in A:
        return A[0]
    for node in deg_ord:
        if node in A:
            return node

# function that returns the complementary neighborhood of a specified node set, that is all the nodes that don't appear in the adjacency lists
# of the nodes in the set. doesn't count duplicates.
def get_set_complementary_neighborhood(G, A):
    compl_neighborhood = []
    for set_node in A:
        compl_neighborhood.append(list(nx.non_neighbors(G, set_node)))
    if not compl_neighborhood:
        return []
    else:
        intersection = set.intersection(*map(set, compl_neighborhood))
        return list(intersection)

# function that returns the neighbors of the nodes that come before the specified node in the degeneracy ordering
def get_neighbors_of_node_before_deg_ord(G, deg_ord, v):
    return list(set.intersection(set(G.adj[v]), set(deg_ord[:deg_ord.index(v)])))

# function that returns the intersection between the specified set A and the nodes that appear before the specified node in the degeneracy order
def get_nodes_in_set_before_spec(deg_ord, A, v):
    return list(set.intersection(set(A), set(deg_ord[:deg_ord.index(v)])))

# function that returns the intersection between the specified set A and the nodes that appear before the specified node or at the specified node in the degeneracy order
def get_nodes_in_set_before_equal_spec(deg_ord, A, v):
    return list(set.intersection(set(A), set(deg_ord[:deg_ord.index(v)+1])))

# function that returns the intersection between the specified set A and the nodes that appear after the specified node in the degeneracy order
def get_nodes_in_set_after_spec(deg_ord, A, v):
    return list(set.intersection(set(A), set(deg_ord[deg_ord.index(v)+1:])))

# function that returns the set I<v intersected with the complementary neighborhood of v, that is I'v
def get_I_prime_set(G, deg_ord, IS, v):
    return list(set.intersection(set(get_nodes_in_set_before_spec(deg_ord, IS, v)), set(nx.non_neighbors(G, v))))

# function that returns true whether the specified IS is root in the solution digraph
def is_root(G, IS, deg_ord):
    return get_minimum_elem_in_deg_ord(deg_ord, IS) == get_parent_index(G, IS, deg_ord)

# function that restores the state of the parent node
def parent_state(G, IS, deg_ord):
    v = get_parent_index(G, IS, deg_ord)
    I_before_v = get_nodes_in_set_before_spec(deg_ord, IS, v)
    return complete_MIS(G, I_before_v, deg_ord), v

# function that returns the next child candidate to explore in the solution DiGraph based on the current IS and node
def get_next_candidate_ms(G, IS, deg_ord, v, checked):
    diff_V_IS = [node for node in G.nodes if node not in IS and node not in checked]
    diff_after_v = get_nodes_in_set_after_spec(deg_ord, diff_V_IS, v)
    return get_minimum_elem_in_deg_ord(deg_ord, diff_after_v)

# function that returns the complete MIS based on the current IS, that is the MIS obtained by continuously adding the minimum available node in the degeneracy
# ordering provided as parameter
def complete_MIS(G, IS, deg_ord):
    V_set = get_set_complementary_neighborhood(G, IS)
    while len(V_set) > 0:
        node = get_minimum_elem_in_deg_ord(deg_ord, V_set)
        IS.append(node)
        V_set.remove(node)
        for adj_node in list(G.adj[node]):
            if adj_node in V_set:
                V_set.remove(adj_node)
    return IS

# function that returns the index of the parent of the specified IS
def get_parent_index(G, IS, deg_ord):
    candidates = []
    for node in IS:
        if set(complete_MIS(G, get_nodes_in_set_before_equal_spec(deg_ord, IS, node), deg_ord)) == set(IS):
            candidates.append(node)
    return get_minimum_elem_in_deg_ord(deg_ord, candidates)

# function that builds the additional array that stores, for each node in G, the amount of neighbors it has in the set I<v
def build(G, A):
    ws = []
    for v in list(G.nodes):
        ws.append(len(set(G.adj[v]).intersection(set(A))))
    return ws

# function that updates the additional array
def update(G, ws, A):
    for v in list(G.nodes):
        ws[v] += len(set(G.adj[v]).intersection(set(A)))

# restores the parent state and re-builds the additional array
def restore(G, IS, deg_ord, v):
    IS, v = parent_state(G, IS, deg_ord)
    prev = v
    I_before_v = get_nodes_in_set_before_spec(deg_ord, IS, v)
    ws = build(G, I_before_v)
    return IS, ws, v, prev

# function that checks if a given node in the solution DiGraph has a child that needs to be visited
def child_exist_fast(G, IS, v, ws, deg_ord):
    neigh_before_v = get_neighbors_of_node_before_deg_ord(G, deg_ord, v)
    I_intersec_neigh_before_v = set(IS).intersection(set(neigh_before_v))
    for x in I_intersec_neigh_before_v:
        for y in G.adj[x]:
            ws[y] -= 1
    C = [node for node in list(G.nodes) if ws[node] == 0]
    I_prime_v = get_I_prime_set(G, deg_ord, IS, v)
    cn_I_prime_v_plus_v = get_set_complementary_neighborhood(G, I_prime_v + [v])
    min_cn_I = get_minimum_elem_in_deg_ord(deg_ord, cn_I_prime_v_plus_v)
    if not get_minimum_elem_in_deg_ord(deg_ord, [v, min_cn_I]) == v:
        return False
    while len(C) > 0:
        min_c = get_minimum_elem_in_deg_ord(deg_ord, C)
        diff_I_I_prime_v = set(IS).difference(set(I_prime_v))
        if min_c in diff_I_I_prime_v:
            C = set(C).difference(set(list(G.adj[min_c]) + [min_c]))
        else:
            update(G, ws, set(IS).intersection(set(neigh_before_v)))
            return min_c in IS

# function that performs the DFS traversal of the solution DiGraph in which each node is a MIS
def iterative_spawn(G, IS, deg_ord, set_of_found_MIS):
    v = get_parent_index(G, IS, deg_ord)
    prev = v
    ws = build(G, get_nodes_in_set_before_spec(deg_ord, IS, v))
    paths_checked = []
    while True:
        childless = True
        checked = []
        while get_next_candidate_ms(G, IS, deg_ord, v, checked) is not None:
            v = get_next_candidate_ms(G, IS, deg_ord, v, checked)
            #print('state:', IS, v, paths_checked, checked)
            update(G, ws, set(get_nodes_in_set_before_spec(deg_ord, IS, v)).difference(set(get_nodes_in_set_before_spec(deg_ord, IS, prev))))
            prev = v
            if not get_minimum_elem_in_deg_ord(deg_ord, [v, get_parent_index(G, IS, deg_ord)]) == v:
                if child_exist_fast(G, IS, v, ws, deg_ord):
                    I_prime_v = get_I_prime_set(G, deg_ord, IS, v)
                    if complete_MIS(G, I_prime_v + [v], deg_ord) not in paths_checked:
                        IS = complete_MIS(G, I_prime_v + [v], deg_ord)
                        if set(IS) not in [set(MIS) for MIS in set_of_found_MIS]:
                            #print('found:', IS, v)
                            set_of_found_MIS.append(IS)
                            childless = False
                            ws = build(G, get_nodes_in_set_before_spec(deg_ord, IS, v))
                            break
                        else:
                            if IS not in paths_checked:
                                #print('done:', IS)
                                paths_checked.append(IS)
                    else:
                        checked.append(v)
                else:
                    checked.append(v)
                    if v == deg_ord[-1]:
                        if IS not in paths_checked:
                            #print('done with path:', IS)
                            paths_checked.append(IS)
        if childless:
            if is_root(G, IS, deg_ord):
                return
            else:
                IS, ws, v, prev = restore(G, IS, deg_ord, v)
                #print("restoring parent:", IS)

# function that prepares roots for the DFS traversal and then calls the main function
def wrap_iterative_spawn(G, mode):
    deg_ord = get_degeneracy_ordering(G)
    G_rev_ord = deg_ord[::-1]
    set_of_found_MIS = []
    for node in G_rev_ord:
        IS = []
        for v in G_rev_ord:
            if G.degree[v] == 0:
                IS.append(v)
        IS.append(node)
        root = complete_MIS(G, IS, G_rev_ord)
        #print('ROOT:', root)
        if set(root) not in [set(MIS) for MIS in set_of_found_MIS]:
            set_of_found_MIS.append(root)
        iterative_spawn(G, root, G_rev_ord, set_of_found_MIS)
    if mode == 'all':
        return set_of_found_MIS
    else:
        return max(set_of_found_MIS, key = len)

if __name__ == '__main__':
    main()
