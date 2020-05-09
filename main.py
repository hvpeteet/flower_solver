from __future__ import annotations

from enum import IntEnum, auto
from typing import Type, Tuple, List, Set, Dict, cast, Deque
from collections import defaultdict, deque
from graphviz import Digraph
import heapq

COLOR_TABLE = {
    'rryyWWss' : 'White',
    'rryyWWSs' : 'White',
    'rryyWWSS' : 'White',
    'rryyWwss' : 'White',
    'rryyWwSs' : 'White',
    'rryyWwSS' : 'White',
    'rryywwss' : 'Purple',
    'rryywwSs' : 'Purple',
    'rryywwSS' : 'Purple',
    'rrYyWWss' : 'Yellow',
    'rrYyWWSs' : 'Yellow',
    'rrYyWWSS' : 'Yellow',
    'rrYyWwss' : 'White',
    'rrYyWwSs' : 'White',
    'rrYyWwSS' : 'White',
    'rrYywwss' : 'Purple',
    'rrYywwSs' : 'Purple',
    'rrYywwSS' : 'Purple',
    'rrYYWWss' : 'Yellow',
    'rrYYWWSs' : 'Yellow',
    'rrYYWWSS' : 'Yellow',
    'rrYYWwss' : 'Yellow',
    'rrYYWwSs' : 'Yellow',
    'rrYYWwSS' : 'Yellow',
    'rrYYwwss' : 'White',
    'rrYYwwSs' : 'White',
    'rrYYwwSS' : 'White',
    'RryyWWss' : 'Red',
    'RryyWWSs' : 'Pink',
    'RryyWWSS' : 'White',
    'RryyWwss' : 'Red',
    'RryyWwSs' : 'Pink',
    'RryyWwSS' : 'White',
    'Rryywwss' : 'Red',
    'RryywwSs' : 'Pink',
    'RryywwSS' : 'Purple',
    'RrYyWWss' : 'Orange',
    'RrYyWWSs' : 'Yellow',
    'RrYyWWSS' : 'Yellow',
    'RrYyWwss' : 'Red',
    'RrYyWwSs' : 'Pink',
    'RrYyWwSS' : 'White',
    'RrYywwss' : 'Red',
    'RrYywwSs' : 'Pink',
    'RrYywwSS' : 'Purple',
    'RrYYWWss' : 'Orange',
    'RrYYWWSs' : 'Yellow',
    'RrYYWWSS' : 'Yellow',
    'RrYYWwss' : 'Orange',
    'RrYYWwSs' : 'Yellow',
    'RrYYWwSS' : 'Yellow',
    'RrYYwwss' : 'Red',
    'RrYYwwSs' : 'Pink',
    'RrYYwwSS' : 'White',
    'RRyyWWss' : 'Black',
    'RRyyWWSs' : 'Red',
    'RRyyWWSS' : 'Pink',
    'RRyyWwss' : 'Black',
    'RRyyWwSs' : 'Red',
    'RRyyWwSS' : 'Pink',
    'RRyywwss' : 'Black',
    'RRyywwSs' : 'Red',
    'RRyywwSS' : 'Pink',
    'RRYyWWss' : 'Orange',
    'RRYyWWSs' : 'Orange',
    'RRYyWWSS' : 'Yellow',
    'RRYyWwss' : 'Red',
    'RRYyWwSs' : 'Red',
    'RRYyWwSS' : 'White',
    'RRYywwss' : 'Black',
    'RRYywwSs' : 'Red',
    'RRYywwSS' : 'Purple',
    'RRYYWWss' : 'Orange',
    'RRYYWWSs' : 'Orange',
    'RRYYWWSS' : 'Yellow',
    'RRYYWwss' : 'Orange',
    'RRYYWwSs' : 'Orange',
    'RRYYWwSS' : 'Yellow',
    'RRYYwwss' : 'Blue',
    'RRYYwwSs' : 'Red',
    'RRYYwwSS' : 'White'
}

class Gene(IntEnum):
    gg = auto()
    Gg = auto()
    GG = auto()

class Flower:
    def __init__(self, r : Gene, y : Gene, w : Gene, s : Gene):
        self.r = r
        self.y = y
        self.w = w
        self.s = s

    def toTuple(self) -> Tuple[Gene, Gene, Gene, Gene]:
        return (self.r, self.y, self.w, self.s)
    
    def __hash__(self) -> int:
        return hash(self.toTuple())

    def __eq__(self, other) -> bool:
        return self.toTuple() == other.toTuple()

    def __str__(self) -> str:
        final_str = ''
        # Red
        if self.r == Gene.gg:
            final_str += 'rr'
        elif self.r == Gene.Gg:
            final_str += 'Rr'
        else:
            final_str += 'RR'
        # Yellow
        if self.y == Gene.gg:
            final_str += 'yy'
        elif self.y == Gene.Gg:
            final_str += 'Yy'
        else:
            final_str += 'YY'
        # White
        if self.w == Gene.gg:
            final_str += 'ww'
        elif self.w == Gene.Gg:
            final_str += 'Ww'
        else:
            final_str += 'WW'
        # Shade
        if self.s == Gene.gg:
            final_str += 'ss'
        elif self.s == Gene.Gg:
            final_str += 'Ss'
        else:
            final_str += 'SS'
        return final_str

    def getChildrenWithProbabilities(self, other:Flower) -> List[Tuple[Flower, float]]:
        single_gene_children_counts = [getSingleGeneChildrenWithCounts(self.toTuple()[i], other.toTuple()[i]) for i in range(len(self.toTuple()))]
        final_counts: Dict[Flower, int] = {}
        for r_gene, r_count in single_gene_children_counts[0]:
            for y_gene, y_count in single_gene_children_counts[1]:
                for w_gene, w_count in single_gene_children_counts[2]:
                    for s_gene, s_count in single_gene_children_counts[3]:
                        child = Flower(r_gene, y_gene, w_gene, s_gene)
                        count = r_count * y_count * w_count * s_count
                        final_counts[child] = final_counts.get(child, 0) + count
        final_probabilities: List[Tuple[Flower, float]] = [(child, float(count) / 256) for child, count in final_counts.items()]
        return final_probabilities

def getSingleGeneChildrenWithCounts(parent_1 : Gene, parent_2 : Gene):
    # Parent 1 contributes left gene, parent 2 contributes alele 2
    allele_pairs: List[List[int]] = []
    for parent in [parent_1, parent_2]:
        if parent == Gene.gg:
            allele_pairs.append([0,0])
        elif parent == Gene.Gg:
            allele_pairs.append([0,1])
        else:
            allele_pairs.append([1, 1])
    summed_allele_counts: Dict[int, int] = {}
    for p1_allele in allele_pairs[0]:
        for p2_allele in allele_pairs[1]:
            s = p1_allele + p2_allele
            summed_allele_counts[s] = summed_allele_counts.get(s, 0) + 1
    final_counts: Dict[Gene, int] = {}
    for allele_sum, count in summed_allele_counts.items():
        if allele_sum == 0:
            final_counts[Gene.gg] = final_counts.get(Gene.gg, 0) + count
        elif allele_sum == 1:
            final_counts[Gene.Gg] = final_counts.get(Gene.Gg, 0) + count
        else:
            final_counts[Gene.GG] = final_counts.get(Gene.GG, 0) + count
    return final_counts.items()


def alleleFromString(allele_str : str) -> Gene:
    if len(allele_str) != 2:
        raise ValueError(allele_str + ' is not a valid allele')
    nUpper = sum([1 if c.isupper() else 0 for c in allele_str])
    if nUpper == 0:
        return Gene.gg
    elif nUpper == 1:
        return Gene.Gg
    return Gene.GG

def flowerFromString(genes_str : str) -> Flower:
    if len(genes_str) != 8:
        raise ValueError(genes_str + ' is not a valid set of alleles')
    return Flower(alleleFromString(genes_str[0:2]), alleleFromString(genes_str[2:4]), alleleFromString(genes_str[4:6]), alleleFromString(genes_str[6:8]))

def sortableFlower(flower: Flower):
    return flower.toTuple()

class Node:
    pass

class PairedNode(Node):
    def __init__(self, flower_1: Flower, flower_2: Flower):
        flowers_list = [flower_1, flower_2]
        flowers_list.sort(key=sortableFlower)
        self.flowers = tuple(flowers_list)
    def __hash__(self):
        return hash(self.flowers)
    def __eq__(self, other):
        if isinstance(other, PairedNode):
            return self.flowers == other.flowers
        return False
    def __str__(self):
        return '(' + str(self.flowers[0]) + ',' + str(self.flowers[1]) + ')'

class SingleNode(Node):
    def __init__(self, flower: Flower):
        self.flower = flower
    def __hash__(self):
        return hash(self.flower)
    def __eq__(self, other):
        if isinstance(other, SingleNode):
            return self.flower == other.flower
        return False
    def __str__(self):
        return str(self.flower)

# Costs are expected values of the number of days it will take.
def createGraph(flowers: List[Flower]) -> Dict[Node, Set[Node]]:
    adj_list: Dict[Node, Set[Node]] = {}
    explored: Set[Node] = set()
    to_explore: Set[Node] = set([SingleNode(flower) for flower in flowers])
    while to_explore:
        exploring = to_explore.pop()
        if exploring in explored:
            continue
        if isinstance(exploring, PairedNode):
            cast(PairedNode, exploring)
            adj_list[exploring] = set()
            for child, _ in exploring.flowers[0].getChildrenWithProbabilities(exploring.flowers[1]):
                child_node = SingleNode(child)
                if str(child) in COLOR_TABLE:
                    to_explore.add(child_node)
                    adj_list[exploring].add(child_node)
        else:
            exploring = cast(SingleNode, exploring)
            adj_list[exploring] = set()
            for other_node in explored:
                if isinstance(other_node, PairedNode):
                    continue
                other_node = cast(SingleNode, other_node)
                pairing = PairedNode(exploring.flower, other_node.flower)
                to_explore.add(pairing)
                adj_list[exploring].add(pairing)
                adj_list[other_node].add(pairing)
            self_pairing = PairedNode(exploring.flower, exploring.flower)
            to_explore.add(self_pairing)
            adj_list[exploring].add(self_pairing)
        explored.add(exploring)
    return adj_list

def isTerminalNodeFunc(color: str):
    return lambda node: COLOR_TABLE[str(node.flower)] == color if isinstance(node, SingleNode) else False

def reverseGraph(graph: Dict[Node, Set[Node]]) -> Dict[Node, Set[Node]]:
    rgraph: Dict[Node, Set[Node]] = {}
    for node in graph.keys():
        rgraph[node] = set()
    for src, neighbors in graph.items():
        for neighbor in neighbors:
            rgraph[neighbor].add(src)
    return rgraph

ABSURDLY_LARGE = 1 << 64

def _expectedTurns(start_node: Node, end_node: Node, probabilities: Dict[Tuple[PairedNode, SingleNode], float]) -> int:
    if isinstance(start_node, SingleNode):
        if end_node.flowers[0] == end_node.flowers[1]:
            return 1
        return 0
    expected_breed_turns = 1 / probabilities[(start_node, end_node)]
    dup_times = 0
    while expected_breed_turns > 4:
        expected_breed_turns /= 2
        dup_times += 1
    return int(expected_breed_turns) + dup_times + 1 # +1 for growth

def newSearch(graph, probabilities: Dict[Tuple[PairedNode, SingleNode], float], start_nodes, isGoalNode) -> Tuple[Dict[Node, Set[Node]], Dict[Node, int]]:
    rgraph = reverseGraph(graph)
    # Find all costs.
    min_costs: Dict[Node, int] = {}
    solo_node_min_cost_parent: Dict[SingleNode, PairedNode] = {}
    for node in graph.keys():
        min_costs[node] = ABSURDLY_LARGE
    for node in start_nodes:
        min_costs[node] = 0
    print("Calculating all path costs")
    for i in range(len(rgraph) + 2): # loop x #nodes (length of longest path), run an extra time so I don't have to synchronize the steps.
        print('\r', i, ' / ', len(rgraph) + 2, end='')
        for node, neighbors in rgraph.items():
            if node in start_nodes:
                continue
            if isinstance(node, SingleNode):
                # print(nodeWithColor(node))
                node = cast(SingleNode, node)
                min_cost, min_parent = min([(_expectedTurns(cast(PairedNode, neighbor), node, probabilities) + min_costs[neighbor], neighbor) for neighbor in neighbors], key=lambda tuple_container: tuple_container[0])
                solo_node_min_cost_parent[node] = cast(PairedNode, min_parent)
                min_costs[node] = min_cost
            if isinstance(node, PairedNode):
                min_cost = max([_expectedTurns(neighbor, node, probabilities) + min_costs[neighbor] for neighbor in neighbors])
                min_costs[node] = min_cost
    # for node, cost in min_costs.items():
    #     print(nodeWithColor(node), ':(', cost, ') [[ ', ', '.join([nodeWithColor(n) for n in graph[node]]), ' ]]')
    # Search greedy backwards
    # back_search_start_nodes: List[Node] = list(filter(lambda x: min_costs[x] < ABSURDLY_LARGE, filter(isGoalNode, graph.keys())))

    final_node = None
    final_cost = ABSURDLY_LARGE
    for node, cost in min_costs.items():
        if isGoalNode(node) and cost < final_cost:
            final_cost = cost
            final_node = node
    to_explore = set([final_node])
    explored = set()
    print("\nsearching")
    while to_explore:
        exploring = to_explore.pop()
        if exploring in explored or exploring in start_nodes:
            continue
        # Only explore the cheapest path.
        if isinstance(exploring, SingleNode):
            to_explore.add(solo_node_min_cost_parent[exploring])
        if isinstance(exploring, PairedNode):
            # Add both solo nodes that feed this one.
            for n in rgraph[exploring]:
                to_explore.add(n)
        explored.add(exploring)
    return explored, min_costs


def getSaneGraph(graph, start_nodes):
    # Find insane edges (where an edge is taken amoung other similar color edges)
    insane_edges: Set[Tuple[PairedNode, SingleNode]] = set()
    for src, dst_set in graph.items():
        if isinstance(src, SingleNode):
            continue
        color_counts:Dict[str, int] = defaultdict(int)
        for dst in dst_set:
            color_counts[COLOR_TABLE[str(dst.flower)]] += 1
        for dst in dst_set:
            color = COLOR_TABLE[str(dst.flower)]
            if color_counts[color] > 1:
                insane_edges.add((src, dst))
    # for paired, single in insane_edges:
    #     print("insane edge: ", paired, single)
    # Prune until no more puning needs to be done.
    # Any non-start leaf node needs to be pruned
    changed_last_iter = True
    sane_rgraph = reverseGraph(graph)
    while changed_last_iter:
        changed_last_iter = False
        nodes_to_remove = set()
        for node, neighbors in sane_rgraph.items():
            edges_to_remove = set()
            for neighbor in neighbors:
                # Remove any "insane" edges.
                if (neighbor, node) in insane_edges:
                    edges_to_remove.add(neighbor)
                # Remove any edges to non-existant nodes.
                if neighbor not in sane_rgraph:
                    # print('neighbor ', neighbor , ' not in rgraph')
                    edges_to_remove.add(neighbor)
            if edges_to_remove:
                # for edge_endpoint in edges_to_remove:
                    # Note that this looks reversed because we are working on the rgraph
                    # print('removing edge', edge_endpoint, '-->' , node)
                changed_last_iter = True
            sane_rgraph[node] = sane_rgraph[node] - edges_to_remove
            # Remove any SingleNode with no neighbors that is not in the starting set.
            if isinstance(node, SingleNode):
                if len(sane_rgraph[node]) == 0 and node not in start_nodes:
                    nodes_to_remove.add(node)
            # Remove any PairedNode with < 2 neighbors.
            if isinstance(node, PairedNode):
                # If the node no longer has inputs.
                if len(sane_rgraph[node]) < 2:
                    if node.flowers[0] != node.flowers[1]:
                        nodes_to_remove.add(node)
                    elif len(sane_rgraph[node]) == 0:
                        nodes_to_remove.add(node)
                # If the node no longer has outputs.
                if not any(n in sane_rgraph and node in sane_rgraph[n] for n in graph[node]):
                    nodes_to_remove.add(node)
        if nodes_to_remove:
            changed_last_iter = True
        for node in nodes_to_remove:
            del sane_rgraph[node]
            # print('deleting node ', str(node))
    sane_graph = reverseGraph(sane_rgraph)
    return sane_graph

def nodeWithColor(node):
    if isinstance(node, SingleNode):
        return COLOR_TABLE[str(node)] + ' <' + str(node) + '>'
    else:
        return '(' + COLOR_TABLE[str(node.flowers[0])] + ' <' + str(node.flowers[0]) + '>, ' +  COLOR_TABLE[str(node.flowers[1])] + ' <' + str(node.flowers[1])  + '>)'

def debugPrintGraph(graph):
    for node, neighbors in graph.items():
        print(nodeWithColor(node), " --> [", ', '.join([nodeWithColor(neighbor) for neighbor in neighbors]) , "]")

def testSaneGraphNoRemoval():
    seed_red = flowerFromString('RRyyWWSs')
    seed_yellow = flowerFromString('rrYYWWss')
    seed_white = flowerFromString('rryyWwss')
    # TODO(hvpeteet): Place tests in their own file once this isn't hacky. I will likely want to also place the Flower and Gene classes in seperate files as well.
    basicGraph = {}
    basicGraph[SingleNode(seed_red)] = set([PairedNode(seed_red, seed_yellow)])
    basicGraph[SingleNode(seed_yellow)] = set([PairedNode(seed_red, seed_yellow)])
    basicGraph[PairedNode(seed_red, seed_yellow)] = set([SingleNode(flowerFromString('RrYyWWss'))])
    basicGraph[SingleNode(flowerFromString('RrYyWWss'))] = set()
    sane = getSaneGraph(basicGraph, [SingleNode(seed_red), SingleNode(seed_yellow)])
    if (len(basicGraph) != len(sane)):
        print('testSaneGraphNoRemoval failed')
        print("Printing basic graph")
        debugPrintGraph(basicGraph)
        print('Printing sane basic graph')
        debugPrintGraph(sane)
    else:
        print('testSaneGraphNoRemoval passed')

def testSaneGraphInsaneEdge():
    seed_red = SingleNode(flowerFromString('RRyyWWSs'))
    seed_yellow = SingleNode(flowerFromString('rrYYWWss'))
    # seed_white = SingleNode(flowerFromString('rryyWwss'))
    orange_1 = SingleNode(flowerFromString('RrYyWWss'))
    orange_2 = SingleNode(flowerFromString('RrYYWWss'))
    blue = SingleNode(flowerFromString('RRYYwwss'))
    red_yellow = PairedNode(seed_red.flower, seed_yellow.flower)
    # yellow_white = PairedNode(seed_red.flower, seed_white.flower)
    red_orange = PairedNode(seed_red.flower, orange_1.flower)
    # Have the two oranges conflict from red_orange, but not red_yellow
    basicGraph = {}
    basicGraph[seed_red] = set([red_yellow, red_orange])
    basicGraph[seed_yellow] = set([red_yellow])
    basicGraph[red_yellow] = set([blue, orange_1])
    basicGraph[red_orange] = set([orange_1, orange_2, blue])
    basicGraph[blue] = set()
    basicGraph[orange_1] = set([red_orange])
    basicGraph[orange_2] = set()
    # basicGraph[seed_white] = set([yellow_white])

    sane = getSaneGraph(basicGraph, [seed_yellow, seed_red])
    if (len(sane) != 4):
        print('testSaneGraphInsaneEdge failed')
        print("------------Printing basic graph------------")
        debugPrintGraph(basicGraph)
        print('------------Printing sane basic graph------------')
        debugPrintGraph(sane)
    else:
        print('testSaneGraphInsaneEdge passed')

def runTests():
    testSaneGraphNoRemoval()
    testSaneGraphInsaneEdge()
    
def renderGraph(title, graph, filepath):
    dot = Digraph(comment=title)
    for src, adj in graph.items():
        dot.node(nodeWithColor(src))
        for dst in adj:
            dot.edge(nodeWithColor(src), nodeWithColor(dst))
    dot.render(filepath)


def main():
    seed_red = flowerFromString('RRyyWWSs')
    seed_yellow = flowerFromString('rrYYWWss')
    seed_white = flowerFromString('rryyWwss')
    # TODO(hvpeteet): Place tests in their own file once this isn't hacky. I will likely want to also place the Flower and Gene classes in seperate files as well.
    print('Running tests')
    runTests()
    # Assuming roses only
    print('Running main')
    seed_red = flowerFromString('RRyyWWSs')
    seed_yellow = flowerFromString('rrYYWWss')
    seed_white = flowerFromString('rryyWwss')
    start_flowers = [seed_red, seed_yellow, seed_white] #[seed_red, seed_yellow, seed_white]
    start_nodes = [SingleNode(n) for n in start_flowers]
    print('Creating full graph')
    graph = createGraph(start_flowers)
    print('finding sane graph')
    sane_graph = getSaneGraph(graph, start_nodes)

    # print('writing sane graph')
    # dot = Digraph(comment='Sane')
    # for src, adj in sane_graph.items():
    #     dot.node(nodeWithColor(src), nodeWithColor(src))
    #     for dst in adj:
    #         dot.edge(nodeWithColor(src), nodeWithColor(dst))
    # dot.render('sane')

    print('finding probabilities')
    probabilities: Dict[Tuple[PairedNode, SingleNode], float] = {}
    for node in sane_graph.keys():
        if isinstance(node, PairedNode):
            for child, prob in node.flowers[0].getChildrenWithProbabilities(node.flowers[1]):
                if SingleNode(child) in sane_graph:
                    probabilities[(node, SingleNode(child))] = prob

    for node_pair in probabilities:
        if node_pair[0] not in sane_graph:
            print("+++MISSING FROM SANE GRAPH+++ <", nodeWithColor(node_pair[0]) ,">")
        if node_pair[1] not in sane_graph:
            print("+++MISSING FROM SANE GRAPH+++ <", nodeWithColor(node_pair[1]) ,">")
    for node, neighbors in sane_graph.items():
        if isinstance(node, PairedNode):
            for neighbor in neighbors:
                if (node, neighbor) not in probabilities:
                    print("+++MISSING FROM PROB+++ <", nodeWithColor(node), ', ', nodeWithColor(neighbor) ,">")

    print('finding fastest path')
    fastest_path, min_costs = newSearch(sane_graph, probabilities, start_nodes, isTerminalNodeFunc('Blue'))
    fastest_path |= set(start_nodes)

    dot = Digraph(comment='Breeding Best')
    for src, adj in sane_graph.items():
        if src in fastest_path:
            dot.node(nodeWithColor(src), nodeWithColor(src) + '::' + str(min_costs[src]))
        for dst in adj:
            if src in fastest_path and dst in fastest_path:
                dot.edge(nodeWithColor(src), nodeWithColor(dst), str(_expectedTurns(src, dst, probabilities)) + '(' + str(probabilities.get((src, dst), 1.0)) + ')')
    dot.render('final.gv')

if __name__ == '__main__':
    main()