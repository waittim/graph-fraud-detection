import os
import re
import dgl
import numpy as np
import torch as th
import logging
#from data import parse_edgelist, get_features, read_edges, read_masked_nodes
import pandas as pd


def get_features(id_to_node, node_features):
    """

    :param id_to_node: dictionary mapping node names(id) to dgl node idx
    :param node_features: path to file containing node features
    :return: (np.ndarray, list) node feature matrix in order and new nodes not yet in the graph
    """
    indices, features, new_nodes = [], [], []
    max_node = max(id_to_node.values())

    is_1st_line = True
    with open(node_features, "r") as fh:
        for line in fh:
            # hard-coding to ignore the 1st line of header
            # if is_1st_line:
            #     is_1st_line = False
            #     continue

            node_feats = line.strip().split(",")
            node_id = node_feats[0]
            feats = np.array(list(map(float, node_feats[1:])))
            features.append(feats)
            if node_id not in id_to_node:
                max_node += 1
                id_to_node[node_id] = max_node
                new_nodes.append(max_node)

            indices.append(id_to_node[node_id])

    features = np.array(features).astype('float32')
    features = features[np.argsort(indices), :]
    return features, new_nodes


def get_labels(id_to_node, n_nodes, target_node_type, labels_path, masked_nodes_path, additional_mask_rate=0):
    """

    :param id_to_node: dictionary mapping node names(id) to dgl node idx
    :param n_nodes: number of user nodes in the graph
    :param target_node_type: column name for target node type
    :param labels_path: filepath containing labelled nodes
    :param masked_nodes_path: filepath containing list of nodes to be masked
    :param additional_mask_rate: additional_mask_rate: float for additional masking of nodes with labels during training
    :return: (list, list) train and test mask array
    """
    node_to_id = {v: k for k, v in id_to_node.items()}
    user_to_label = pd.read_csv(labels_path).set_index(target_node_type)
    labels = user_to_label.loc[map(int, pd.Series(node_to_id)[np.arange(n_nodes)].values)].values.flatten()
    masked_nodes = read_masked_nodes(masked_nodes_path)
    train_mask, test_mask = _get_mask(id_to_node, node_to_id, n_nodes, masked_nodes,
                                      additional_mask_rate=additional_mask_rate)
    return labels, train_mask, test_mask


def read_masked_nodes(masked_nodes_path):
    """
    Returns a list of nodes extracted from the path passed in

    :param masked_nodes_path: filepath containing list of nodes to be masked i.e test users
    :return: list
    """
    with open(masked_nodes_path, "r") as fh:
        masked_nodes = [line.strip() for line in fh]
    return masked_nodes


def _get_mask(id_to_node, node_to_id, num_nodes, masked_nodes, additional_mask_rate):
    """
    Returns the train and test mask arrays

    :param id_to_node: dictionary mapping node names(id) to dgl node idx
    :param node_to_id: dictionary mapping dgl node idx to node names(id)
    :param num_nodes: number of user/account nodes in the graph
    :param masked_nodes: list of nodes to be masked during training, nodes without labels
    :param additional_mask_rate: float for additional masking of nodes with labels during training
    :return: (list, list) train and test mask array
    """
    train_mask = np.ones(num_nodes)
    test_mask = np.zeros(num_nodes)
    for node_id in masked_nodes:
        train_mask[id_to_node[node_id]] = 0
        test_mask[id_to_node[node_id]] = 1
    if additional_mask_rate and additional_mask_rate < 1:
        unmasked = np.array([idx for idx in range(num_nodes) if node_to_id[idx] not in masked_nodes])
        yet_unmasked = np.random.permutation(unmasked)[:int(additional_mask_rate*num_nodes)]
        train_mask[yet_unmasked] = 0
    return train_mask, test_mask


def _get_node_idx(id_to_node, node_type, node_id, ptr):
    if node_type in id_to_node:
        if node_id in id_to_node[node_type]:
            node_idx = id_to_node[node_type][node_id]
        else:
            id_to_node[node_type][node_id] = ptr
            node_idx = ptr
            ptr += 1
    else:
        id_to_node[node_type] = {}
        id_to_node[node_type][node_id] = ptr
        node_idx = ptr
        ptr += 1

    return node_idx, id_to_node, ptr


def parse_edgelist(edges, id_to_node, header=False, source_type='user', sink_type='user'):
    """
    Parse an edgelist path file and return the edges as a list of tuple
    :param edges: path to comma separated file containing bipartite edges with header for edgetype
    :param id_to_node: dictionary containing mapping for node names(id) to dgl node indices
    :param header: boolean whether or not the file has a header row
    :param source_type: type of the source node in the edge. defaults to 'user' if no header
    :param sink_type: type of the sink node in the edge. defaults to 'user' if no header.
    :return: (list, dict) a list containing edges of a single relationship type as tuples and updated id_to_node dict.
    """
    edge_list = []
    rev_edge_list = []
    source_pointer, sink_pointer = 0, 0
    with open(edges, "r") as fh:
        for i, line in enumerate(fh):
            source, sink = line.strip().split(",")
            if i == 0:
                if header:
                    source_type, sink_type = source, sink
                if source_type in id_to_node:
                    source_pointer = max(id_to_node[source_type].values()) + 1
                if sink_type in id_to_node:
                    sink_pointer = max(id_to_node[sink_type].values()) + 1
                continue

            source_node, id_to_node, source_pointer = _get_node_idx(id_to_node, source_type, source, source_pointer)
            if source_type == sink_type:
                sink_node, id_to_node, source_pointer = _get_node_idx(id_to_node, sink_type, sink, source_pointer)
            else:
                sink_node, id_to_node, sink_pointer = _get_node_idx(id_to_node, sink_type, sink, sink_pointer)

            edge_list.append((source_node, sink_node))
            rev_edge_list.append((sink_node, source_node))

    return edge_list, rev_edge_list, id_to_node, source_type, sink_type


def read_edges(edges, nodes=None):
    """
    Read edges and node features

    :param edges: path to comma separated file containing all edges
    :param nodes: path to comma separated file containing all nodes + features
    :return: (list, list, list, dict) sources, sinks, features and id_to_node dictionary containing mappings
    from node names(id) to dgl node indices
    """
    node_pointer = 0
    id_to_node = {}
    features = []
    sources, sinks = [], []
    if nodes is not None:
        with open(nodes, "r") as fh:
            for line in fh:
                node_feats = line.strip().split(",")
                node_id = node_feats[0]
                if node_id not in id_to_node:
                    id_to_node[node_id] = node_pointer
                    node_pointer += 1
                    if len(node_feats) > 1:
                        feats = np.array(list(map(float, node_feats[1:])))
                        features.append(feats)
        with open(edges, "r") as fh:
            for line in fh:
                source, sink = line.strip().split(",")
                sources.append(id_to_node[source])
                sinks.append(id_to_node[sink])
    else:
        with open(edges, "r") as fh:
            for line in fh:
                source, sink = line.strip().split(",")
                if source not in id_to_node:
                    id_to_node[source] = node_pointer
                    node_pointer += 1
                if sink not in id_to_node:
                    id_to_node[sink] = node_pointer
                    node_pointer += 1
                sources.append(id_to_node[source])
                sinks.append(id_to_node[sink])

    return sources, sinks, features, id_to_node
















def get_logger(name):
    logger = logging.getLogger(name)
    log_format = '%(asctime)s %(levelname)s %(name)s: %(message)s'
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger.setLevel(logging.INFO)
    return logger















logging = get_logger(__name__)

def get_edgelists(edgelist_expression, directory):
    if "," in edgelist_expression:
        return edgelist_expression.split(",")
    files = os.listdir(directory)
    compiled_expression = re.compile(edgelist_expression)
    return [filename for filename in files if compiled_expression.match(filename)]

def construct_graph(training_dir, edges, nodes, target_node_type):

    print("Getting relation graphs from the following edge lists : {} ".format(edges))
    edgelists, id_to_node = {}, {}
    for i, edge in enumerate(edges):
        edgelist, rev_edgelist, id_to_node, src, dst = parse_edgelist(os.path.join(training_dir, edge), id_to_node, header=True)
        if src == target_node_type:
            src = 'target'
        if dst == target_node_type:
            dst = 'target'

        if src == 'target' and dst == 'target':
            print("Will add self loop for target later......")
        else:
            edgelists[(src, src + '<>' + dst, dst)] = edgelist
            edgelists[(dst, dst + '<>' + src, src)] = rev_edgelist
            print("Read edges for {} from edgelist: {}".format(src + '<' + dst + '>', os.path.join(training_dir, edge)))

    # get features for target nodes
    features, new_nodes = get_features(id_to_node[target_node_type], os.path.join(training_dir, nodes))
    print("Read in features for target nodes")

    # add self relation
    edgelists[('target', 'self_relation', 'target')] = [(t, t) for t in id_to_node[target_node_type].values()]

    g = dgl.heterograph(edgelists)
    print(
        "Constructed heterograph with the following metagraph structure: Node types {}, Edge types{}".format(
            g.ntypes, g.canonical_etypes))
    print("Number of nodes of type target : {}".format(g.number_of_nodes('target')))

    # print(type(th.from_numpy(features)))

    g.nodes['target'].data['features'] = th.from_numpy(features)

    target_id_to_node = id_to_node[target_node_type]
    id_to_node['target'] = target_id_to_node

    del id_to_node[target_node_type]

    return g, features, target_id_to_node, id_to_node
