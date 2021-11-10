import numpy as np
import tensorflow as tf
from scipy import sparse as sp
import spektral
if spektral.__version__ <= '1.0.3':
    from spektral.data.utils import get_spec, to_disjoint, prepend_none
    from spektral.layers.ops import sp_matrix_to_sp_tensor
    from spektral.data.graph import Graph
    from spektral.data.dataset import Dataset
    from spektral.data.loaders import DisjointLoader
else:
    from spektral.data.utils import get_spec, to_disjoint, prepend_none
    from spektral.utils.sparse import sp_matrix_to_sp_tensor
    from spektral.data.graph import Graph
    from spektral.data.dataset import Dataset
    from spektral.data.loaders import DisjointLoader


class ExtendedGraph(Graph):
    """
    Extension of spektral.data.graph.Graph to incorporate graph-level features.

    - `u`, for the graph features;
    """
    def __init__(self, x=None, a=None, e=None, y=None, u=None, **kwargs):
        self.u = u
        super(ExtendedGraph, self).__init__(x=x, a=a, e=e, y=y, **kwargs)
        
    def numpy(self):
        return tuple(ret for ret in [self.x, self.a, self.e, self.y, self.u] if ret is not None)

    def __repr__(self):
        return "Graph(n_nodes={}, n_node_features={}, n_edge_features={}, n_graph_features={}, n_labels={})".format(
            self.n_nodes, self.n_node_features, self.n_edge_features, self.n_graph_features, self.n_labels
        )

    @property
    def n_graph_features(self):
        if self.u is not None:
            shp = np.shape(self.u)
            return 1 if len(shp) == 0 else shp[-1]
        else:
            return None 


class ExtendedDataset(Dataset):
    """
    Extended to incorporate graph-level features.
    """
    @property
    def n_graph_features(self):
        if len(self.graphs) >= 1:
            return self.graphs[0].n_graph_features
        else:
            return None

    @property
    def signature(self):
        """
        Extended to incorporate graph-level features.

        This property computes the signature of the dataset, which can be
        passed to `spektral.data.utils.to_tf_signature(signature)` to compute
        the TensorFlow signature. You can safely ignore this property unless
        you are creating a custom `Loader`.

        A signature consist of the TensorFlow TypeSpec, shape, and dtype of
        all characteristic matrices of the graphs in the Dataset. This is
        returned as a dictionary of dictionaries, with keys `x`, `a`, `e`, and
        `y` for the four main data matrices.

        Each sub-dictionary will have keys `spec`, `shape` and `dtype`.
        """
        signature = {}
        graph = self.graphs[0]  # This is always non-empty
        if graph.x is not None:
            signature['x'] = dict()
            signature['x']['spec'] = get_spec(graph.x)
            signature['x']['shape'] = (None, self.n_node_features)
            signature['x']['dtype'] = tf.as_dtype(graph.x.dtype)
        if graph.a is not None:
            signature['a'] = dict()
            signature['a']['spec'] = get_spec(graph.a)
            signature['a']['shape'] = (None, None)
            signature['a']['dtype'] = tf.as_dtype(graph.a.dtype)
        if graph.e is not None:
            signature['e'] = dict()
            signature['e']['spec'] = get_spec(graph.e)
            signature['e']['shape'] = (None, self.n_edge_features)
            signature['e']['dtype'] = tf.as_dtype(graph.e.dtype)
        if graph.y is not None:
            signature['y'] = dict()
            signature['y']['spec'] = get_spec(graph.y)
            signature['y']['shape'] = (self.n_labels,)
            signature['y']['dtype'] = tf.as_dtype(np.array(graph.y).dtype)
        if graph.u is not None:
            signature['u'] = dict()
            signature['u']['spec'] = get_spec(graph.u)
            signature['u']['shape'] = (self.n_graph_features,)
            signature['u']['dtype'] = tf.as_dtype(graph.u.dtype)
        return signature

def extended_to_tf_signature(signature):
    """
    Extended to incorporate graph-level features.

    Converts a Dataset signature to a TensorFlow signature.
    :param signature: a Dataset signature.
    :return: a TensorFlow signature.
    """
    output = []
    keys = ["x", "a", "e", "i", "u"]
    for k in keys:
        if k in signature:
            shape = signature[k]["shape"]
            dtype = signature[k]["dtype"]
            spec = signature[k]["spec"]
            output.append(spec(shape, dtype))
    output = tuple(output)
    if "y" in signature:
        shape = signature["y"]["shape"]
        dtype = signature["y"]["dtype"]
        spec = signature["y"]["spec"]
        output = (output, spec(shape, dtype))

    return output


class ExtendedDisjointLoader(DisjointLoader):
    """
    Extended to incorporate graph-level features.
    """
    def collate(self, batch):
        if spektral.__version__ <= '1.0.3':
            packed = self.pack(batch, return_dict=True)
        else:
            packed = self.pack(batch)
        y = None
        if "y" in self.dataset.signature:
            y = packed.pop("y_list")
            y = np.vstack(y) if self.node_level else np.array(y)

        u = None
        if "u" in self.dataset.signature:
            u = packed.pop("u_list")
            u = np.array(u)

        output = to_disjoint(**packed)

        # Sparse matrices to SparseTensors
        output = list(output)
        for i in range(len(output)):
            if sp.issparse(output[i]):
                output[i] = sp_matrix_to_sp_tensor(output[i])
        output = tuple(output)

        if u is not None:
            output += (u,)

        if y is None:
            return output
        else:
            return output, y

    def tf_signature(self):
        """
        Adjacency matrix has shape [n_nodes, n_nodes]
        Node features have shape [n_nodes, n_node_features]
        Edge features have shape [n_edges, n_edge_features]
        Targets have shape [..., n_labels]
        """
        signature = self.dataset.signature
        if "y" in signature:
            signature["y"]["shape"] = prepend_none(signature["y"]["shape"])
        if "u" in signature:
            signature["u"]["shape"] = prepend_none(signature["u"]["shape"])
        if "a" in signature:
            signature["a"]["spec"] = tf.SparseTensorSpec

        signature["i"] = dict()
        signature["i"]["spec"] = tf.TensorSpec
        signature["i"]["shape"] = (None,)
        signature["i"]["dtype"] = tf.as_dtype(tf.int64)

        return extended_to_tf_signature(signature)