from torch_geometric.data import Data


class PairedData(Data):
    def __init__(self, edge_index_s, edge_attr_s, edge_flag_s, x_s, adj_s,
                 edge_index_t, edge_attr_t, edge_flag_t, x_t, adj_t, y):
        super(PairedData, self).__init__()
        self.edge_index_s = edge_index_s
        self.edge_attr_s = edge_attr_s
        self.edge_flag_s = edge_flag_s
        self.adj_s = adj_s
        self.x_s = x_s
        self.edge_index_t = edge_index_t
        self.edge_attr_t = edge_attr_t
        self.edge_flag_t = edge_flag_t
        self.adj_t = adj_t
        self.x_t = x_t
        self.y = y

    def __inc__(self, key, value):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value)
