import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
import dgl
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def flatten_hete_ndata(out_G, node_key='h', ntypes=
                       ['identifier', 'mod', 'alias', 'arg', 'arguments', 'boolop', 'cmpop', 'comprehension', 'excepthandler', 'expr', 'keyword', 'operator', 'slice', 'stmt', 'unaryop', 'withitem']):
    # print(type(out_G.ndata[node_key]))
    # print(out_G.ndata[node_key].shape)
    if type(out_G.ndata[node_key]) == type(torch.tensor(0)):
        return out_G.ndata[node_key]
    if ntypes is None:
        return torch.cat([out_G.ndata[node_key][k] for k in out_G.ndata[node_key]])
    else:
        return torch.cat([out_G.ndata[node_key][k] for k in ntypes if k in out_G.ndata[node_key]])

class HGTLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout = 0.2, use_norm = False,
                    G_node_dict = None, G_edge_dict = None, device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(HGTLayer, self).__init__()

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.num_types     = num_types
        self.num_relations = num_relations
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        
        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()
        self.use_norm    = use_norm
        self.device = device
        
        for t in range(num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
            
        self.relation_pri   = nn.Parameter(torch.ones(num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))

        self.zero_relation_pri = nn.Parameter(torch.zeros(self.n_heads), requires_grad=False)
        self.zero_relation_att = nn.Parameter(torch.zeros(n_heads, self.d_k, self.d_k), requires_grad=False)
        self.zero_relation_msg = nn.Parameter(torch.zeros(n_heads, self.d_k, self.d_k), requires_grad=False)


        self.skip           = nn.Parameter(torch.ones(num_types))
        self.drop           = nn.Dropout(dropout)

        self.G_node_dict = G_node_dict
        self.G_edge_dict = G_edge_dict
        
        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def edge_attention(self, edges):
        if edges.data['id'].size(0) == 0:
            relation_att = self.zero_relation_att
            relation_msg = self.zero_relation_msg
            relation_pri = self.zero_relation_pri
            # import pdb;pdb.set_trace()
            # return {'a': att, 'v': val}
        else:
            etype = edges.data['id'][0]
            relation_att = self.relation_att[etype]
            relation_pri = self.relation_pri[etype]
            relation_msg = self.relation_msg[etype]
        key   = torch.bmm(edges.src['k'].transpose(1,0), relation_att).transpose(1,0)
        att   = (edges.dst['q'] * key).sum(dim=-1) * relation_pri / self.sqrt_dk
        val   = torch.bmm(edges.src['v'].transpose(1,0), relation_msg).transpose(1,0)
        return {'a': att, 'v': val}
    
    def message_func(self, edges):
        # if edges.data['id'].size(0) == 0:
            # return 
        return {'v': edges.data['v'], 'a': edges.data['a']}
    
    def reduce_func(self, nodes):
        att = F.softmax(nodes.mailbox['a'], dim=1)
        h   = torch.sum(att.unsqueeze(dim = -1) * nodes.mailbox['v'], dim=1)
        return {'t': h.view(-1, self.out_dim)}
        
    def forward(self, G, inp_key, out_key):
        # node_dict, edge_dict = G.node_dict, G.edge_dict
        node_dict, edge_dict = self.G_node_dict, self.G_edge_dict
        for srctype, etype, dsttype in G.canonical_etypes:
            k_linear = self.k_linears[node_dict[srctype]]
            v_linear = self.v_linears[node_dict[srctype]] 
            q_linear = self.q_linears[node_dict[dsttype]]
            
            G.nodes[srctype].data['k'] = k_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[srctype].data['v'] = v_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[dsttype].data['q'] = q_linear(G.nodes[dsttype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            
            # if G.edata['id'][(srctype, etype, dsttype)].shape == torch.Size([0]):
            #     continue
            G.apply_edges(func=self.edge_attention,
                          etype=(srctype, etype, dsttype))
        # G.multi_update_all({etype : (self.message_func, self.reduce_func) \
        #                     for etype in edge_dict}, cross_reducer = 'mean')
        G.multi_update_all({(srctype, etype, dsttype): (self.message_func, self.reduce_func)
                            for srctype, etype, dsttype in G.canonical_etypes}, cross_reducer='mean')
        for ntype in G.ntypes:
            n_id = node_dict[ntype]
            alpha = torch.sigmoid(self.skip[n_id])
            # import pdb;pdb.set_trace()
            if 't' not in G.nodes[ntype].data:
                G.nodes[ntype].data[out_key] = torch.Tensor(0, self.out_dim).to(self.device)
                continue
            trans_out = self.a_linears[n_id](G.nodes[ntype].data['t'])
            trans_out = trans_out * alpha + G.nodes[ntype].data[inp_key] * (1-alpha)
            if self.use_norm:
                G.nodes[ntype].data[out_key] = self.drop(self.norms[n_id](trans_out))
            else:
                G.nodes[ntype].data[out_key] = self.drop(trans_out)
    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)
                
class HGTEncoder(nn.Module):
    def __init__(self, G_node_dict, G_edge_dict, n_inp, n_hid, n_out, n_layers, n_heads, len_src_vocab, device, use_norm = True):
        super(HGTEncoder, self).__init__()
        self.gcs = nn.ModuleList()
        self.n_inp = n_inp
        self.embed_dim = n_inp
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers
        self.G_node_dict = G_node_dict
        self.G_edge_dict = G_edge_dict
        self.len_src_vocab = len_src_vocab
        self.device = device
        self.adapt_ws  = nn.ModuleList()
        for t in range(len(self.G_node_dict)):
            self.adapt_ws.append(nn.Linear(n_inp,   n_hid))
        for _ in range(n_layers):
            self.gcs.append(HGTLayer(n_hid, n_hid, len(self.G_node_dict), len(self.G_edge_dict), n_heads, use_norm = use_norm,
                                     G_node_dict=self.G_node_dict, G_edge_dict=self.G_edge_dict, device=self.device))
        
        self.node_emb = nn.Embedding(self.len_src_vocab, self.embed_dim)
        
        # self.out = nn.Linear(n_hid, n_out)

    # def forward(self, G, out_key = '__ALL__'):
    def forward(self, G):
        # embedding nodes and one-hot edges
        G = G.to(self.device)
        for srctype, etype, dsttype in G.canonical_etypes:
            G.edges[srctype, etype, dsttype].data['id'] = torch.ones(G.number_of_edges(
                (srctype, etype, dsttype)), dtype=torch.long).to(self.device) * self.G_edge_dict[etype]
        for ntype in G.ntypes:
            if 'name' not in G.nodes[ntype].data:
                G.nodes[ntype].data['name'] = torch.LongTensor([]).to(self.device)
            G.nodes[ntype].data['inp'] = self.node_emb(G.nodes[ntype].data['name'].long())
        # import pdb; pdb.set_trace();
        for ntype in G.ntypes:
            n_id = self.G_node_dict[ntype]
            G.nodes[ntype].data['h'] = torch.tanh(self.adapt_ws[n_id](G.nodes[ntype].data['inp']))
        for i in range(self.n_layers):
            self.gcs[i](G, 'h', 'h')
        
        return G
        if out_key == '__ALL__':
            return flatten_hete_ndata(G, 'h')
        return G.nodes[out_key].data['h']
        # return self.out(G.nodes[out_key].data['h'])

    def length_array_to_mask_tensor(self, length_array, cuda=True, valid_entry_has_mask_one=False):
        max_len = max(length_array)
        batch_size = len(length_array)

        mask = np.zeros((batch_size, max_len), dtype=np.uint8)
        for i, seq_len in enumerate(length_array):
            if valid_entry_has_mask_one:
                mask[i][:seq_len] = 1
            else:
                mask[i][seq_len:] = 1

        # mask = torch.ByteTensor(mask)
        mask = torch.BoolTensor(mask)
        return mask.cuda() if cuda else mask
        
    def __repr__(self):
        return '{}(n_inp={}, n_hid={}, n_out={}, n_layers={})'.format(
            self.__class__.__name__, self.n_inp, self.n_hid,
            self.n_out, self.n_layers)
