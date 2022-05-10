import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dgl
import torch
import numpy as np
import pickle
import tqdm

import re

from java_parser.treesitter2anytree import *
LANG = 'java'
GEN_AST = True
CST_NUM = {
    'wrap_num': 4,
    'NOTUSE_loc': 2,
    'LEFT_loc': 3,
    'END_loc': -1
}
AST_NUM = {
    'wrap_num': 2,
    'NOTUSE_loc': 1,
    'LEFT_loc': -1,
    'END_loc': None
}
if GEN_AST:
    CHOSEN_NUM = AST_NUM
else:
    CHOSEN_NUM = CST_NUM

REMOVE_FIRST_COMMENT = False
WRAP_CODE = False

with open(os.path.join(os.path.dirname(__file__),'java_asdl.pkl'),'rb') as f:
    java_asdl_dict = pickle.load(f)


def get_ast_edge_type(parent_node, child_node):
    return java_asdl_dict.get_edge_type(parent_node,child_node)


def get_ast_dict(code):
    REPLACE_FUNC_NAME = True
    REPLACE_FUNC_NAME_STR = 'NOTUSE'
    if WRAP_CODE:
        parse_rst = parse_java_example_with_wrap(code,False)
    else:
        parse_rst = parse_java_example_without_wrap(code,False)
    assert(parse_rst[0])
    parse_tree = parse_rst[-1]
    # AnyNode(index=7, is_ast_node=True, name='public final', type='modifiers') ->
    # {'name': node,'type':identifier_type,'children':[], 'toParentField':parent_field,'is_terminal':False}
    if REMOVE_FIRST_COMMENT:
        if parse_rst[-1].children[0].type == 'comment':
            parse_tree = parse_rst[-1].children[1]
        else:
            parse_tree = parse_rst[-1].children[0]
        assert(parse_tree.type == 'method_declaration')
    if REPLACE_FUNC_NAME:
        first_id_node = [e for e in parse_tree.children if e.type == 'identifier']
        if len(first_id_node) != 0:
            first_id_node[0].name = REPLACE_FUNC_NAME_STR
    
    def to_dict(node, add_list, parent_field = 'ast'):
        if len(node.children) == 0:
            add_list.append({'name': node.name, 'type': node.type, 'children': [], 'toParentField': parent_field,'is_terminal':True})
            return

        this_dict = {'name': node.type,'type':node.type,'children':[], 'toParentField':parent_field,'is_terminal':False}
        add_list.append(this_dict)
        for child in node.children:
            parent_field = get_ast_edge_type(node, child)
            to_dict(child, this_dict['children'], parent_field)
        
    rst_list = []
    to_dict(parse_tree, rst_list)
    return rst_list[0]


def parse_java_example_without_wrap(code,DEBUG=False):
    # path_prefix = 'program|class_declaration|class_body'
    # wrap_code = "public class NOTUSE {"
    
    # wrap_num = CHOSEN_NUM['wrap_num']
    # NOTUSE_loc = CHOSEN_NUM['NOTUSE_loc']
    # LEFT_loc = CHOSEN_NUM['LEFT_loc']
    # END_loc = CHOSEN_NUM['END_loc']


    # code = wrap_code + code + "}"
    leaf_nodes, paths = get_leaf_node_list(code,
                                           generate_ast=GEN_AST, language=LANG)
    py_rst_tree, leaf_node = myparse(code,
                                           generate_ast=GEN_AST, language=LANG)
    
    if DEBUG:
        # print(' '.join(leaf_nodes))
        return py_rst_tree, leaf_node
    for e in paths:
        if 'ERROR' in e:
            if DEBUG:
                print(1)
            return False, leaf_nodes, paths
    
    paths = paths
    leaf_nodes = leaf_nodes
    # import pdb;pdb.set_trace()
    return True, leaf_nodes, paths, py_rst_tree



def parse_java_example_with_wrap(code,DEBUG=False):
    path_prefix = 'program|class_declaration|class_body'
    wrap_code = "public class NOTUSE {"
    
    wrap_num = CHOSEN_NUM['wrap_num']
    NOTUSE_loc = CHOSEN_NUM['NOTUSE_loc']
    LEFT_loc = CHOSEN_NUM['LEFT_loc']
    END_loc = CHOSEN_NUM['END_loc']


    code = wrap_code + code + "}"
    leaf_nodes, paths = get_leaf_node_list(code,
                                           generate_ast=GEN_AST, language=LANG)
    py_rst_tree, leaf_node = myparse(code,
                                           generate_ast=GEN_AST, language=LANG)
    
    if DEBUG:
        # print(' '.join(leaf_nodes))
        return py_rst_tree, leaf_node
    for e in paths:
        if 'ERROR' in e:
            if DEBUG:
                print(1)
            return False, leaf_nodes[wrap_num:END_loc], paths[wrap_num:END_loc]
    if GEN_AST:
        loc_bool = (leaf_nodes[NOTUSE_loc] == 'NOTUSE')
    else:
        loc_bool = (leaf_nodes[NOTUSE_loc] == 'NOTUSE' and leaf_nodes[LEFT_loc] == '{')
    if loc_bool:
        paths = paths[wrap_num:END_loc]
        leaf_nodes = leaf_nodes[wrap_num:END_loc]
        remove_prefix_flag = False
        for p in paths:
            if not p.startswith(path_prefix):
                remove_prefix_flag = True
                break
        if remove_prefix_flag:
            if DEBUG:
                print(2)
            return False, leaf_nodes, paths
        else:
            paths = ['program' + e[len(path_prefix):] for e in paths]
            return True, leaf_nodes, paths, py_rst_tree.children[0].children[-1]
    if DEBUG:
        print(3)
    return False, leaf_nodes[wrap_num:END_loc], paths[wrap_num:END_loc]




def heterograph_graph_parser(py_code, return_graph= True):
    ast_tree = get_ast_dict(py_code)
    ast_tree['toleftSiblingIdx'] = -1

    edge_dict = {}
    # ast_parent2child, ast_child2parent, next_token, last_tokens
    each_node_dict = {}

    def check_each_node_dict(typ, idx):
        if typ not in each_node_dict:
            each_node_dict[typ] = []
        if idx not in each_node_dict[typ]:
            each_node_dict[typ].append(idx)
        return each_node_dict[typ].index(idx)



    nodes = {'all_node': [],'identifier_node':[]}
    # all_identifiers = []

    def find_left_sibling_idx(parent_idx, this_sibling_idx):
        all_siblings = [e for e in nodes['all_node'] if e['parent_idx'] == parent_idx]
        left_sibling = [e for e in all_siblings if e['sibling_idx'] == this_sibling_idx - 1]
        if len(left_sibling) == 0:
            return -1
        else:
            return left_sibling[0]['idx']




    def visit_tree(node, parent_idx = -1, sibling_idx = 0):
        now_idx = len(nodes['all_node'])

        nodes['all_node'].append({'idx': now_idx,
                        'name': node['name'],
                        'type': node['type'],
                        'toParentField': node['toParentField'],
                        'parent_idx': parent_idx,
                        'sibling_idx': sibling_idx
                                  })
        if 'identifier' in node['type']:
            nodes['identifier_node'].append(now_idx)

        now_parent_idx = now_idx

        if parent_idx != -1:
            # 加父子边

            edge_type = (
                str(nodes['all_node'][parent_idx]['type']),
                str(nodes['all_node'][now_idx]['toParentField']),
                str(nodes['all_node'][now_idx]['type'])
                )
            if edge_type not in edge_dict:
                edge_dict[edge_type] = [[], []]



            edge_dict[edge_type][0].append(check_each_node_dict(str(nodes['all_node'][parent_idx]['type']), parent_idx))
            edge_dict[edge_type][1].append(check_each_node_dict(str(nodes['all_node'][now_idx]['type']), now_idx))

            edge_type = (
                str(nodes['all_node'][now_idx]['type']),
                str(nodes['all_node'][now_idx]['toParentField'] + '_reverse'),
                str(nodes['all_node'][parent_idx]['type'])
                )
            if edge_type not in edge_dict:
                edge_dict[edge_type] = [[], []]


            edge_dict[edge_type][0].append(check_each_node_dict(str(nodes['all_node'][now_idx]['type']), now_idx))    
            edge_dict[edge_type][1].append(check_each_node_dict(str(nodes['all_node'][parent_idx]['type']), parent_idx))

        left_sibling_idx = find_left_sibling_idx(parent_idx, sibling_idx)
        # left_sibling_idx = -1
        if left_sibling_idx != -1:
            # 加兄弟边
            edge_type = (
                str(nodes['all_node'][left_sibling_idx]['type']),
                str('left_sibling_to_right'),
                str(nodes['all_node'][now_idx]['type'])
            )
            if edge_type not in edge_dict:
                edge_dict[edge_type] = [[], []]

            edge_dict[edge_type][0].append(check_each_node_dict(str(nodes['all_node'][left_sibling_idx]['type']), left_sibling_idx))
            edge_dict[edge_type][1].append(check_each_node_dict(str(nodes['all_node'][now_idx]['type']), now_idx))

            edge_type = (
                str(nodes['all_node'][now_idx]['type']),
                str('right_sibling_to_left'),
                str(nodes['all_node'][left_sibling_idx]['type'])
            )
            if edge_type not in edge_dict:
                edge_dict[edge_type] = [[], []]
            edge_dict[edge_type][0].append(check_each_node_dict(str(nodes['all_node'][now_idx]['type']), now_idx))
            edge_dict[edge_type][1].append(check_each_node_dict(str(nodes['all_node'][left_sibling_idx]['type']), left_sibling_idx))

        now_idx += 1

        children_node = node['children']
        for c_i in range(len(children_node)):
            c = children_node[c_i]
            visit_tree(c, now_parent_idx,c_i)

    visit_tree(ast_tree)
    identifier_node_len = len(nodes['identifier_node'])
    # identifier_node_len = 1
    if identifier_node_len >= 2:
        for i in range(len(nodes['identifier_node']) - 1):
            this_one = nodes['identifier_node'][i]
            next_one = nodes['identifier_node'][i + 1]
            this_type = nodes['all_node'][this_one]['type']
            next_type = nodes['all_node'][next_one]['type']

            edge_type = (
                str(this_type),
                str('next_token'),
                str(next_type)
                )

            if edge_type not in edge_dict:
                edge_dict[edge_type] = [[], []]

            edge_dict[edge_type][0].append(check_each_node_dict(str(nodes['all_node'][this_one]['type']), this_one))
            edge_dict[edge_type][1].append(check_each_node_dict(str(nodes['all_node'][next_one]['type']), next_one))

            edge_type = (
                str(next_type),
                str('last_token'),
                str(this_type)
                )

            if edge_type not in edge_dict:
                edge_dict[edge_type] = [[], []]

            edge_dict[edge_type][0].append(check_each_node_dict(str(nodes['all_node'][next_one]['type']), next_one))
            edge_dict[edge_type][1].append(check_each_node_dict(str(nodes['all_node'][this_one]['type']), this_one))

    new_dict = {}
    for k in edge_dict:
        v = edge_dict[k]
        new_dict[k] = (np.array(v[0]), np.array(v[1]))

    # return nodes,edge_dict,new_dict,dgl.heterograph(new_dict),each_node_dict
    if return_graph == False:
        return nodes, new_dict, each_node_dict
    out_nodes = nodes
    out_g = dgl.heterograph(new_dict)
    for ntype in out_g.ntypes:
        out_g.nodes[ntype].data['idx'] = torch.tensor(each_node_dict[ntype])

    return nodes, out_g

############################################################################
# copy from /home/username/workspace/code2seq-master/ogb_python/extract.py
############################################################################

# def __delim_name(name):
#     if name in {METHOD_NAME, NUM}:
#         return name

#     def camel_case_split(identifier):
#         matches = re.finditer(
#             '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)',
#             identifier,
#         )
#         return [m.group(0) for m in matches]

#     blocks = []
#     for underscore_block in name.split('_'):
#         blocks.extend(camel_case_split(underscore_block))

#     return '|'.join(block.lower() for block in blocks)


def tokenizer(name):
    def camel_case_split(identifier):
        matches = re.finditer(
            '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)',
            identifier,
        )
        return [m.group(0) for m in matches]
    blocks = []
    for underscore_block in re.split(' |_', name):
        blocks.extend(camel_case_split(underscore_block))
    return blocks


def heterograph_graph_parser_subtoken(py_code, return_graph= True, share_subtoken = True):
    ast_tree = get_ast_dict(py_code)
    ast_tree['toleftSiblingIdx'] = -1

    edge_dict = {}
    # ast_parent2child, ast_child2parent, next_token, last_tokens
    each_node_dict = {}

    def check_each_node_dict(typ, idx):
        if typ not in each_node_dict:
            each_node_dict[typ] = []
        if idx not in each_node_dict[typ]:
            each_node_dict[typ].append(idx)
        return each_node_dict[typ].index(idx)



    nodes = {'all_node': [],'identifier_node':[],'subtoken_node':[]}
    # all_identifiers = []

    def find_left_sibling_idx(parent_idx, this_sibling_idx):
        all_siblings = [e for e in nodes['all_node'] if e['parent_idx'] == parent_idx]
        left_sibling = [e for e in all_siblings if e['sibling_idx'] == this_sibling_idx - 1]
        if len(left_sibling) == 0:
            return -1
        else:
            return left_sibling[0]['idx']

    def get_subtoken_node_idx(subtoken, identifier_idx = -1,subtoken_sibling_idx = 0,share = True):
        if share == True:
            subtoken_nodes = [e for e in nodes['all_node'] if e['name'] == subtoken and e['type'] == 'subtoken']
            if len(subtoken_nodes) != 0:
                subtoken_idx = subtoken_nodes[0]['idx']
                if identifier_idx != -1:
                    nodes['all_node'][subtoken_idx]['identifier_idx'].append(identifier_idx)
                if subtoken_sibling_idx != -1:
                    nodes['all_node'][subtoken_idx]['subtoken_sibling_idx'].append(subtoken_sibling_idx)
                return subtoken_idx

        subtoken_idx = len(nodes['all_node'])
        nodes['all_node'].append({'idx': subtoken_idx,
                'name': subtoken,
                'type': 'subtoken',
                'toParentField': 'toSubtoken',
                'parent_idx': identifier_idx,
                'sibling_idx': subtoken_sibling_idx,
                'identifier_idx': [identifier_idx],
                'subtoken_sibling_idx': [subtoken_sibling_idx]
                                  })
        nodes['subtoken_node'].append(subtoken_idx)
        return subtoken_idx



    def visit_tree(node, parent_idx = -1, sibling_idx = 0):
        now_idx = len(nodes['all_node'])

        nodes['all_node'].append({'idx': now_idx,
                        'name': node['name'],
                        'type': node['type'],
                        'toParentField': node['toParentField'],
                        'parent_idx': parent_idx,
                        'sibling_idx': sibling_idx
                                  })
        if 'identifier' in node['type']:
            nodes['identifier_node'].append(now_idx)
            identifier_idx = now_idx
            subtoken_sibling_idx = 0

            # 增加subtoken边
            subtokens = tokenizer(node['name'])
            for subtoken in subtokens:
                subtoken_idx = get_subtoken_node_idx(
                    subtoken, identifier_idx, subtoken_sibling_idx, share_subtoken)
                # subtoken_idx = len(nodes['all_node'])
                # nodes['all_node'].append({'idx':subtoken_idx,
                #     'name':subtoken,
                #     'type':'subtoken',
                #     'toParentField':'toSubtoken',
                #     'parent_idx':identifier_idx,
                #     'sibling_idx':subtoken_sibling_idx
                #     })
                # nodes['subtoken_node'].append(subtoken_idx)
                subtoken_sibling_idx += 1

                # 加父子边
                if identifier_idx != -1:
                    # 加父子边
                    edge_type = (
                        str(nodes['all_node'][identifier_idx]['type']),
                        str(nodes['all_node'][subtoken_idx]['toParentField']),
                        str(nodes['all_node'][subtoken_idx]['type'])
                        )
                    if edge_type not in edge_dict:
                        edge_dict[edge_type] = [[], []]
                    edge_dict[edge_type][0].append(check_each_node_dict(str(nodes['all_node'][identifier_idx]['type']), identifier_idx))
                    edge_dict[edge_type][1].append(check_each_node_dict(str(nodes['all_node'][subtoken_idx]['type']), subtoken_idx))
                    edge_type = (
                        str(nodes['all_node'][subtoken_idx]['type']),
                        str(nodes['all_node'][subtoken_idx]['toParentField'] + '_reverse'),
                        str(nodes['all_node'][identifier_idx]['type'])
                        )
                    if edge_type not in edge_dict:
                        edge_dict[edge_type] = [[], []]
                    edge_dict[edge_type][0].append(check_each_node_dict(str(nodes['all_node'][subtoken_idx]['type']), subtoken_idx))    
                    edge_dict[edge_type][1].append(check_each_node_dict(str(nodes['all_node'][identifier_idx]['type']), identifier_idx))


        now_parent_idx = now_idx

        if parent_idx != -1:
            # 加父子边

            edge_type = (
                str(nodes['all_node'][parent_idx]['type']),
                str(nodes['all_node'][now_idx]['toParentField']),
                str(nodes['all_node'][now_idx]['type'])
                )
            if edge_type not in edge_dict:
                edge_dict[edge_type] = [[], []]



            edge_dict[edge_type][0].append(check_each_node_dict(str(nodes['all_node'][parent_idx]['type']), parent_idx))
            edge_dict[edge_type][1].append(check_each_node_dict(str(nodes['all_node'][now_idx]['type']), now_idx))

            edge_type = (
                str(nodes['all_node'][now_idx]['type']),
                str(nodes['all_node'][now_idx]['toParentField'] + '_reverse'),
                str(nodes['all_node'][parent_idx]['type'])
                )
            if edge_type not in edge_dict:
                edge_dict[edge_type] = [[], []]


            edge_dict[edge_type][0].append(check_each_node_dict(str(nodes['all_node'][now_idx]['type']), now_idx))    
            edge_dict[edge_type][1].append(check_each_node_dict(str(nodes['all_node'][parent_idx]['type']), parent_idx))

        left_sibling_idx = find_left_sibling_idx(parent_idx, sibling_idx)
        # left_sibling_idx = -1
        if left_sibling_idx != -1:
            # 加兄弟边
            edge_type = (
                str(nodes['all_node'][left_sibling_idx]['type']),
                str('left_sibling_to_right'),
                str(nodes['all_node'][now_idx]['type'])
            )
            if edge_type not in edge_dict:
                edge_dict[edge_type] = [[], []]

            edge_dict[edge_type][0].append(check_each_node_dict(str(nodes['all_node'][left_sibling_idx]['type']), left_sibling_idx))
            edge_dict[edge_type][1].append(check_each_node_dict(str(nodes['all_node'][now_idx]['type']), now_idx))

            edge_type = (
                str(nodes['all_node'][now_idx]['type']),
                str('right_sibling_to_left'),
                str(nodes['all_node'][left_sibling_idx]['type'])
            )
            if edge_type not in edge_dict:
                edge_dict[edge_type] = [[], []]
            edge_dict[edge_type][0].append(check_each_node_dict(str(nodes['all_node'][now_idx]['type']), now_idx))
            edge_dict[edge_type][1].append(check_each_node_dict(str(nodes['all_node'][left_sibling_idx]['type']), left_sibling_idx))

        now_idx += 1

        children_node = node['children']
        for c_i in range(len(children_node)):
            c = children_node[c_i]
            visit_tree(c, now_parent_idx,c_i)

    visit_tree(ast_tree)
    identifier_node_len = len(nodes['identifier_node'])
    # identifier_node_len = 1
    if identifier_node_len >= 2:
        for i in range(len(nodes['identifier_node']) - 1):
            this_one = nodes['identifier_node'][i]
            next_one = nodes['identifier_node'][i + 1]
            this_type = nodes['all_node'][this_one]['type']
            next_type = nodes['all_node'][next_one]['type']

            edge_type = (
                str(this_type),
                str('next_token'),
                str(next_type)
                )

            if edge_type not in edge_dict:
                edge_dict[edge_type] = [[], []]

            edge_dict[edge_type][0].append(check_each_node_dict(str(this_type), this_one))
            edge_dict[edge_type][1].append(check_each_node_dict(str(next_type), next_one))

            edge_type = (
                str(next_type),
                str('last_token'),
                str(this_type)
                )

            if edge_type not in edge_dict:
                edge_dict[edge_type] = [[], []]

            edge_dict[edge_type][0].append(check_each_node_dict(str(next_type), next_one))
            edge_dict[edge_type][1].append(check_each_node_dict(str(this_type), this_one))

    new_dict = {}
    for k in edge_dict:
        v = edge_dict[k]
        new_dict[k] = (np.array(v[0]), np.array(v[1]))

    # return nodes,edge_dict,new_dict,dgl.heterograph(new_dict),each_node_dict
    if return_graph == False:
        return nodes, new_dict, each_node_dict
    out_nodes = nodes
    out_g = dgl.heterograph(new_dict)
    for ntype in out_g.ntypes:
        out_g.nodes[ntype].data['idx'] = torch.tensor(each_node_dict[ntype])

    return nodes, out_g


def dict2graph(new_dict, each_node_dict):
    out_g = dgl.heterograph(new_dict)
    for ntype in out_g.ntypes:
        out_g.nodes[ntype].data['idx'] = torch.tensor(each_node_dict[ntype])
    return out_g