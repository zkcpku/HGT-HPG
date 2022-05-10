# coding=utf-8
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dgl
import torch
import numpy as np
import pickle
import tqdm

import re
import ast
from python_parser.asdl import *
# import astor

# bpe
from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base',do_lower_case=False)




asdl_text = open(os.path.join(os.path.dirname(__file__), 'py3_asdl.simplified.txt')).read()
grammar = ASDLGrammar.from_text(asdl_text)
mod_type = ASDLType('mod')
identifier_type = ASDLType('identifier')

# AST_type = []

# [ASDLCompositeType(alias),
#  ASDLCompositeType(arg),
#  ASDLCompositeType(arguments),
#  ASDLCompositeType(boolop),
#  ASDLPrimitiveType(bytes),
#  ASDLCompositeType(cmpop),
#  ASDLCompositeType(comprehension),
#  ASDLCompositeType(excepthandler),
#  ASDLCompositeType(expr),
#  ASDLCompositeType(expr_context),
#  ASDLPrimitiveType(identifier),
#  ASDLPrimitiveType(int),
#  ASDLCompositeType(keyword),
#  ASDLCompositeType(mod),
#  ASDLPrimitiveType(object),
#  ASDLCompositeType(operator),
#  ASDLPrimitiveType(singleton),
#  ASDLCompositeType(slice),
#  ASDLCompositeType(stmt),
#  ASDLPrimitiveType(string),
#  ASDLCompositeType(unaryop),
#  ASDLCompositeType(withitem)]


def python_ast_to_asdl_ast(py_ast_node, grammar):
    # node should be composite
    py_node_name = type(py_ast_node).__name__
    # assert py_node_name.startswith('_ast.')

    production = grammar.get_prod_by_ctr_name(py_node_name)

    fields = []
    for field in production.fields:
        field_value = getattr(py_ast_node, field.name)
        asdl_field = RealizedField(field)
        if field.cardinality == 'single' or field.cardinality == 'optional':
            if field_value is not None:  # sometimes it could be 0
                if grammar.is_composite_type(field.type):
                    child_node = python_ast_to_asdl_ast(field_value, grammar)
                    asdl_field.add_value(child_node)
                else:
                    asdl_field.add_value(str(field_value))
        # field with multiple cardinality
        elif field_value is not None:
                if grammar.is_composite_type(field.type):
                    for val in field_value:
                        child_node = python_ast_to_asdl_ast(val, grammar)
                        asdl_field.add_value(child_node)
                else:
                    for val in field_value:
                        asdl_field.add_value(str(val))

        fields.append(asdl_field)

    asdl_node = AbstractSyntaxTree(production, realized_fields=fields)

    return asdl_node


def asdl_ast_to_python_ast(asdl_ast_node, grammar):
    py_node_type = getattr(sys.modules['ast'], asdl_ast_node.production.constructor.name)
    py_ast_node = py_node_type()

    for field in asdl_ast_node.fields:
        # for composite node
        field_value = None
        if grammar.is_composite_type(field.type):
            if field.value and field.cardinality == 'multiple':
                field_value = []
                for val in field.value:
                    node = asdl_ast_to_python_ast(val, grammar)
                    field_value.append(node)
            elif field.value and field.cardinality in ('single', 'optional'):
                field_value = asdl_ast_to_python_ast(field.value, grammar)
        else:
            # for primitive node, note that primitive field may have `None` value
            if field.value is not None:
                if field.type.name == 'object':
                    if '.' in field.value or 'e' in field.value:
                        field_value = float(field.value)
                    elif isint(field.value):
                        field_value = int(field.value)
                    else:
                        raise ValueError('cannot convert [%s] to float or int' % field.value)
                elif field.type.name == 'int':
                    field_value = int(field.value)
                else:
                    field_value = field.value

            # FIXME: hack! if int? is missing value in ImportFrom(identifier? module, alias* names, int? level), fill with 0
            elif field.name == 'level':
                field_value = 0

        # must set unused fields to default value...
        if field_value is None and field.cardinality == 'multiple':
            field_value = list()

        setattr(py_ast_node, field.name, field_value)

    return py_ast_node

def to_dict(node,add_list, parent_field = 'body'):
    if type(node) == type(" "):
        add_list.append({'name':node,'type':identifier_type,'children':[], 'toParentField':parent_field})
        return
    this_dict = {'name':node.production.constructor.name,'type':node.production.type,'children':[], 'toParentField':parent_field}
    add_list.append(this_dict)
    for field in node.fields:
        if field.value is not None:
            for val_node in field.as_value_list:
                if isinstance(field.type, ASDLCompositeType):
                    to_dict(val_node,this_dict['children'], parent_field = field.name)
                else:
                    to_dict(str(val_node),this_dict['children'], parent_field = field.name)    

def asdl2dict(node):
    rst_list = []
    to_dict(node,rst_list)
    return rst_list[0]

def get_ast_dict(py_code):
    py_ast = ast.parse(py_code)
    # asdl_ast = python_ast_to_asdl_ast(py_ast.body[0], grammar)
    asdl_asts = [python_ast_to_asdl_ast(e, grammar) for e in py_ast.body]
    # rst = asdl2dict(asdl_ast)
    rsts = [asdl2dict(e) for e in asdl_asts]

    return {'name':'module','type':mod_type,'children':rsts, 'toParentField': None}


def nameAddType_and_collapsed(d):
    d['name'] += (' | ' + str(d['type']) + ' | ' + str(d['toParentField']))
    d['collapsed'] = False
    for e in d['children']:
        nameAddType_and_collapsed(e)
    return d


def show_ast_tree(py_code):
    print(grammar.types)
    assert(len(set(grammar.types)) == len(grammar.types))

    # print(get_ast_dict(py_code))
    # print(name_add_type(get_ast_dict(py_code)))
    from pyecharts.charts import Tree
    c = Tree().add("",[nameAddType_and_collapsed(get_ast_dict(py_code))])
    c.render()



def graph_parser(py_code):
    py_ast = ast.parse(py_code)
    asdl_asts = [python_ast_to_asdl_ast(e, grammar) for e in py_ast.body]
    rsts = [asdl2dict(e) for e in asdl_asts]
    ast_tree = {'name':'module','type':mod_type,'children':rsts}
    # return ast_tree

    # now_idx = 0
    nodes = {'all_node':[],'identifier_node':[]}
    edges = {'ast_child':[],'ast_parent':[]}
    # all_identifiers = []

    def visit_tree(node,parent_idx = -1):
        now_idx = len(nodes['all_node'])
        if parent_idx != -1:
            edges['ast_child'].append((parent_idx,now_idx))
            edges['ast_parent'].append((now_idx,parent_idx))

        nodes['all_node'].append({'idx':now_idx,
                        'name':node['name'],
                        'type':node['type']
                        })
        if node['type'] == identifier_type:
            nodes['identifier_node'].append(now_idx)

        now_parent_idx = now_idx
        now_idx += 1
        
        children_node = node['children']
        for c in children_node:
            visit_tree(c,now_parent_idx)

    visit_tree(ast_tree)

    # TODO: 加上next_token

    return nodes,edges


def heterograph_graph_parser(py_code, return_graph = True):
    
    py_ast = ast.parse(py_code)
    asdl_asts = [python_ast_to_asdl_ast(e, grammar) for e in py_ast.body]
    rsts = [asdl2dict(e) for e in asdl_asts]
    ast_tree = {'name':'module','type':mod_type,'children':rsts,'toParentField': None, 'toleftSiblingIdx':-1}

    edge_dict = {}
    # ast_parent2child, ast_child2parent, next_token, last_tokens
    each_node_dict = {}

    def check_each_node_dict(typ, idx):
        if typ not in each_node_dict:
            each_node_dict[typ] = []
        if idx not in each_node_dict[typ]:
            each_node_dict[typ].append(idx)
        return each_node_dict[typ].index(idx)



    nodes = {'all_node':[],'identifier_node':[]}
    # all_identifiers = []
    def find_left_sibling_idx(parent_idx, this_sibling_idx):
        all_siblings = [e for e in nodes['all_node'] if e['parent_idx'] == parent_idx]
        left_sibling = [e for e in all_siblings if e['sibling_idx'] == this_sibling_idx - 1]
        if len(left_sibling) == 0:
            return -1
        else:
            return left_sibling[0]['idx']




    def visit_tree(node,parent_idx = -1, sibling_idx = 0):
        now_idx = len(nodes['all_node'])

        nodes['all_node'].append({'idx':now_idx,
                        'name':node['name'],
                        'type':node['type'],
                        'toParentField':node['toParentField'],
                        'parent_idx':parent_idx,
                        'sibling_idx':sibling_idx
                        })
        if node['type'] == identifier_type:
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
                edge_dict[edge_type] = [[],[]]



            edge_dict[edge_type][0].append(check_each_node_dict(str(nodes['all_node'][parent_idx]['type']),parent_idx))
            edge_dict[edge_type][1].append(check_each_node_dict(str(nodes['all_node'][now_idx]['type']),now_idx))


            edge_type = (
                        str(nodes['all_node'][now_idx]['type']),
                        str(nodes['all_node'][now_idx]['toParentField'] + '_reverse'),
                        str(nodes['all_node'][parent_idx]['type'])
                        )
            if edge_type not in edge_dict:
                edge_dict[edge_type] = [[],[]]


            edge_dict[edge_type][0].append(check_each_node_dict(str(nodes['all_node'][now_idx]['type']),now_idx))    
            edge_dict[edge_type][1].append(check_each_node_dict(str(nodes['all_node'][parent_idx]['type']),parent_idx))
        
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
                edge_dict[edge_type]=[[], []]

            edge_dict[edge_type][0].append(check_each_node_dict(str(nodes['all_node'][left_sibling_idx]['type']), left_sibling_idx))
            edge_dict[edge_type][1].append(check_each_node_dict(str(nodes['all_node'][now_idx]['type']), now_idx))
            
            edge_type = (
                str(nodes['all_node'][now_idx]['type']),
                str('right_sibling_to_left'),
                str(nodes['all_node'][left_sibling_idx]['type'])
            )
            if edge_type not in edge_dict:
                edge_dict[edge_type]=[[], []]
            edge_dict[edge_type][0].append(check_each_node_dict(str(nodes['all_node'][now_idx]['type']), now_idx))
            edge_dict[edge_type][1].append(check_each_node_dict(str(nodes['all_node'][left_sibling_idx]['type']), left_sibling_idx))

        now_idx += 1
        
        children_node = node['children']
        for c_i in range(len(children_node)):
            c = children_node[c_i]
            visit_tree(c,now_parent_idx,c_i)

    visit_tree(ast_tree)
    identifier_node_len = len(nodes['identifier_node'])
    # identifier_node_len = 1
    if identifier_node_len >= 2:
        for i in range(len(nodes['identifier_node']) - 1):
            this_one = nodes['identifier_node'][i]
            next_one = nodes['identifier_node'][i + 1]

            edge_type = (
                        str(identifier_type),
                        str('next_token'),
                        str(identifier_type)
                        )

            if edge_type not in edge_dict:
                edge_dict[edge_type] = [[],[]]

            edge_dict[edge_type][0].append(check_each_node_dict(str(identifier_type),this_one))
            edge_dict[edge_type][1].append(check_each_node_dict(str(identifier_type),next_one))

            edge_type = (
                        str(identifier_type),
                        str('last_token'),
                        str(identifier_type)
                        )

            if edge_type not in edge_dict:
                edge_dict[edge_type] = [[],[]]

            edge_dict[edge_type][0].append(check_each_node_dict(str(identifier_type),next_one))
            edge_dict[edge_type][1].append(check_each_node_dict(str(identifier_type),this_one))

    new_dict = {}
    for k in edge_dict:
        v = edge_dict[k]
        new_dict[k] = (np.array(v[0]),np.array(v[1]))


    # return nodes,edge_dict,new_dict,dgl.heterograph(new_dict),each_node_dict
    if return_graph == False:
        return nodes, new_dict, each_node_dict
    out_nodes = nodes
    out_g = dgl.heterograph(new_dict)
    for ntype in out_g.ntypes:
        out_g.nodes[ntype].data['idx'] = torch.tensor(each_node_dict[ntype])

    return nodes,out_g

############################################################################
# copy from /home/zhangkechi/workspace/code2seq-master/ogb_python/extract.py
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

def subtoken_tokenizer(name):
    def camel_case_split(identifier):
        matches = re.finditer(
            '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)',
            identifier,
        )
        return [m.group(0) for m in matches]
    blocks = []
    for underscore_block in re.split(' |_',name):
        blocks.extend(camel_case_split(underscore_block))
    return blocks


def heterograph_graph_parser_subtoken(py_code, return_graph = True):
    
    py_ast = ast.parse(py_code)
    asdl_asts = [python_ast_to_asdl_ast(e, grammar) for e in py_ast.body]
    rsts = [asdl2dict(e) for e in asdl_asts]
    ast_tree = {'name':'module','type':mod_type,'children':rsts,'toParentField': None, 'toleftSiblingIdx':-1}

    edge_dict = {}
    # ast_parent2child, ast_child2parent, next_token, last_tokens
    each_node_dict = {}

    def check_each_node_dict(typ, idx):
        if typ not in each_node_dict:
            each_node_dict[typ] = []
        if idx not in each_node_dict[typ]:
            each_node_dict[typ].append(idx)
        return each_node_dict[typ].index(idx)



    nodes = {'all_node':[],'identifier_node':[],'subtoken_node':[]}
    # all_identifiers = []
    def find_left_sibling_idx(parent_idx, this_sibling_idx):
        all_siblings = [e for e in nodes['all_node'] if e['parent_idx'] == parent_idx]
        left_sibling = [e for e in all_siblings if e['sibling_idx'] == this_sibling_idx - 1]
        if len(left_sibling) == 0:
            return -1
        else:
            return left_sibling[0]['idx']

    def get_subtoken_node_idx(subtoken,identifier_idx = -1,subtoken_sibling_idx = 0,share = True):
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
        nodes['all_node'].append({'idx':subtoken_idx,
                'name':subtoken,
                'type':'subtoken',
                'toParentField':'toSubtoken',
                'parent_idx':identifier_idx,
                'sibling_idx':subtoken_sibling_idx,
                'identifier_idx':[identifier_idx],
                'subtoken_sibling_idx':[subtoken_sibling_idx]
                })
        nodes['subtoken_node'].append(subtoken_idx)
        return subtoken_idx



    def visit_tree(node,parent_idx = -1, sibling_idx = 0):
        now_idx = len(nodes['all_node'])

        nodes['all_node'].append({'idx':now_idx,
                        'name':node['name'],
                        'type':node['type'],
                        'toParentField':node['toParentField'],
                        'parent_idx':parent_idx,
                        'sibling_idx':sibling_idx
                        })
        if node['type'] == identifier_type:
            nodes['identifier_node'].append(now_idx)
            identifier_idx = now_idx
            subtoken_sibling_idx = 0

            # 增加subtoken边
            subtokens = subtoken_tokenizer(node['name'])
            for subtoken in subtokens:
                subtoken_idx = get_subtoken_node_idx(subtoken,identifier_idx,subtoken_sibling_idx)
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
                        edge_dict[edge_type] = [[],[]]
                    edge_dict[edge_type][0].append(check_each_node_dict(str(nodes['all_node'][identifier_idx]['type']),identifier_idx))
                    edge_dict[edge_type][1].append(check_each_node_dict(str(nodes['all_node'][subtoken_idx]['type']),subtoken_idx))
                    edge_type = (
                                str(nodes['all_node'][subtoken_idx]['type']),
                                str(nodes['all_node'][subtoken_idx]['toParentField'] + '_reverse'),
                                str(nodes['all_node'][identifier_idx]['type'])
                                )
                    if edge_type not in edge_dict:
                        edge_dict[edge_type] = [[],[]]
                    edge_dict[edge_type][0].append(check_each_node_dict(str(nodes['all_node'][subtoken_idx]['type']),subtoken_idx))    
                    edge_dict[edge_type][1].append(check_each_node_dict(str(nodes['all_node'][identifier_idx]['type']),identifier_idx))






        now_parent_idx = now_idx
        

        if parent_idx != -1:
            # 加父子边

            edge_type = (
                        str(nodes['all_node'][parent_idx]['type']),
                        str(nodes['all_node'][now_idx]['toParentField']),
                        str(nodes['all_node'][now_idx]['type'])
                        )
            if edge_type not in edge_dict:
                edge_dict[edge_type] = [[],[]]



            edge_dict[edge_type][0].append(check_each_node_dict(str(nodes['all_node'][parent_idx]['type']),parent_idx))
            edge_dict[edge_type][1].append(check_each_node_dict(str(nodes['all_node'][now_idx]['type']),now_idx))


            edge_type = (
                        str(nodes['all_node'][now_idx]['type']),
                        str(nodes['all_node'][now_idx]['toParentField'] + '_reverse'),
                        str(nodes['all_node'][parent_idx]['type'])
                        )
            if edge_type not in edge_dict:
                edge_dict[edge_type] = [[],[]]


            edge_dict[edge_type][0].append(check_each_node_dict(str(nodes['all_node'][now_idx]['type']),now_idx))    
            edge_dict[edge_type][1].append(check_each_node_dict(str(nodes['all_node'][parent_idx]['type']),parent_idx))
        
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
                edge_dict[edge_type]=[[], []]

            edge_dict[edge_type][0].append(check_each_node_dict(str(nodes['all_node'][left_sibling_idx]['type']), left_sibling_idx))
            edge_dict[edge_type][1].append(check_each_node_dict(str(nodes['all_node'][now_idx]['type']), now_idx))
            
            edge_type = (
                str(nodes['all_node'][now_idx]['type']),
                str('right_sibling_to_left'),
                str(nodes['all_node'][left_sibling_idx]['type'])
            )
            if edge_type not in edge_dict:
                edge_dict[edge_type]=[[], []]
            edge_dict[edge_type][0].append(check_each_node_dict(str(nodes['all_node'][now_idx]['type']), now_idx))
            edge_dict[edge_type][1].append(check_each_node_dict(str(nodes['all_node'][left_sibling_idx]['type']), left_sibling_idx))

        now_idx += 1
        
        children_node = node['children']
        for c_i in range(len(children_node)):
            c = children_node[c_i]
            visit_tree(c,now_parent_idx,c_i)

    visit_tree(ast_tree)
    identifier_node_len = len(nodes['identifier_node'])
    # identifier_node_len = 1
    if identifier_node_len >= 2:
        for i in range(len(nodes['identifier_node']) - 1):
            this_one = nodes['identifier_node'][i]
            next_one = nodes['identifier_node'][i + 1]

            edge_type = (
                        str(identifier_type),
                        str('next_token'),
                        str(identifier_type)
                        )

            if edge_type not in edge_dict:
                edge_dict[edge_type] = [[],[]]

            edge_dict[edge_type][0].append(check_each_node_dict(str(identifier_type),this_one))
            edge_dict[edge_type][1].append(check_each_node_dict(str(identifier_type),next_one))

            edge_type = (
                        str(identifier_type),
                        str('last_token'),
                        str(identifier_type)
                        )

            if edge_type not in edge_dict:
                edge_dict[edge_type] = [[],[]]

            edge_dict[edge_type][0].append(check_each_node_dict(str(identifier_type),next_one))
            edge_dict[edge_type][1].append(check_each_node_dict(str(identifier_type),this_one))

    new_dict = {}
    for k in edge_dict:
        v = edge_dict[k]
        new_dict[k] = (np.array(v[0]),np.array(v[1]))


    # return nodes,edge_dict,new_dict,dgl.heterograph(new_dict),each_node_dict
    if return_graph == False:
        return nodes, new_dict, each_node_dict
    out_nodes = nodes
    out_g = dgl.heterograph(new_dict)
    for ntype in out_g.ntypes:
        out_g.nodes[ntype].data['idx'] = torch.tensor(each_node_dict[ntype])

    return nodes,out_g



def heterograph_graph_parser_bpe(py_code, return_graph = True):

    py_ast = ast.parse(py_code)
    asdl_asts = [python_ast_to_asdl_ast(e, grammar) for e in py_ast.body]
    rsts = [asdl2dict(e) for e in asdl_asts]
    ast_tree = {'name':'module','type':mod_type,'children':rsts,'toParentField': None, 'toleftSiblingIdx':-1}

    edge_dict = {}
    # ast_parent2child, ast_child2parent, next_token, last_tokens
    each_node_dict = {}

    def check_each_node_dict(typ, idx):
        if typ not in each_node_dict:
            each_node_dict[typ] = []
        if idx not in each_node_dict[typ]:
            each_node_dict[typ].append(idx)
        return each_node_dict[typ].index(idx)



    nodes = {'all_node':[],'identifier_node':[],'subtoken_node':[]}
    # all_identifiers = []
    def find_left_sibling_idx(parent_idx, this_sibling_idx):
        all_siblings = [e for e in nodes['all_node'] if e['parent_idx'] == parent_idx]
        left_sibling = [e for e in all_siblings if e['sibling_idx'] == this_sibling_idx - 1]
        if len(left_sibling) == 0:
            return -1
        else:
            return left_sibling[0]['idx']

    def get_subtoken_node_idx(subtoken,identifier_idx = -1,subtoken_sibling_idx = 0,share = True):
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
        nodes['all_node'].append({'idx':subtoken_idx,
                'name':subtoken,
                'type':'subtoken',
                'toParentField':'toSubtoken',
                'parent_idx':identifier_idx,
                'sibling_idx':subtoken_sibling_idx,
                'identifier_idx':[identifier_idx],
                'subtoken_sibling_idx':[subtoken_sibling_idx]
                })
        nodes['subtoken_node'].append(subtoken_idx)
        return subtoken_idx



    def visit_tree(node,parent_idx = -1, sibling_idx = 0):
        now_idx = len(nodes['all_node'])

        nodes['all_node'].append({'idx':now_idx,
                        'name':node['name'],
                        'type':node['type'],
                        'toParentField':node['toParentField'],
                        'parent_idx':parent_idx,
                        'sibling_idx':sibling_idx
                        })
        if node['type'] == identifier_type:
            nodes['identifier_node'].append(now_idx)
            identifier_idx = now_idx
            subtoken_sibling_idx = 0

            # 增加subtoken边
            subtokens = tokenizer.tokenize(node['name'])
            for subtoken in subtokens:
                subtoken_idx = get_subtoken_node_idx(subtoken,identifier_idx,subtoken_sibling_idx)
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
                        edge_dict[edge_type] = [[],[]]
                    edge_dict[edge_type][0].append(check_each_node_dict(str(nodes['all_node'][identifier_idx]['type']),identifier_idx))
                    edge_dict[edge_type][1].append(check_each_node_dict(str(nodes['all_node'][subtoken_idx]['type']),subtoken_idx))
                    edge_type = (
                                str(nodes['all_node'][subtoken_idx]['type']),
                                str(nodes['all_node'][subtoken_idx]['toParentField'] + '_reverse'),
                                str(nodes['all_node'][identifier_idx]['type'])
                                )
                    if edge_type not in edge_dict:
                        edge_dict[edge_type] = [[],[]]
                    edge_dict[edge_type][0].append(check_each_node_dict(str(nodes['all_node'][subtoken_idx]['type']),subtoken_idx))    
                    edge_dict[edge_type][1].append(check_each_node_dict(str(nodes['all_node'][identifier_idx]['type']),identifier_idx))






        now_parent_idx = now_idx
        

        if parent_idx != -1:
            # 加父子边

            edge_type = (
                        str(nodes['all_node'][parent_idx]['type']),
                        str(nodes['all_node'][now_idx]['toParentField']),
                        str(nodes['all_node'][now_idx]['type'])
                        )
            if edge_type not in edge_dict:
                edge_dict[edge_type] = [[],[]]



            edge_dict[edge_type][0].append(check_each_node_dict(str(nodes['all_node'][parent_idx]['type']),parent_idx))
            edge_dict[edge_type][1].append(check_each_node_dict(str(nodes['all_node'][now_idx]['type']),now_idx))


            edge_type = (
                        str(nodes['all_node'][now_idx]['type']),
                        str(nodes['all_node'][now_idx]['toParentField'] + '_reverse'),
                        str(nodes['all_node'][parent_idx]['type'])
                        )
            if edge_type not in edge_dict:
                edge_dict[edge_type] = [[],[]]


            edge_dict[edge_type][0].append(check_each_node_dict(str(nodes['all_node'][now_idx]['type']),now_idx))    
            edge_dict[edge_type][1].append(check_each_node_dict(str(nodes['all_node'][parent_idx]['type']),parent_idx))
        
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
                edge_dict[edge_type]=[[], []]

            edge_dict[edge_type][0].append(check_each_node_dict(str(nodes['all_node'][left_sibling_idx]['type']), left_sibling_idx))
            edge_dict[edge_type][1].append(check_each_node_dict(str(nodes['all_node'][now_idx]['type']), now_idx))
            
            edge_type = (
                str(nodes['all_node'][now_idx]['type']),
                str('right_sibling_to_left'),
                str(nodes['all_node'][left_sibling_idx]['type'])
            )
            if edge_type not in edge_dict:
                edge_dict[edge_type]=[[], []]
            edge_dict[edge_type][0].append(check_each_node_dict(str(nodes['all_node'][now_idx]['type']), now_idx))
            edge_dict[edge_type][1].append(check_each_node_dict(str(nodes['all_node'][left_sibling_idx]['type']), left_sibling_idx))

        now_idx += 1
        
        children_node = node['children']
        for c_i in range(len(children_node)):
            c = children_node[c_i]
            visit_tree(c,now_parent_idx,c_i)

    visit_tree(ast_tree)
    identifier_node_len = len(nodes['identifier_node'])
    # identifier_node_len = 1
    if identifier_node_len >= 2:
        for i in range(len(nodes['identifier_node']) - 1):
            this_one = nodes['identifier_node'][i]
            next_one = nodes['identifier_node'][i + 1]

            edge_type = (
                        str(identifier_type),
                        str('next_token'),
                        str(identifier_type)
                        )

            if edge_type not in edge_dict:
                edge_dict[edge_type] = [[],[]]

            edge_dict[edge_type][0].append(check_each_node_dict(str(identifier_type),this_one))
            edge_dict[edge_type][1].append(check_each_node_dict(str(identifier_type),next_one))

            edge_type = (
                        str(identifier_type),
                        str('last_token'),
                        str(identifier_type)
                        )

            if edge_type not in edge_dict:
                edge_dict[edge_type] = [[],[]]

            edge_dict[edge_type][0].append(check_each_node_dict(str(identifier_type),next_one))
            edge_dict[edge_type][1].append(check_each_node_dict(str(identifier_type),this_one))

    new_dict = {}
    for k in edge_dict:
        v = edge_dict[k]
        new_dict[k] = (np.array(v[0]),np.array(v[1]))


    # return nodes,edge_dict,new_dict,dgl.heterograph(new_dict),each_node_dict
    if return_graph == False:
        return nodes, new_dict, each_node_dict
    out_nodes = nodes
    out_g = dgl.heterograph(new_dict)
    for ntype in out_g.ntypes:
        out_g.nodes[ntype].data['idx'] = torch.tensor(each_node_dict[ntype])

    return nodes,out_g



def dict2graph(new_dict, each_node_dict):
    out_g = dgl.heterograph(new_dict)
    for ntype in out_g.ntypes:
        out_g.nodes[ntype].data['idx'] = torch.tensor(each_node_dict[ntype])
    return out_g

# 可视化parser图
def showGraph(g,ndata):
    import prettytable as pt
    import networkx as nx
    homo = dgl.to_homogeneous(g,ndata=['idx'])
    graph_idx_dict = homo.ndata['idx'].tolist()
    graph_idx_dict = {graph_idx_dict[e]:e for e in range(len(graph_idx_dict))}
    nx.draw(homo.to_networkx(),with_labels = True)
    all_keys = list(ndata['all_node'][0].keys())
    # print(all_keys)
    origin_keys = ['idx', 'name', 'type', 'toParentField', 'parent_idx', 'sibling_idx']
    subtoken_keys = ['idx', 'name', 'type', 'toParentField', 'identifier_idx', 'subtoken_sibling_idx']
    tb = pt.PrettyTable()
    tb.field_names = ['graph_idx'] + all_keys
    for e in ndata['all_node']:
        if 'identifier_idx' in e:
            all_keys = subtoken_keys
        else:
            all_keys = origin_keys
        tb.add_row([graph_idx_dict[e['idx']]]+[e[k] for k in all_keys])
    print(tb)
# showGraph(heterograph_graph_parser_subtoken(test_code,return_graph=True)[1],heterograph_graph_parser_subtoken(test_code,return_graph=True)[0])

if __name__ == '__main__':
    py_code ='''
for i in range(10):
    print("hello world")

'''
    # show_ast_tree(py_code)
    n,g = heterograph_graph_parser(py_code)
    print(n)
    print("-------------------------")
    # print(e)
    # print(d)
    print(g)
    print(g.ndata)
    # print(ed)



