from anytree import AnyNode, RenderTree
from tree_sitter import Language, Parser


def get_python_sitter(build_path='/home/username/workspace/treesitter_tools/build/csn_parsers.so', language='java'):
    PY_LANGUAGE = Language(
        build_path, language)
    sitter_parser = Parser()
    sitter_parser.set_language(PY_LANGUAGE)

    return sitter_parser


def converttoany(tree, text, abstract=True):
    nodeid = [0]
    currentnodetype = tree.type
    is_ast_node = tree.is_named
    root = AnyNode(type=currentnodetype, name=text[tree.start_byte:tree.end_byte].decode(
        'utf-8'), is_ast_node=is_ast_node, index=0)

    leaf_node = []

    def traverse(tree, parent=None):  # parent: current anynode for tree
        parent_id = parent.index

        # filter out non-ast-node if needed
        def check_literal_node(node):
            if 'literal' in node.type:
                return True
            if node.type in ['string', 'string_literal']:
                return True
            return False
        if abstract:
            filter_children = [
                child for child in tree.children if child.is_named]
        else:
            if check_literal_node(tree):
                filter_children = []
            else:
                filter_children = tree.children

        if len(filter_children) == 0:
            leaf_node.append(parent)

        for child in filter_children:
            # 每个child都有一个与众不同的id吧？
            childtype = child.type
            is_ast_node = child.is_named
            
            # field_name
            # cursor = child.walk()
            # print(cursor.current_field_name())


            # if abstract and not is_ast_node:
            #     continue
            nodeid[0] += 1
            childnode = AnyNode(type=childtype, name=text[child.start_byte:child.end_byte].decode(
                'utf-8'), is_ast_node=is_ast_node, parent=parent, index=nodeid[0])
#             print(text[child.start_byte:child.end_byte],'\t',child.start_byte,'\t',child.end_byte)
            traverse(child, parent=childnode)
    traverse(tree, parent=root)
    rst_tree = root
    return rst_tree, leaf_node


def myparse(py_code, generate_ast=True, language='python'):
    py_parser = get_python_sitter(language=language)
    py_tree = py_parser.parse(bytes(py_code, 'utf-8'))
    py_rst_tree, leaf_node = converttoany(
        py_tree.root_node, bytes(py_code, 'utf-8'), abstract=generate_ast)
    return py_rst_tree, leaf_node


def get_leaf_node_list(py_code, generate_ast=True,language='python'):
    py_rst_tree, leaf_nodes = myparse(py_code, generate_ast=generate_ast,language=language)
    rst_list = [(leaf_node.name, get_path_from_leaf_node(leaf_node))
                for leaf_node in leaf_nodes]
    leaf_name_list = [e[0] for e in rst_list]
    path_list = [e[1] for e in rst_list]
    return leaf_name_list, path_list


def render_tree(tree):
    print(RenderTree(tree))


def get_path_from_leaf_node(leaf_node):
    nodes_on_path = []
    now_node = leaf_node
    while now_node:
        nodes_on_path.append(now_node)
        now_node = now_node.parent
    return '|'.join([node.type for node in nodes_on_path[::-1]])


def unit_test(py_code=None,language='python'):
    if py_code is None:
        py_code = '''print('hello')'''
        # import json
        # with open('/home/username/workspace/data/codexglue/dataset/java/valid.jsonl', 'r') as f:
        #     py_code = f.readlines()
        #     py_code = py_code[0]
        # py_code = json.loads(py_code)['code']
    print("code:")
    print(py_code)
    print("----------")
    print("AST:")
    ast_tree, ast_leaf_nodes = myparse(py_code, generate_ast=True,language=language)
    render_tree(ast_tree)
    print("all_paths:")
    print([(leaf_node.name, get_path_from_leaf_node(leaf_node))
           for leaf_node in ast_leaf_nodes])

    print("----------")
    print("CST:")
    cst_tree, cst_leaf_nodes = myparse(py_code, generate_ast=False,language=language)
    render_tree(cst_tree)
    print("all_paths:")
    print([(leaf_node.name, get_path_from_leaf_node(leaf_node))
          for leaf_node in cst_leaf_nodes])


if __name__ == "__main__":
    py_code = '''
#include <iostream>
using namespace std;
int main()
{
    char temp, list[8] = "cbafedg";
    for (int j = 0; j < 7; j++)
        for (int i = 0; i < 6 - j; i++)
        {
            if ( list[i] > list[i+1] )
            {
                temp = list[i];
                list[i] = list[i+1];
                list[i+1] = temp;
            }
        }
    cout<<list;
    return 0;
}
    '''
    unit_test(py_code,language='python')
