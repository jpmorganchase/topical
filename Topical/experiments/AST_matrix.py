import numpy as np
import ast
import itertools
import re
from collections import OrderedDict
from typing import Tuple, Dict, List, Union


class ASTTree:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encoding_mask(self, text, d):
        encoding = self.tokenizer(text, truncation=True, padding=True, return_tensors='pt')
        try:
            (parent, children) = self.get_tree_info(text)
            node_vector = self.get_nodes_vector(parent)
            node_positions = [self.get_position(node) for node in node_vector]
            node_absolute_positions = [self.row_position_to_text_position(text, pos) for pos in node_positions]
            matched = self.match_node_to_token(parent, encoding, text)
            parents_matrix = self.get_token_parents_matrix(parent, matched, self.get_nodes_vector(parent), d)
        except:
            parents_matrix = None
        return parents_matrix

    def match_node_to_token(self, parent_nodes, tokenized_code, code):
        """ For each token returns the position in the nodes vector with which it is matched
        """
        matched = [None]
        node_vector = self.get_nodes_vector(parent_nodes)
        node_positions = [self.get_position(node) for node in node_vector]
        node_absolute_positions = [self.row_position_to_text_position(code, pos) for pos in node_positions]
        token_absolute_positions = self.get_tokens_spans(tokenized_code)
        if list(set(token_absolute_positions)) == [None]:
            return []
        for n in token_absolute_positions[1:-1]:
            if n in node_absolute_positions:
                matched.append(node_absolute_positions.index(n))
            else:
                node = 0
                while n[0] > node_absolute_positions[node][1] and node < len(node_absolute_positions) - 1:
                    node += 1
                if n[0] >= node_absolute_positions[node][0] and n[1] <= node_absolute_positions[node][1]:
                    matched.append(node)
                else:
                    matched.append(None)
        matched.append(None)
        return matched

    def get_token_children_matrix(self, children_nodes, node_match, nodes_vector, d):
        """ Given an arbitrary distance d, returns the tokens children matrix
        """
        final_matrix = []
        for node_id in node_match:
            if node_id is not None:
                node = nodes_vector[node_id]
                child = self.get_children(node, children_nodes)
                if d > 1:
                    for _ in range(d - 1):
                        child = self.get_children(child, children_nodes)
                if child in nodes_vector:
                    child_id = nodes_vector.index(child)
                else:
                    child_id = -1
                token_feature = [1 if ele == child_id else 0 for ele in node_match]
            else:
                token_feature = list(np.zeros(len(node_match), dtype=np.int32))
            final_matrix.append(token_feature)
        return final_matrix

    def get_token_parents_matrix(self, parents_nodes, node_match, nodes_vector, d):
        """ Given an arbitrary distance d, returns the tokens parents matrix
        """
        final_matrix = []
        for node_id in node_match:
            if node_id is not None:
                node = nodes_vector[node_id]
                child = self.get_parent(node, parents_nodes)
                if d > 1:
                    for _ in range(d - 1):
                        childs = []
                        for c in child:
                            result = self.get_parent(c, parents_nodes)
                            childs.extend(result)
                        child = childs
                    children = [nodes_vector.index(c_node) for c_node in child if c_node in nodes_vector]

                token_feature = [1 if ele in children else 0 for ele in node_match]
            else:
                token_feature = list(np.zeros(len(node_match), dtype=np.int32))
            final_matrix.append(token_feature)
        return final_matrix

    def get_token_siblings_matrix(self, parents_nodes, children_nodes, node_match, nodes_vector, d):
        """ Given an arbitrary distance d, returns the tokens siblings matrix
        (cf. SyntaxBERT paper for siblings definition)
        """
        final_matrix = []
        combinaisons = []
        for i in range(1, d):
            combinaisons.append((i, d - i))
        for node_id in node_match:
            if node_id is not None:
                node = nodes_vector[node_id]
                siblings = []
                for c in combinaisons:
                    parent = self.get_children(node, children_nodes)
                    if c[0] > 1:
                        for _ in range(c[0] - 1):
                            parent = self.get_children(parent, children_nodes)
                    child = self.get_parent(parent, parents_nodes)
                    if c[1] > 1:
                        for _ in range(c[1] - 1):
                            childs = []
                            for i in child:
                                result = self.get_parent(i, parents_nodes)
                                childs.extend(result)
                            child = childs
                    siblings.extend(child)
                siblings_id = [nodes_vector.index(sibling) for sibling in siblings if sibling in nodes_vector]
                token_features = [1 if ele in siblings_id else 0 for ele in node_match]
            else:
                token_features = list(np.zeros(len(node_match), dtype=np.int32))
            final_matrix.append(token_features)
        return final_matrix

    @staticmethod
    def get_tree_info(code: str):
        """ This function walks the code AST to retrieive
        parental relations between nodes in a dict format
        """
        sibling_nodes = []
        parent_nodes = OrderedDict()
        tree = ast.parse(code)
        children = [*ast.iter_child_nodes(tree)]
        sibling_nodes.append([children])
        parent_nodes[tree] = children
        level = 1

        while len(sibling_nodes) == level:
            last_siblings = list(itertools.chain(*sibling_nodes[-1]))
            last_children = []
            for child in last_siblings:
                children = [*ast.iter_child_nodes(child)]
                if len(children) > 0:
                    last_children.append(children)
                    parent_nodes[child] = children
            if len(last_children) > 0:
                sibling_nodes.append(last_children)
            level += 1

        children_nodes = {}
        for parent, children in parent_nodes.items():
            for child in children:
                children_nodes[child] = parent

        return parent_nodes, children_nodes

    @staticmethod
    def row_position_to_text_position(code: str, p: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """ Node position are defined by their line and position in this line in the code.
        We retrieive the absolute position of a node in the stringified code.
        """
        list_code = re.split(r'(?<=\n)', code)
        beginning = sum([len(line) for line in list_code[:p[0] - 1]]) + p[1]
        end = sum([len(line) for line in list_code[:p[2] - 1]]) + p[3]
        return beginning, end

    @staticmethod
    def get_children(node, children_info) -> Union[List, None]:
        """ For a given node, returns the parent if the node is part of the tree
        (as functions and class def are not taken into consideration,
        we need to handle this case by returning NoneType)
        """
        if node in children_info.keys():
            return children_info[node]
        else:
            return None

    @staticmethod
    def get_parent(node, parent_info):
        """ For a given node, returns the children if the node has some,
        else returns the info that it is a leaf of the tree thanks to an empty list

        """
        if node in parent_info.keys():
            return parent_info[node]
        else:
            return []

    def get_nodes_vector(self, parent_info: Dict):
        """ Returns a list containing all the nodes in the tree
        """
        ele_vector = [value for values in parent_info.values() for value in values]
        ele_vector.insert(0, list(parent_info.keys())[0])
        augmented_ele_vector = [(node, self.get_position(node))
                                for node in ele_vector
                                if self.get_position(node) is not None]
        augmented_ele_vector = sorted(augmented_ele_vector, key=lambda x: (x[1][0], x[1][1]))
        return [ele[0] for ele in augmented_ele_vector]

    @staticmethod
    def get_position(node) -> Union[Tuple[int, int, int, int], None]:
        """ Get the position of a node in the tree, not the absolute position
        """
        if 'lineno' in list(dir(node)):
            return node.lineno, node.col_offset, node.end_lineno, node.end_col_offset
        else:
            return None

    @staticmethod
    def get_tokens_spans(tokenizer_output):
        """Returns the absolute positions of the tokens in the original code string"""
        tokens_position = []
        token_ids = tokenizer_output.ids
        for i, token_id in enumerate(token_ids):
            try:
                char_span = tokenizer_output.token_to_chars(i)
                tokens_position.append((char_span.start, char_span.end))
            except:
                tokens_position.append(None)

        return tokens_position
