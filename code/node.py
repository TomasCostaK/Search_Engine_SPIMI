# Auxiliary class to help the boosting with positional params
class Node:
    def __init__(self, elem_value, list_number, column_number):
        self.elem_value = elem_value
        self.list_number = list_number
        self.column_number = column_number

    def __lt__(self, other):
        return self.elem_value < other.elem_value