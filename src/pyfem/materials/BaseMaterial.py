import copy

from numpy import zeros
from pyfem.io.Properties import Properties


class BaseMaterial:

    def __init__(self, props):

        self.out_data = None
        self.numerical_tangent = False
        self.store_output_flag = False

        self.old_history = {}
        self.new_history = {}

        self.out_labels = []

    def set_history_parameter(self, name, val) -> None:
        self.new_history[name] = val

    def get_history_parameter(self, name):
        if isinstance(self.old_history[name], float):
            return self.old_history[name]
        else:
            return self.old_history[name].copy()

    def commit_history(self) -> None:
        self.old_history = copy.deepcopy(self.new_history)

    def set_output_labels(self, labels) -> None:
        self.out_labels = labels
        self.out_data = zeros(len(self.out_labels))

    def store_outputs(self, data) -> None:
        if self.store_output_flag:
            self.out_data = data


if __name__ == "__main__":
    props = Properties()
    mat = BaseMaterial(props)
    print(mat)
