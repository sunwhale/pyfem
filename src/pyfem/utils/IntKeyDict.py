from typing import List


class IntKeyDict(dict):
    """
    关键字只能为整数（Integer）类型的字典，已经存在的键值无法通过赋值方法（__setitem__）更改。
    """

    def __init__(self):
        super().__init__()
        self.indices_to_ids = {}
        self.ids_to_indices = {}
        self.update_indices()

    def __setitem__(self, key: int, value: any) -> None:
        if isinstance(key, int):
            if key in self:
                raise KeyError(f"Key {key} already exists")
            else:
                super().__setitem__(key, value)
        else:
            raise TypeError("Key must be an integer")

    def update_indices(self):
        list_of_keys = list(self.keys())
        self.ids_to_indices = {key: i for i, key in enumerate(list_of_keys)}
        self.indices_to_ids = {i: key for i, key in enumerate(list_of_keys)}

    def add_item_by_id(self, id_: int, item: any) -> None:
        self[id_] = item

    def get_items_by_ids(self, ids: List[int]) -> List[any]:
        if isinstance(ids, list):
            return [self[id_] for id_ in ids]
        else:
            raise TypeError("Argument to get_items_by_ids() must be a list")

    def get_indices_by_ids(self, ids: List[int]) -> List[int]:
        if isinstance(ids, list):
            if len(self.ids_to_indices) != len(self):
                self.update_indices()
            return [self.ids_to_indices[id] for id in ids]
        else:
            raise TypeError("Argument to get_indices_by_ids() must be a list")

    def get_ids_by_indices(self, indices: List[int]) -> List[int]:
        if isinstance(indices, list):
            if len(self.ids_to_indices) != len(self):
                self.update_indices()
            return [self.indices_to_ids[index] for index in indices]
        else:
            raise TypeError("Argument to get_ids_by_indices() must be a list")


if __name__ == "__main__":
    a = IntKeyDict()
    # a[1] = f'x{1}'
    # a[100] = f'x{100}'
    # a[10] = f'x{10}'
    for i in range(0, 1000000, 2):
        a[i] = f'x{i}'
    a.update_indices()
    print(a.get_indices_by_ids([0, 2, 4]))
    print(a.get_ids_by_indices([1, 3, 5]))
