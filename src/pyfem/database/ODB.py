# -*- coding: utf-8 -*-
"""

"""
import h5py  # type: ignore
from numpy import array, ndarray, empty, zeros, hstack
from numpy.linalg import norm

from pyfem.utils.visualization import object_slots_to_string


class ODB:
    """
    cells = [4, 2, 6, 8, 5, 4, 5, 8, 4, 1, 4, 6, 3, 7, 8, 4, 8, 7, 0, 4]
    elements = [[2, 6, 8, 5], [5, 8, 4, 1], [6, 3, 7, 8], [8, 7, 0, 4]]
    """

    __slots_dict__: dict = {
        'title': ('str', 'odb名称'),
        'path': ('str', '文件路径'),
        'version': ('str', '求解器版本'),
        'materials': ('dict', '材料字典'),
        'dimension': ('int', '空间维度'),
        'nodes': ('ndarray', '节点坐标数组'),
        'cells': ('ndarray', '单元连接信息数组'),
        'celltypes': ('ndarray', '单元类型数组'),
        'elements': ('list[ndarray]', '单元connectivity列表'),
        'node_sets': ('dict[str, list[int]]', '节点集合字典'),
        'element_sets': ('dict[str, list[int]]', '单元集合字典'),
        'steps': ('dict', '载荷步字典'),
        'vtu_count': ('int', '读取vtu文件时的计数器'),
        'is_deformed': ('bool', '判断数据中是否有位移'),
        'field_output_labels': ('dict', 'fieldOutputs变量的标签字典'),
        'xy_data_dict': ('dict', '存储导出的xy数据字典'),
        'current_step': ('str', 'interactor中当前正在渲染的step名字'),
    }

    __slots__: list = [slot for slot in __slots_dict__.keys()]

    def __init__(self) -> None:
        self.title: str = ''
        self.path: str = ''
        self.version: str = ''
        self.materials: dict = {}
        self.dimension: int = -1
        self.nodes: ndarray = empty(0)
        self.cells: ndarray = empty(0)
        self.celltypes: ndarray = empty(0)
        self.elements: list[ndarray] = []
        self.node_sets: dict[str, list[int]] = {}
        self.element_sets: dict[str, list[int]] = {}
        self.steps: dict = {}
        self.vtu_count: int = 0
        self.is_deformed: bool = False
        self.field_output_labels: dict = {}
        self.xy_data_dict: dict[str, ndarray] = {}
        self.current_step = 'Step-1'

    def reset(self):
        self.__init__()

    def load_hdf5(self, file_path: str) -> str:
        try:
            f = h5py.File(file_path, 'r')
            self.title = f['title'][()]
            self.path = file_path
            self.version = f['version'][()]
            self.dimension = f['dimension'][()]
            self.nodes = f['nodes'][()]

            if self.dimension == 2:
                new_column = zeros((self.nodes.shape[0], 1))
                self.nodes = hstack((self.nodes, new_column))
            self.cells = f['elements']['cells'][()]
            self.celltypes = f['elements']['celltypes'][()]

            # 下面的函数用于从cells计算单元的connectivity列表
            i = 0
            while i < len(self.cells):
                n = self.cells[i]
                self.elements.append(self.cells[i + 1:i + 1 + n])
                i += n + 1

            for key, value in f['node_sets'].items():
                self.node_sets[key] = value[()]

            for key, value in f['element_sets'].items():
                self.element_sets[key] = value[()]

            for step_name in f['steps'].keys():
                self.steps[step_name] = {
                    'frames': []
                }
                self.field_output_labels[step_name] = {}

                frame_ids = sorted([int(frame_id) for frame_id in f['steps'][step_name]['frames'].keys()])

                for frame_id in frame_ids:
                    frame_data = {'incrementNumber': f['steps'][step_name]['frames'][str(frame_id)]['incrementNumber'][()],
                                  'frameValue': f['steps'][step_name]['frames'][str(frame_id)]['frameValue'][()],
                                  'fieldOutputs': {}}
                    for key, value in f['steps'][step_name]['frames'][str(frame_id)]['fieldOutputs'].items():
                        frame_data['fieldOutputs'][key] = {}
                        if self.dimension == 2 and key == 'U':
                            u_2d = value['bulkDataBlocks'][()]
                            u_3d = hstack((u_2d, zeros((u_2d.shape[0], 1))))
                            frame_data['fieldOutputs'][key]['bulkDataBlocks'] = u_3d
                        else:
                            frame_data['fieldOutputs'][key]['bulkDataBlocks'] = value['bulkDataBlocks'][()]

                        componentLabels = [v.decode() for v in value['componentLabels'][()] if v != b'']
                        validInvariants = [v.decode() for v in value['validInvariants'][()] if v != b'']
                        description = value['description'][()].decode()
                        type = value['type'][()].decode()

                        frame_data['fieldOutputs'][key]['componentLabels'] = componentLabels
                        frame_data['fieldOutputs'][key]['validInvariants'] = validInvariants
                        frame_data['fieldOutputs'][key]['description'] = description
                        frame_data['fieldOutputs'][key]['type'] = type

                        self.field_output_labels[step_name][key] = {'componentLabels': componentLabels,
                                                                    'validInvariants': validInvariants,
                                                                    'description': description,
                                                                    'type': type}

                    self.steps[step_name]['frames'].append(frame_data)

            f.close()
            return 'Succeed to load file: ' + f'\"{file_path}\"\n'
        except NotImplementedError:
            try:
                f.close()
            except NotImplementedError:
                pass
            return 'Failed to load file: ' + f'\"{file_path}\"\n'

    def get_nodes_data(self, step_name: str, nodes: list[int], name: str) -> tuple[ndarray, ndarray, ndarray]:
        field_name = name
        frame_ids = []
        frame_values = []
        data = []
        frames = self.steps[step_name]['frames']
        for frame_id, frame_value in enumerate(frames):
            frame_ids.append(frame_id)
            frame_values.append(frame_value['frameValue'])
            data.append(frame_value['fieldOutputs'][field_name]['bulkDataBlocks'][nodes])
        return array(frame_ids), array(frame_values), array(data)

    def get_node_set_data(self, step_name: str, node_set_name: str, name: str) -> tuple[ndarray, ndarray, ndarray]:
        return self.get_nodes_data(step_name, self.node_sets[node_set_name], name)

    def to_string(self, level: int = 1) -> str:
        return object_slots_to_string(self, level)

    def show(self) -> None:
        print(self.to_string())

    def get_frame_data(self, step_name: str, frame_id: int, name: str) -> ndarray:
        if len(name.split('-')) == 1:
            field_name = name
            field_data = self.steps[step_name]['frames'][frame_id]['fieldOutputs'][field_name]
            return field_data['bulkDataBlocks']

        elif len(name.split('-')) == 2:
            field_name = name.split('-')[0]
            field_option = name.split('-')[1]
            field_data = self.steps[step_name]['frames'][frame_id]['fieldOutputs'][field_name]

            if field_option in field_data['componentLabels']:
                col = field_data['componentLabels'].index(field_option)
                return field_data['bulkDataBlocks'][:, col]
            if field_option == 'MAGNITUDE':
                return norm(field_data['bulkDataBlocks'], axis=1)

        else:
            raise NotImplementedError


if __name__ == '__main__':
    odb = ODB()
    odb.load_hdf5(r'F:\Github\pyfem\examples\mechanical\rectangle_hole_3D\Job-1.hdf5')
    print(odb.field_output_labels)
