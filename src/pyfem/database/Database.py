# -*- coding: utf-8 -*-
"""

"""
import os

import h5py
from numpy import array, ndarray

from pyfem import __version__
from pyfem.assembly.Assembly import Assembly
from pyfem.utils.visualization import object_slots_to_string_ndarray


class Database:
    """
    求解器基类。

    :ivar assembly: 装配体对象
    :vartype assembly: Assembly

    :ivar solver: 求解器属性
    :vartype solver: Solver

    :ivar dof_solution: 求解得到自由度的值
    :vartype dof_solution: ndarray
    """

    __slots_dict__: dict = {
        'assembly': ('Assembly', '装配体对象'),
        'title': ('str', '装配体对象'),
        'version': ('str', '求解器属性'),
        'materials': ('dict', '求解器属性'),
        'dimension': ('ndarray', '求解得到自由度的值'),
        'nodes': ('ndarray', '求解得到自由度的值'),
        'elements': ('ndarray', '求解得到自由度的值'),
        'node_sets': ('ndarray', '求解得到自由度的值'),
        'element_sets': ('ndarray', '求解得到自由度的值'),
        'steps': ('ndarray', '求解得到自由度的值'),
    }

    __slots__: list = [slot for slot in __slots_dict__.keys()]

    def __init__(self, assembly: Assembly) -> None:
        self.assembly: Assembly = assembly
        self.title: str = assembly.props.title
        self.version: str = f'PYFEM {__version__}'
        self.materials: dict = {}
        self.dimension: int = assembly.props.mesh_data.dimension
        self.nodes: ndarray = assembly.props.mesh_data.nodes
        self.elements: list[ndarray] = assembly.props.mesh_data.elements
        self.node_sets: dict[str, list[int]] = assembly.props.mesh_data.node_sets
        self.element_sets: dict[str, list[int]] = assembly.props.mesh_data.element_sets
        self.steps: dict = {}

    def to_string(self, level: int = 1) -> str:
        return object_slots_to_string_ndarray(self, level)

    def show(self) -> None:
        print(self.to_string())

    def write(self) -> None:
        for output in self.assembly.props.outputs:
            if output.is_save:
                if output.type == 'vtk':
                    pass
                if output.type == 'hdf5':
                    self.add_hdf5()

    def init_hdf5(self) -> None:
        job_name = self.assembly.props.input_file.stem
        output_file = self.assembly.props.work_path.joinpath(f'{job_name}.hdf5')
        if os.path.exists(output_file):
            os.remove(output_file)
        self.add_hdf5()

    def add_hdf5(self) -> None:
        assembly = self.assembly
        props = self.assembly.props
        timer = self.assembly.timer

        job_name = props.input_file.stem
        output_file = props.work_path.joinpath(f'{job_name}.hdf5')

        f = h5py.File(output_file, 'a')

        if 'title' not in f.keys():
            f.create_dataset('title', data=self.title)
        if 'version' not in f.keys():
            f.create_dataset('version', data=self.version)
        if 'dimension' not in f.keys():
            f.create_dataset('dimension', data=self.dimension)
        if 'nodes' not in f.keys():
            f.create_dataset('nodes', data=self.nodes)
        if 'elements' not in f.keys():
            elements_group = f.create_group('elements')

            connectivity_list = []
            cells_list = []
            offset_list = [0]
            celltypes_list = []

            offset = 0
            for connectivity in self.elements:
                cells_list.append(len(connectivity))
                for node_id in connectivity:
                    connectivity_list.append(node_id)
                    cells_list.append(node_id)
                offset += len(connectivity)
                offset_list.append(offset)
                if self.dimension == 2:
                    celltypes_list.append(9)
                elif self.dimension == 3:
                    celltypes_list.append(12)

            elements_group.create_dataset('connectivity', data=array(connectivity_list, dtype='int32'))
            elements_group.create_dataset('offsets', data=array(offset_list, dtype='int32'))
            elements_group.create_dataset('cells', data=array(cells_list, dtype='int32'))
            elements_group.create_dataset('celltypes', data=array(celltypes_list, dtype='int32'))

        if 'node_sets' not in f.keys():
            f.create_group('node_sets')
            for key, value in self.node_sets.items():
                f['node_sets'].create_dataset(key, data=value)

        if 'element_sets' not in f.keys():
            f.create_group('element_sets')
            for key, value in self.element_sets.items():
                f['element_sets'].create_dataset(key, data=value)

        if 'steps' not in f.keys():
            f.create_group('steps')
            for step_name in ['Step-1']:
                f['steps'].create_group(step_name)
                frameId = str(0)
                f['steps'][step_name].create_dataset('frame_count', data=int(frameId))
                f['steps'][step_name].create_group('frames')
        else:
            for step_name in ['Step-1']:
                frame_count = f['steps'][step_name]['frame_count'][()]
                frameId = str(frame_count + 1)
                frame_count = f['steps'][step_name]['frame_count'][()] = int(frameId)

        for step_name in ['Step-1']:
            f['steps'][step_name]['frames'].create_group(frameId)
            f['steps'][step_name]['frames'][frameId].create_dataset('incrementNumber', data=timer.increment)
            f['steps'][step_name]['frames'][frameId].create_dataset('frameValue', data=timer.time0)
            f['steps'][step_name]['frames'][frameId].create_group('fieldOutputs')

            if 'T' in props.dof.names:
                col_T = props.dof.names.index('T')
                dof_T = assembly.dof_solution.reshape(-1, len(props.dof.names))[:, col_T]
                f['steps'][step_name]['frames'][frameId]['fieldOutputs'].create_group('T')
                f['steps'][step_name]['frames'][frameId]['fieldOutputs']['T'].create_dataset('bulkDataBlocks', data=dof_T)
                f['steps'][step_name]['frames'][frameId]['fieldOutputs']['T'].create_dataset('componentLabels', data=array([''], dtype=object))
                f['steps'][step_name]['frames'][frameId]['fieldOutputs']['T'].create_dataset('validInvariants', data=array([''], dtype=object))
                f['steps'][step_name]['frames'][frameId]['fieldOutputs']['T'].create_dataset('description', data='Temperature')
                f['steps'][step_name]['frames'][frameId]['fieldOutputs']['T'].create_dataset('type', data='SCALAR')

            if 'u1' in props.dof.names:
                if self.dimension == 2:
                    value = assembly.dof_solution.reshape(-1, len(props.dof.names))[:, 0:2]
                    f['steps'][step_name]['frames'][frameId]['fieldOutputs'].create_group('U')
                    f['steps'][step_name]['frames'][frameId]['fieldOutputs']['U'].create_dataset('bulkDataBlocks', data=value)
                    f['steps'][step_name]['frames'][frameId]['fieldOutputs']['U'].create_dataset('componentLabels', data=array(['U1', 'U2'], dtype=object))
                    f['steps'][step_name]['frames'][frameId]['fieldOutputs']['U'].create_dataset('validInvariants', data=array(['MAGNITUDE'], dtype=object))
                    f['steps'][step_name]['frames'][frameId]['fieldOutputs']['U'].create_dataset('description', data='Displacement')
                    f['steps'][step_name]['frames'][frameId]['fieldOutputs']['U'].create_dataset('type', data='VECTOR')
                elif self.dimension == 3:
                    value = assembly.dof_solution.reshape(-1, len(props.dof.names))[:, 0:3]
                    f['steps'][step_name]['frames'][frameId]['fieldOutputs'].create_group('U')
                    f['steps'][step_name]['frames'][frameId]['fieldOutputs']['U'].create_dataset('bulkDataBlocks', data=value)
                    f['steps'][step_name]['frames'][frameId]['fieldOutputs']['U'].create_dataset('componentLabels',
                                                                                                 data=array(['U1', 'U2', 'U3'], dtype=object))
                    f['steps'][step_name]['frames'][frameId]['fieldOutputs']['U'].create_dataset('validInvariants', data=array(['MAGNITUDE'], dtype=object))
                    f['steps'][step_name]['frames'][frameId]['fieldOutputs']['U'].create_dataset('description', data='Displacement')
                    f['steps'][step_name]['frames'][frameId]['fieldOutputs']['U'].create_dataset('type', data='VECTOR')
                else:
                    raise NotImplementedError

                if self.dimension == 2:
                    value = assembly.fint.reshape(-1, len(props.dof.names))[:, 0:2]
                    f['steps'][step_name]['frames'][frameId]['fieldOutputs'].create_group('RF')
                    f['steps'][step_name]['frames'][frameId]['fieldOutputs']['RF'].create_dataset('bulkDataBlocks', data=value)
                    f['steps'][step_name]['frames'][frameId]['fieldOutputs']['RF'].create_dataset('componentLabels', data=array(['RF1', 'RF2'], dtype=object))
                    f['steps'][step_name]['frames'][frameId]['fieldOutputs']['RF'].create_dataset('validInvariants', data=array(['MAGNITUDE'], dtype=object))
                    f['steps'][step_name]['frames'][frameId]['fieldOutputs']['RF'].create_dataset('description', data='Displacement')
                    f['steps'][step_name]['frames'][frameId]['fieldOutputs']['RF'].create_dataset('type', data='VECTOR')
                elif self.dimension == 3:
                    value = assembly.fint.reshape(-1, len(props.dof.names))[:, 0:3]
                    f['steps'][step_name]['frames'][frameId]['fieldOutputs'].create_group('RF')
                    f['steps'][step_name]['frames'][frameId]['fieldOutputs']['RF'].create_dataset('bulkDataBlocks', data=value)
                    f['steps'][step_name]['frames'][frameId]['fieldOutputs']['RF'].create_dataset('componentLabels',
                                                                                                  data=array(['RF1', 'RF2', 'RF3'], dtype=object))
                    f['steps'][step_name]['frames'][frameId]['fieldOutputs']['RF'].create_dataset('validInvariants', data=array(['MAGNITUDE'], dtype=object))
                    f['steps'][step_name]['frames'][frameId]['fieldOutputs']['RF'].create_dataset('description', data='Displacement')
                    f['steps'][step_name]['frames'][frameId]['fieldOutputs']['RF'].create_dataset('type', data='VECTOR')
                else:
                    raise NotImplementedError

            if "phi" in props.dof.names:
                col_phi = props.dof.names.index("phi")
                dof_phi = assembly.dof_solution.reshape(-1, len(props.dof.names))[:, col_phi]
                f['steps'][step_name]['frames'][frameId]['fieldOutputs'].create_group('PHI')
                f['steps'][step_name]['frames'][frameId]['fieldOutputs']['PHI'].create_dataset('bulkDataBlocks', data=dof_phi)
                f['steps'][step_name]['frames'][frameId]['fieldOutputs']['PHI'].create_dataset('componentLabels', data=())
                f['steps'][step_name]['frames'][frameId]['fieldOutputs']['PHI'].create_dataset('validInvariants', data=())
                f['steps'][step_name]['frames'][frameId]['fieldOutputs']['PHI'].create_dataset('description', data='Phasefield')
                f['steps'][step_name]['frames'][frameId]['fieldOutputs']['PHI'].create_dataset('type', data='SCALAR')

            for key, value in assembly.field_variables.items():
                f['steps'][step_name]['frames'][frameId]['fieldOutputs'].create_group(key)
                f['steps'][step_name]['frames'][frameId]['fieldOutputs'][key].create_dataset('bulkDataBlocks', data=value)
                f['steps'][step_name]['frames'][frameId]['fieldOutputs'][key].create_dataset('componentLabels', data=array([''], dtype=object))
                f['steps'][step_name]['frames'][frameId]['fieldOutputs'][key].create_dataset('validInvariants', data=array([''], dtype=object))
                f['steps'][step_name]['frames'][frameId]['fieldOutputs'][key].create_dataset('description', data=key)
                f['steps'][step_name]['frames'][frameId]['fieldOutputs'][key].create_dataset('type', data='SCALAR')

        f.close()


if __name__ == "__main__":
    # print_slots_dict(Database.__slots_dict__)

    from pyfem.io.Properties import Properties

    props = Properties()
    props.read_file(r'..\..\..\examples\mechanical\plane\Job-1.toml')
    assembly = Assembly(props)

    database = Database(assembly)

    database.init_hdf5()
    database.add_hdf5()
    # print(database.elements)

    # database.show()
