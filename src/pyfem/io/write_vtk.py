# -*- coding: utf-8 -*-
"""

"""
from xml.etree.ElementTree import ElementTree, Element, SubElement

from pyfem.assembly.Assembly import Assembly


def write_vtk(assembly: Assembly) -> None:
    """
    将计算结果过写入vtk文件。
    """
    props = assembly.props
    timer = assembly.timer
    dimension = props.mesh_data.dimension

    root = Element("VTKFile", {
        "type": "UnstructuredGrid",
        "version": "0.1",
        "byte_order": "LittleEndian"
    })

    ugrid = SubElement(root, "UnstructuredGrid")
    piece = SubElement(ugrid, "Piece", {
        "NumberOfPoints": str(len(props.mesh_data.nodes)),
        "NumberOfCells": str(len(props.mesh_data.elements))
    })

    # 添加节点数据
    point_data = SubElement(piece, "PointData")

    if "T" in props.dof.names:
        temp = SubElement(point_data, "DataArray", {
            "type": "Float64",
            "Name": "Temperature",
            "NumberOfComponents": "1",
            "format": "ascii"
        })
        temp.text = ""
        col_T = props.dof.names.index("T")
        dof_T = assembly.dof_solution.reshape(-1, len(props.dof.names))[:, col_T]
        for T in dof_T:
            temp.text += f"{T} \n"

    if "phi" in props.dof.names:
        temp = SubElement(point_data, "DataArray", {
            "type": "Float64",
            "Name": "PhaseField",
            "NumberOfComponents": "1",
            "format": "ascii"
        })
        temp.text = ""
        col_phi = props.dof.names.index("phi")
        dof_phi = assembly.dof_solution.reshape(-1, len(props.dof.names))[:, col_phi]
        for phi in dof_phi:
            temp.text += f"{phi} \n"

    if "u1" in props.dof.names:
        disp = SubElement(point_data, "DataArray", {
            "type": "Float64",
            "Name": "Displacement",
            "NumberOfComponents": "3",
            "format": "ascii"
        })
        disp.text = ""
        if dimension == 2:
            for u1, u2 in assembly.dof_solution.reshape(-1, len(props.dof.names))[:, 0:2]:
                disp.text += f"{u1} {u2} 0.0 \n"
        elif dimension == 3:
            for u1, u2, u3 in assembly.dof_solution.reshape(-1, len(props.dof.names))[:, 0:3]:
                disp.text += f"{u1} {u2} {u3} \n"
        else:
            raise NotImplementedError

    for field_name, field_values in assembly.field_variables.items():
        field = SubElement(point_data, "DataArray", {
            "type": "Float64",
            "Name": field_name,
            "NumberOfComponents": "1",
            "format": "ascii"
        })
        field.text = ""
        for value in field_values:
            field.text += f"{value}\n"

    #
    points = SubElement(piece, "Points")
    node_coords = SubElement(points, "DataArray", {
        "type": "Float64",
        "NumberOfComponents": "3",
        "format": "ascii"
    })
    node_coords.text = ""
    for coord in props.mesh_data.nodes:
        if dimension == 2:
            node_coords.text += " ".join("{:.6f}".format(c) for c in coord) + " 0.0\n"
        elif dimension == 3:
            node_coords.text += " ".join("{:.6f}".format(c) for c in coord) + "\n"
        else:
            raise NotImplementedError

    #
    cells = SubElement(piece, "Cells")
    conn_elem = SubElement(cells, "DataArray", {
        "type": "Int32",
        "Name": "connectivity",
        "format": "ascii"
    })
    offset_elem = SubElement(cells, "DataArray", {
        "type": "Int32",
        "Name": "offsets",
        "format": "ascii"
    })
    types_elem = SubElement(cells, "DataArray", {
        "type": "UInt8",
        "Name": "types",
        "format": "ascii"
    })
    conn_elem.text = ""
    offset_elem.text = ""
    types_elem.text = ""
    offset = 0
    for connectivity in props.mesh_data.elements:
        conn_elem.text += " ".join(str(node_id) for node_id in connectivity) + "\n"
        offset += len(connectivity)
        offset_elem.text += "{}\n".format(offset)
        if dimension == 2:
            types_elem.text += "9\n"
        elif dimension == 3:
            types_elem.text += "12\n"  # 12表示六面体单元类型

    tree = ElementTree(root)

    job_name = props.input_file.stem

    output_file = props.work_path.joinpath(f'{job_name}-{timer.increment}.vtu')

    tree.write(output_file, xml_declaration=True, encoding='utf-8')


def write_pvd(assembly: Assembly) -> None:
    """
    将多个vtk文件信息写入pvd文件。
    """
    timer = assembly.timer
    job_name = assembly.props.input_file.stem
    output_file = assembly.props.work_path.joinpath(f'{job_name}.pvd')

    with open(output_file, 'w') as f:
        f.write("<VTKFile byte_order='LittleEndian' type='Collection' version='0.1'>\n")
        f.write("<Collection>\n")

        for frame in timer.frame_ids:
            f.write("<DataSet file='" + f"{job_name}-{frame}.vtu" + "' groups='' part='0' timestep='" + str(
                frame) + "'/>\n")

        f.write("</Collection>\n")
        f.write("</VTKFile>\n")


if __name__ == "__main__":
    pass
