from xml.etree.ElementTree import ElementTree, Element, SubElement

from pyfem.assembly.Assembly import Assembly
from pyfem.io.Properties import Properties


def write_vtk(props: Properties, assembly: Assembly):
    root = Element("VTKFile", {
        "type": "UnstructuredGrid",
        "version": "0.1",
        "byte_order": "LittleEndian"
    })

    ugrid = SubElement(root, "UnstructuredGrid")
    piece = SubElement(ugrid, "Piece", {
        "NumberOfPoints": str(len(props.nodes)),
        "NumberOfCells": str(len(props.elements))
    })

    # 添加节点数据
    point_data = SubElement(piece, "PointData")
    temp = SubElement(point_data, "DataArray", {
        "type": "Float64",
        "Name": "Temperature",
        "NumberOfComponents": "1",
        "format": "ascii"
    })
    temp.text = ""
    for _ in props.nodes.items():
        temp.text += "0.0\n"  # 在这里添加具体的场量数值

    disp = SubElement(point_data, "DataArray", {
        "type": "Float64",
        "Name": "Displacement",
        "NumberOfComponents": "3",
        "format": "ascii"
    })
    disp.text = ""
    for u1, u2 in assembly.dof_solution.reshape(-1, 2):
        disp.text += f"{u1} {u2} 0.0 \n"

    # for u1, u2, u3 in assembly.dof_solution.reshape(-1, 3):
    #     disp.text += f"{u1} {u2} {u3} \n"

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
    for _, coord in props.nodes.items():
        if len(coord) == 2:
            node_coords.text += " ".join("{:.6f}".format(c) for c in coord) + " 0.0\n"
        else:
            node_coords.text += " ".join("{:.6f}".format(c) for c in coord) + "\n"

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
    for _, connectivity in props.elements.items():
        conn_elem.text += " ".join(str(node_id) for node_id in connectivity) + "\n"
        offset += len(connectivity)
        offset_elem.text += "{}\n".format(offset)
        types_elem.text += "9\n"  # 12表示六面体单元类型

    tree = ElementTree(root)

    job_name = props.input_file.stem

    output_file = props.work_path.joinpath(job_name + '.vtu')

    tree.write(output_file, xml_declaration=True, encoding='utf-8')


if __name__ == "__main__":
    pass
