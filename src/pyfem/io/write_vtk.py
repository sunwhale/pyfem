from xml.etree.ElementTree import ElementTree, Element, SubElement

from pyfem.io.Properties import Properties


def write_vtk(props: Properties, x):
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
    for u1, u2 in x.reshape(-1, 2):
        disp.text += f"{u1} {u2} 0.0 \n"

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
    tree.write("Job-1.vtu", xml_declaration=True, encoding='utf-8')


if __name__ == "__main__":
    from pyfem.io.Properties import Properties
    from pyfem.assembly.Assembly import Assembly
    from scipy.sparse.linalg import spsolve  # type: ignore

    props = Properties()
    props.read_file(r'F:\Github\pyfem\examples\rectangle\rectangle.toml')
    props.verify()
    # props.show()
    assembly = Assembly(props)

    assembly.show()

    A = assembly.global_stiffness
    b = assembly.fext

    from time import time

    t1 = time()
    x = spsolve(A, b)

    write_vtk(props, x)
