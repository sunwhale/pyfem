class ElementDiagram:
    tetra4 = r"""
         4
        /|\
       / | \
      /  |  \
     /   |   \
    1----|----3
     \   |   /
      \  |  /
       \ | /
        \|/
         2
    """

    hexahedron8 = r"""
        7---------------6
       /|              /|
      / |     x1 x2   / |
     /  |     | /    /  |
    4---+-----|/----5   |
    |   |     o--x0 |   |
    |   3-----------+---2
    |  /            |  /
    | /             | /
    |/              |/
    0---------------1
    """

    quad4 = r"""
    3---------------2
    |       x1      |
    |       |       |
    |       o--x0   |
    |               |
    |               |
    0---------------1
    """

    quad8 = r"""
    3-------6-------2
    |       x1      |
    |       |       |
    7       o--x0   5
    |               |
    |               |
    0-------4-------1
    """

if __name__=="__main__":
    element_diagram = ElementDiagram()
    print(element_diagram.quad4)
    print(element_diagram.hexahedron8)
