class IsoElementDiagram:
    line2 = r"""
    0---------------1
            +-->x0"""

    line3 = r"""
    0-------1-------2
            +-->x0"""

    tria3 = r"""
    2
    * *
    *   *
    *     *
    x1      *
    |         *
    0--x0 * * * 1"""

    quad4 = r"""
    3---------------2
    |       x1      |
    |       |       |
    |       +--x0   |
    |               |
    |               |
    0---------------1"""

    quad8 = r"""
    3-------6-------2
    |       x1      |
    |       |       |
    7       +--x0   5
    |               |
    |               |
    0-------4-------1"""

    quad9 = r"""
    3-------6-------2
    |       x1      |
    |       |       |
    7       9--x0   5
    |               |
    |               |
    0-------4-------1"""

    # tetra4 = r"""
    # 3
    # * * *
    # *   *   *
    # *     *     *
    # *       *       2
    # *         *    * *
    # *           **    *
    # *          *  *    *
    # *        *      *   *
    # *      *          *  *
    # x2   x1             * *
    # |  *                  **
    # 0--x0 * * * * * * * * * 1"""

    tetra4 = r"""
    3
    * **
    *   * *
    *     *  *
    *       *   2
    *        **  *
    x2    *     * *
    |  x1         **
    0--x0 * * * * * 1"""

    hex8 = r"""
        7---------------6
       /|              /|
      / |     x2 x1   / |
     /  |     | /    /  |
    4---+-----|/----5   |
    |   |     +--x0 |   |
    |   3-----------+---2
    |  /            |  /
    | /             | /
    |/              |/
    0---------------1"""


if __name__ == "__main__":
    print(IsoElementDiagram.quad4)
    print(IsoElementDiagram.tetra4)
    print(IsoElementDiagram.hex8)