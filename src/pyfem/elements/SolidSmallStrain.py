from numpy import zeros, dot, array

from pyfem.elements.BaseElement import BaseElement
from pyfem.io.Dofs import Dofs
from pyfem.io.Material import Material
from pyfem.io.Section import Section
from pyfem.utils.wrappers import show_running_time
from pyfem.elements.IsoElementShape import IsoElementShape
from pyfem.materials.PlaneStress import PlaneStress


class PlaneSmallStress(BaseElement):

    def __init__(self, iso_element_shape: IsoElementShape, section: Section, material: Material, material_tangent: PlaneStress):
        super().__init__(iso_element_shape)
        self.material = material
        self.material_tangent = material_tangent
        self.section = section
        # self.dofs = dofs

    def get_b_matrix(self):

        b = zeros(shape=(4, 3, 8))

        for gp_shape_gradient, gp_jacobi_inv in zip(self.iso_element_shape.gp_shape_gradients, self.gp_jacobi_inv):
            dhdx = dot(gp_shape_gradient, gp_jacobi_inv)

        print(dhdx.shape)

        dhdx = dot(self.iso_element_shape.gp_shape_gradients, self.jacobi_inv)

        print(dhdx.shape)

        # for i, dp in enumerate(dhdx):
        #
        #     b[0, i * 2 + 0] = dp[0]
        #     b[1, i * 2 + 1] = dp[1]
        #     b[2, i * 2 + 0] = dp[1]
        #     b[2, i * 2 + 1] = dp[0]

        return b


@show_running_time
def main():
    iso_element_shapes = {
        'quad4': IsoElementShape('quad4'),
        'line2': IsoElementShape('line2')
    }
    from pyfem.io.Properties import Properties

    props = Properties()
    props.read_file(r'F:\Github\pyfem\examples\rectangle\rectangle.toml')

    elements = props.elements
    nodes = props.nodes
    materials = props.materials

    # print(PlaneStress(materials[0]).to_string())

    # print(elements.to_string(level=0))
    # print(props.sections[0].element_sets)

    element_list = []

    section_of_element_set = {}
    for element_set in elements.element_sets:
        for section in props.sections:
            if element_set in section.element_sets:
                section_of_element_set[element_set] = section

    # print(section_of_element_set)

    for element_set_name, element_set in elements.element_sets.items():
        section = props.sections[0]
        material = props.materials[0]
        material_stiffness = PlaneStress(material)
        for element_id in element_set:
            connectivity = elements[element_id]
            if len(connectivity) == 4:
                iso_quad4 = iso_element_shapes['quad4']
                element_object = PlaneSmallStress(iso_quad4, section, material, material_stiffness)
                element_object.connectivity = connectivity
                node_coords = nodes.get_items_by_ids(list(connectivity))
                element_object.node_coords = array(node_coords)
                element_object.cal_jacobi()
                element_list.append(element_object)

    # print(element_list[0].iso_element_shape.to_string())
    # print(element_list[0].iso_element_shape.gp_shape_gradients[0])
    # print(element_list[0].jacobi_inv[0])

    element_list[0].get_b_matrix()

if __name__ == "__main__":
    main()

    # def getTangentStiffness(self, elemdat):
    #
    #     shape_data = get_element_shape_data(elemdat.coords)
    #
    #     elemdat.outlabel.append(self.outputLabels)
    #     elemdat.outdata = zeros(shape=(len(elemdat.nodes), self.nstr))
    #
    #     for iData in shape_data:
    #         b = self.getBmatrix(iData.dhdx)
    #
    #         self.kin.strain = dot(b, elemdat.state)
    #         self.kin.dstrain = dot(b, elemdat.dstate)
    #
    #         sigma, tang = self.mat.getStress(self.kin)
    #
    #         elemdat.stiff += dot(b.transpose(), dot(tang, b)) * iData.weight
    #         elemdat.fint += dot(b.transpose(), sigma) * iData.weight
    #
    #         self.appendNodalOutput(self.mat.outLabels(), self.mat.outData())
    #
    # # -------------------------------------------------------------------------
    #
    # def getInternalForce(self, elemdat):
    #
    #     shape_data = get_element_shape_data(elemdat.coords)
    #
    #     elemdat.outlabel.append(self.outputLabels)
    #     elemdat.outdata = zeros(shape=(len(elemdat.nodes), self.nstr))
    #
    #     for iData in shape_data:
    #         b = self.getBmatrix(iData.dhdx)
    #
    #         self.kin.strain = dot(b, elemdat.state)
    #         self.kin.dstrain = dot(b, elemdat.dstate)
    #
    #         sigma, tang = self.mat.getStress(self.kin)
    #
    #         elemdat.fint += dot(b.transpose(), sigma) * iData.weight
    #
    #         self.appendNodalOutput(self.mat.outLabels(), self.mat.outData())
    #
    # # -------------------------------------------------------------------------------
    #
    # def getDissipation(self, elemdat):
    #
    #     shape_data = get_element_shape_data(elemdat.coords)
    #
    #     for iData in shape_data:
    #         b = self.getBmatrix(iData.dhdx)
    #
    #         self.kin.strain = dot(b, elemdat.state)
    #         self.kin.dstrain = dot(b, elemdat.dstate)
    #
    #         self.mat.getStress(self.kin)
    #
    #         self.kin.dgdstrain = zeros(3)
    #         self.kin.g = 0.0
    #
    #         elemdat.fint += dot(b.transpose(), self.kin.dgdstrain) * iData.weight
    #         elemdat.diss += self.kin.g * iData.weight
    #
    #
    #
    # def getMassMatrix(self, elemdat):
    #
    #     shape_data = get_element_shape_data(elemdat.coords)
    #
    #     rho = elemdat.matprops.rho
    #
    #     for iData in shape_data:
    #         N = self.getNmatrix(iData.h)
    #         elemdat.mass += dot(N.transpose(), N) * rho * iData.weight
    #
    #     elemdat.lumped = sum(elemdat.mass)
    #
    # # --------------------------------------------------------------------------
    #
    # def getBmatrix(self, dphi):
    #
    #     b = zeros(shape=(self.nstr, self.dofCount()))
    #
    #     if self.rank == 2:
    #         for i, dp in enumerate(dphi):
    #             b[0, i * 2 + 0] = dp[0]
    #             b[1, i * 2 + 1] = dp[1]
    #             b[2, i * 2 + 0] = dp[1]
    #             b[2, i * 2 + 1] = dp[0]
    #     elif self.rank == 3:
    #         for i, dp in enumerate(dphi):
    #             b[0, i * 3 + 0] = dp[0]
    #             b[1, i * 3 + 1] = dp[1]
    #             b[2, i * 3 + 2] = dp[2]
    #
    #             b[3, i * 3 + 1] = dp[2]
    #             b[3, i * 3 + 2] = dp[1]
    #
    #             b[4, i * 3 + 0] = dp[2]
    #             b[4, i * 3 + 2] = dp[0]
    #
    #             b[5, i * 3 + 0] = dp[1]
    #             b[5, i * 3 + 1] = dp[0]
    #
    #     return b
    #
    # # ------------------------------------------------------------------------------
    #
    # def getNmatrix(self, h):
    #
    #     N = zeros(shape=(self.rank, self.rank * len(h)))
    #
    #     for i, a in enumerate(h):
    #         for j in list(range(self.rank)):
    #             N[j, self.rank * i + j] = a
    #
    #     return N
