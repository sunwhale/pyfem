from numpy import zeros

from pyfem.materials.MaterialManager import MaterialManager


class Element(list):
    dof_types = []

    def __init__(self, elnodes, props):
        list.__init__(self, elnodes)

        self.family = "CONTINUUM"
        self.history = {}
        self.current = {}
        self.solver_status = props.solver_status

        for name, val in props:
            if name == "material":
                self.matProps = val

                self.matProps.rank = props.rank
                self.matProps.solver_status = self.solver_status
                self.mat = MaterialManager(self.matProps)

            setattr(self, name, val)

    def dofCount(self):

        return len(self) * len(self.dof_types)

    def getNodes(self):
        return self

    def get_type(self):
        return self.element_type

    def appendNodalOutput(self, labels, data, weight=1.0):

        for i, name in enumerate(labels):
            if not hasattr(self.globdat, name):
                self.globdat.outputNames.append(name)

                setattr(self.globdat, name, zeros(len(self.globdat.nodes)))
                setattr(self.globdat, name + 'Weights', zeros(len(self.globdat.nodes)))

            outMat = getattr(self.globdat, name)
            outWeights = getattr(self.globdat, name + 'Weights')

            if data.ndim == 1:
                for idx in self.globdat.nodes.get_indices_by_ids(self):
                    outMat[idx] += data[i]
                    outWeights[idx] += weight
            else:
                for j, idx in enumerate(self.globdat.nodes.get_indices_by_ids(self)):
                    outMat[idx] += data[j, i]
                    outWeights[idx] += weight

    def set_history_parameter(self, name, val):
        self.current[name] = val

    def get_history_parameter(self, name):
        return self.history[name]

    def commit_history(self):
        self.history = self.current.copy()
        self.current = {}

        if hasattr(self, "mat"):
            self.mat.commit_history()

    def commit(self, elemdat):
        pass

    def loadFactor(self):

        return self.solver_status.lam



