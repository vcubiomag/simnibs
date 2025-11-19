import numpy as np

from simnibs.simulation.onlinefem import FemTargetPointCloud




class TestRegionOfInterest:
    def test_RegionOfInterestInitializer_custom_center(self, sphere3_msh):
        nodes = np.array(
            [
                [-1.2, 1.4, 7.1],
                [-0.9, 1.4, 7.2],
                [-1.0, 1.3, 7.1],
                [-0.6, 1.3, 7.2],
                [-0.7, 1.2, 7.1],
            ]
        )
        con = np.array([[0, 1, 2], [2, 3, 1], [4, 3, 2]])
        center = np.mean(nodes[con,], axis=1)
        roi = FemTargetPointCloud(center=center, mesh=sphere3_msh)

    def test_RegionOfInterestInitializer_custom_domains(self, sphere3_msh):
        roi = FemTargetPointCloud(
            sphere3_msh,
            sphere3_msh.elements_baricenters()[(sphere3_msh.elm.tag1 == 3) | (sphere3_msh.elm.tag1 == 4)],
        )
