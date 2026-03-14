import os
import numpy as np
from simnibs.mesh_tools import mesh_io
from simnibs import SIMNIBSDIR
from simnibs.simulation.onlinefem import FemTargetPointCloud


class TestRegionOfInterest:
    def test_RegionOfInterestInitializer_custom_center(self):
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
        fn_mesh = os.path.join(
            SIMNIBSDIR, "_internal_resources", "testing_files", "sphere3.msh"
        )
        mesh = mesh_io.read_msh(fn_mesh)
        roi = FemTargetPointCloud(center=center, mesh=mesh)

    def test_RegionOfInterestInitializer_custom_domains(self):
        fn_mesh = os.path.join(
            SIMNIBSDIR, "_internal_resources", "testing_files", "sphere3.msh"
        )
        mesh = mesh_io.read_msh(fn_mesh)
        roi = FemTargetPointCloud(
            mesh,
            mesh.elements_baricenters()[mesh.elm.get_tags(3) | mesh.elm.get_tags(4)],
        )
