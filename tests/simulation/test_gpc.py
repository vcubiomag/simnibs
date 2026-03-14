import numpy as np
import h5py
import os
import tempfile
import functools
from mock import Mock, patch
from copy import deepcopy
import pytest
import pygpc
from collections import OrderedDict

from simnibs import SIMNIBSDIR
from simnibs.simulation import fem, sim_struct, gpc as simnibs_gpc
from simnibs.mesh_tools import mesh_io
from simnibs.simulation.sim_struct import TDCSLIST


@pytest.fixture(scope="module")
def sphere3():
    return mesh_io.read_msh(
        os.path.join(SIMNIBSDIR, "_internal_resources", "testing_files", "sphere3.msh")
    )


@pytest.fixture
def cube_msh():
    fn = os.path.join(
        SIMNIBSDIR, "_internal_resources", "testing_files", "cube_w_electrodes.msh"
    )
    return mesh_io.read_msh(fn)


@pytest.fixture(scope="module")
def gpc_regression_instance():
    random_vars = [0, 1, 2]
    pdftype = ['beta', 'uniform', 'normal']
    pdfparas = [[3,3,-2,2],[-1, 1], [0,2]]
    poly_idx = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    coords_norm = np.array([[0.5, 0.2, 0.2], [0.1, 0.9, 0.6], [0.1, 0.3, 0.6]])
    model = pygpc.testfunctions.Dummy()
    parameters = simnibs_gpc.prep_parameters(random_vars=random_vars, pdf_types=pdftype, pdf_paras=pdfparas)
    problem = pygpc.Problem(model, parameters)
    regobj = simnibs_gpc.gPC_regression(problem,0,poly_idx,coords_norm,'TMS')
    return regobj


class TestHDF5:
    def test_write_hdf5(self):
        if os.path.isfile("test.hdf5"):
            os.remove("test.hdf5")
        simnibs_gpc.write_data_hdf5(np.ones((10, 1)), "ones", "test.hdf5", "path/to/")
        with h5py.File("test.hdf5", "r") as f:
            data = f["path/to/ones"][()]
        os.remove("test.hdf5")
        assert np.all(data == 1)

    def test_read_hdf5(self):
        if os.path.isfile("test.hdf5"):
            os.remove("test.hdf5")
        with h5py.File("test.hdf5", "a") as f:
            f.create_dataset("path/to/ones", data=np.ones((10, 1)))
        data = simnibs_gpc.read_data_hdf5("ones", "test.hdf5", "path/to/")
        os.remove("test.hdf5")
        assert np.all(data == 1)


class TestgPC_Regression:
    def test_gPC_Regression_initialization(self, gpc_regression_instance):
        parameters = {'cond_0': {'pdf_type':'beta', 'pdf_shape': [3,3], 'pdf_limits':[-2,2]},
                      'cond_1': {'pdf_type':'beta', 'pdf_shape': [1,1], 'pdf_limits':[-1,1]},
                      'cond_2': {'pdf_type':'norm', 'pdf_shape': [0,2]}}
        poly_idx = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        coords_norm = np.array([[0.5, 0.2, 0.2], [0.1, 0.9, 0.6], [0.1, 0.3, 0.6]])
        np.testing.assert_equal(gpc_regression_instance.multi_indices, poly_idx)
        np.testing.assert_equal(gpc_regression_instance.coords_norm, coords_norm)
        for key, props in parameters.items():
            param = gpc_regression_instance.problem.parameters_random[key]
            assert param.pdf_type == props['pdf_type']
            np.testing.assert_equal(param.pdf_shape, props['pdf_shape'])
            if param.pdf_type != 'norm':
                np.testing.assert_equal(param.pdf_limits, props['pdf_limits'])

    def test_save_hdf5(self, gpc_regression_instance):
        if os.path.isfile("test.hdf5"):
            os.remove("test.hdf5")

        gpc_regression_instance.save_hdf5('test.hdf5')
        with h5py.File('test.hdf5', 'r') as f:
            assert np.all(f['gpc_object/random_vars'][()] == [b"0", b"1", b"2"])
            assert np.all(f['gpc_object/pdftype'][()] == [b"beta", b"beta", b"norm"])
            assert np.allclose(f['gpc_object/pdfshape'][()], np.array([[3, 3], [1, 1], [0, 2]]))
            assert np.allclose(f['gpc_object/limits'], np.array([[-2.0, 2.0], [-1, 1], [-1e10, 1e10]]))
            assert np.allclose(f['gpc_object/poly_idx'],
                               np.array(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])))
            assert np.allclose(f['gpc_object/grid/coords_norm'],
                               np.array([[0.5, 0.2, 0.2], [0.1, 0.9, 0.6], [0.1, 0.3, 0.6]]))

        os.remove("test.hdf5")

    def test_load_hdf5(self, gpc_regression_instance):
        if os.path.isfile('test.hdf5'):
            os.remove('test.hdf5')
        gpc_regression_instance.save_hdf5('test.hdf5')
        gpc_regression = simnibs_gpc.gPC_regression.read_hdf5('test.hdf5')
        for key, props in gpc_regression.problem.parameters_random.items():
            param = gpc_regression_instance.problem.parameters_random[key]
            assert param.pdf_type == props.pdf_type
            np.testing.assert_equal(param.pdf_shape, props.pdf_shape)
            if param.pdf_type != 'norm':
                np.testing.assert_equal(param.pdf_limits, props.pdf_limits)
        np.testing.assert_equal(gpc_regression_instance.multi_indices,
                                gpc_regression.multi_indices)
        np.testing.assert_equal(gpc_regression_instance.coords_norm,
                                gpc_regression.coords_norm)
        os.remove('test.hdf5')

    def test_postprocessing(self, gpc_regression_instance, sphere3):
        fn = "gpc_testing.hdf5"
        if os.path.isfile(fn):
            os.remove(fn)
        msh = sphere3.crop_mesh(elm_type=4)
        msh.write_hdf5(fn, "mesh_roi/")
        # "normalized" coordinates
        coords_norm = np.array([[-0.5], [0], [0.7]])
        # Define regression object.
        # The random variable is the conductivity of the layer 3
        # Uniform distribution, between 0 and 10
        model = pygpc.testfunctions.Dummy()
        parameters = OrderedDict()
        parameters['cond_3'] = pygpc.Beta([1,1],[0,10])
        problem = pygpc.Problem(model, parameters)
        gpc_reg = \
            simnibs_gpc.gPC_regression(problem,[0],[[0],[1]],coords_norm,
                                       'TCS', data_file=fn)
        # Create potentials
        pot = np.tile(msh.nodes.node_coord[None, :, 0], (coords_norm.shape[0], 1))
        lst = sim_struct.SimuList()
        lst.cond[3].value = 1  # Layer 4 conductivity
        lst.cond[4].value = 10  # Layer 5 conductivity
        lst._write_conductivity_to_hdf5(fn)
        for i, c in enumerate(gpc_reg.grid.coords):
            pot[i] *= c  # Linear relationship between potential and random variable
        #  Record potentials
        with h5py.File(fn, "a") as f:
            f.create_group("mesh_roi/data_matrices")
            f["mesh_roi/data_matrices"].create_dataset("v_samples", data=pot)
        #  Postprocess
        gpc_reg.postprocessing("eEjJ")
        with h5py.File(fn, "r") as f:
            # Linear relationship between random variable and potential, uniform
            # distribution
            mean_E = f["mesh_roi/elmdata/E_mean"][()]
            mean_E_expected = -np.array([5, 0, 0]) * 1e3
            assert np.allclose(mean_E, mean_E_expected)
            mean_e = f["mesh_roi/elmdata/magnE_mean"][()]
            assert np.allclose(mean_e, 5e3)
            # J is a bit more complicated
            mean_J = f['mesh_roi/elmdata/J_mean'][()]
            assert np.allclose(mean_J[msh.elm.tag1==4],
                               mean_E_expected)
            assert np.allclose(mean_J[msh.elm.tag1==5],
                               10 * mean_E_expected)
            coeffs, _ = gpc_reg.expand(gpc_reg.grid.coords ** 2)
            mean = gpc_reg.get_mean(coeffs)
            assert np.allclose(mean_J[msh.elm.tag1==3],
                               mean * -np.array([1, 0, 0]) * 1e3)
            dsets = f['mesh_roi/elmdata/'].keys()

            std_E = f["mesh_roi/elmdata/E_std"][()]
            assert np.allclose(std_E, np.array([np.sqrt(1e8 / 12), 0.0, 0.0]))
            assert "magnE_std" in dsets
            assert "J_std" in dsets

            assert "E_sobol_3" in dsets
            assert "magnE_sobol_3" in dsets
            assert "J_sobol_3" in dsets

            assert "E_sensitivity_3" in dsets
            assert "magnE_sensitivity_3" in dsets
            assert "J_sensitivity_3" in dsets

        os.remove(fn)


@pytest.fixture
def sampler_args(sphere3):
    poslist = sim_struct.SimuList()

    poslist.cond[2].name = "inner"
    poslist.cond[2].distribution_type = "beta"
    poslist.cond[2].distribution_parameters = [2, 3, 0.3, 0.4]
    poslist.cond[3].name = "middle"
    poslist.cond[3].value = 1
    poslist.cond[4].name = "outer"
    poslist.cond[4].value = 10
    with tempfile.NamedTemporaryFile(suffix=".hdf5") as f:
        fn_hdf5 = f.name
    if os.path.isfile(fn_hdf5):
        os.remove(fn_hdf5)

    return sphere3, poslist, fn_hdf5, [3]


class TestSampler:
    def test_set_up_sampler(self, sampler_args):
        mesh, poslist, fn_hdf5, roi = sampler_args
        S = simnibs_gpc.gPCSampler(mesh, poslist, fn_hdf5, roi)

        assert S.mesh_roi.elm.get_tags(3).all()

        S.create_hdf5()
        with h5py.File(fn_hdf5, "r") as f:
            assert f["mesh"]
            assert f["mesh_roi"]
            assert f["cond"]
            assert np.all(f["roi"][()] == 3)

        S2 = simnibs_gpc.gPCSampler.load_hdf5(fn_hdf5)
        assert S2.mesh.nodes.nr == mesh.nodes.nr
        assert S2.mesh.elm.nr == mesh.elm.nr
        assert S2.roi == roi
        for i, c in enumerate(S2.poslist.cond):
            assert c.name == poslist.cond[i].name
            assert c.value == poslist.cond[i].value
            assert c.distribution_type == poslist.cond[i].distribution_type
            assert c.distribution_parameters == poslist.cond[i].distribution_parameters

    def test_record_data_matrix(self, sampler_args):
        mesh, poslist, fn_hdf5, roi = sampler_args
        S = simnibs_gpc.gPCSampler(mesh, poslist, fn_hdf5, roi)

        rand_vars1 = [0.1, 0.2]
        potential1 = np.random.rand(100)
        E1 = np.random.rand(100, 3)
        S.record_data_matrix(potential1, "potential", "data")
        S.record_data_matrix(rand_vars1, "random_vars", "data")
        S.record_data_matrix(E1, "E", "data")

        rand_vars2 = [0.2, 0.4]
        potential2 = np.random.rand(100)
        E2 = np.random.rand(100, 3)
        S.record_data_matrix(potential2, "potential", "data")
        S.record_data_matrix(rand_vars2, "random_vars", "data")
        S.record_data_matrix(E2, "E", "data")

        with h5py.File(fn_hdf5, "r") as f:
            assert np.allclose(f["data/random_vars"][0, :], rand_vars1)
            assert np.allclose(f["data/random_vars"][1, :], rand_vars2)
            assert np.allclose(f["data/potential"][0, :], potential1)
            assert np.allclose(f["data/potential"][1, :], potential2)
            assert np.allclose(f["data/E"][0, ...], E1)
            assert np.allclose(f["data/E"][1, ...], E2)

    def test_calc_E(self, sampler_args):
        mesh, poslist, fn_hdf5, roi = sampler_args
        S = simnibs_gpc.gPCSampler(mesh, poslist, fn_hdf5, roi)
        v = mesh_io.NodeData(mesh.nodes.node_coord[:, 0], mesh=mesh)
        E = S._calc_E(v, None)
        assert np.allclose(E, [-1e3, 0, 0])

    def test_update_poslist(self, sampler_args):
        mesh, poslist, fn_hdf5, roi = sampler_args
        S = simnibs_gpc.gPCSampler(mesh, poslist, fn_hdf5, roi)
        parameters = OrderedDict()
        parameters["cond_3"] = 0.35
        poslist = S._update_poslist(parameters)
        assert np.isclose(poslist.cond[2].value, 0.35)
        assert np.isclose(poslist.cond[3].value, 1)
        assert np.isclose(poslist.cond[4].value, 10)

    def test_run_N_random_simulations(self, sampler_args):
        mesh, poslist, fn_hdf5, roi = sampler_args
        S = simnibs_gpc.gPCSampler(mesh, poslist, fn_hdf5, roi)
        S.run_simulation = Mock()
        S.run_N_random_simulations(1000)
        for c in S.run_simulation.call_args_list:
            assert c[0][0][0] <= 0.4
            assert c[0][0][0] >= 0.3

    def test_tdcs_set_up(self, sampler_args):
        mesh, poslist, fn_hdf5, roi = sampler_args
        S = simnibs_gpc.TDCSgPCSampler(
            mesh, poslist, fn_hdf5, [1101, 1102], [-1, 1], roi
        )
        S.create_hdf5()
        with h5py.File(fn_hdf5, "r") as f:
            assert f["mesh"]
            assert f["mesh_roi"]
            assert f["cond"]
            assert np.all(f["roi"][()] == 3)
            assert np.all(f["el_tags"][()] == [1101, 1102])
            assert np.allclose(f["el_currents"][()], [-1, 1])

        S2 = simnibs_gpc.TDCSgPCSampler.load_hdf5(fn_hdf5)
        assert S2.mesh.nodes.nr == mesh.nodes.nr
        assert S2.mesh.elm.nr == mesh.elm.nr
        assert S2.roi == roi
        for i, c in enumerate(S2.poslist.cond):
            assert c.name == poslist.cond[i].name
            assert c.value == poslist.cond[i].value
            assert c.distribution_type == poslist.cond[i].distribution_type
            assert c.distribution_parameters == poslist.cond[i].distribution_parameters

    @patch.object(simnibs_gpc, "fem")
    def test_tdcs_run(self, mock_fem, sampler_args):
        mesh, poslist, fn_hdf5, roi = sampler_args
        v = mesh.nodes.node_coord[:, 0]
        v_roi = mesh.crop_mesh(roi).nodes.node_coord[:, 0]

        mock_fem.tdcs.side_effect = [
            mesh_io.NodeData(v, mesh=mesh),
            mesh_io.NodeData(-v, mesh=mesh),
        ]

        S = simnibs_gpc.TDCSgPCSampler(
            mesh, poslist, fn_hdf5, [1101, 1102], [-1, 1], roi
        )

        parameters = OrderedDict()
        parameters['cond_3'] = 1
        E1 = S.run_simulation(parameters)
        assert E1.shape == (3 * np.sum(mesh.elm.tag1 == 3), )
        assert np.allclose(E1.reshape(-1, 3), [-1e3, 0, 0])

        parameters['cond_3'] = 2
        S.run_simulation(parameters)
        with h5py.File(fn_hdf5, 'r') as f:
            assert np.allclose(f['random_var_samples'][()], [[1], [2]])
            assert np.allclose(f['mesh_roi/data_matrices/v_samples'][0, :], v_roi)
            assert np.allclose(f['mesh_roi/data_matrices/v_samples'][1, :],-v_roi)
            assert np.allclose(f['mesh_roi/data_matrices/E_samples'][0, :],[-1e3, 0., 0.])
            assert np.allclose(f['mesh_roi/data_matrices/E_samples'][1, :],[1e3, 0., 0.])

    def test_tms_set_up(self, sampler_args):
        mesh, poslist, fn_hdf5, roi = sampler_args
        matsimnibs = np.eye(4)
        didt = 1e5
        coil = "coil.nii.gz"
        S = simnibs_gpc.TMSgPCSampler(
            mesh, poslist, fn_hdf5, coil, matsimnibs, didt, roi
        )
        S.create_hdf5()
        with h5py.File(fn_hdf5, "r") as f:
            assert f["mesh"]
            assert f["mesh_roi"]
            assert f["cond"]
            assert np.all(f["roi"][()] == 3)
            assert np.allclose(f["matsimnibs"], np.eye(4))
            assert np.allclose(f["didt"][()], 1e5)
            assert f["fnamecoil"][()] == coil.encode()

        S2 = simnibs_gpc.TMSgPCSampler.load_hdf5(fn_hdf5)
        assert S2.mesh.nodes.nr == mesh.nodes.nr
        assert S2.mesh.elm.nr == mesh.elm.nr
        assert S2.roi == roi
        assert np.allclose(S2.matsimnibs, S.matsimnibs)
        assert np.allclose(S2.didt, S.didt)
        assert S2.fnamecoil == S.fnamecoil
        for i, c in enumerate(S2.poslist.cond):
            assert c.name == poslist.cond[i].name
            assert c.value == poslist.cond[i].value
            assert c.distribution_type == poslist.cond[i].distribution_type
            assert c.distribution_parameters == poslist.cond[i].distribution_parameters




class TestRunGPC:
    def test_prep_gpc(self):
        cond = [sim_struct.COND(), sim_struct.COND(), sim_struct.COND(), sim_struct.COND(), sim_struct.COND()]
        cond[0].distribution_type = 'uniform'
        cond[0].distribution_parameters = [0.2, 0.3]
        cond[2].distribution_type = 'beta'
        cond[2].distribution_parameters = [2, 3, 0.3, 0.4]
        cond[4].distribution_type = 'normal'
        cond[4].distribution_parameters = [0.2, 0.3]
        simlist = sim_struct.SimuList()
        simlist.cond = cond
        parameters_ref = {'cond_1': {'pdf_type':'beta', 'pdf_shape': [1, 1], 'pdf_limits':[0.2, 0.3]},
                          'cond_3': {'pdf_type':'beta', 'pdf_shape': [2, 3], 'pdf_limits':[0.3, 0.4]},
                          'cond_5': {'pdf_type':'norm', 'pdf_shape': [0.2, 0.3]}}
        parameters, random_vars = simnibs_gpc.prep_gpc(simlist)
        assert np.all(random_vars == [1, 3, 5])
        for key, props in parameters_ref.items():
            param = parameters[key]
            assert param.pdf_type == props['pdf_type']
            np.testing.assert_equal(param.pdf_shape, props['pdf_shape'])
            if param.pdf_type != 'norm':
                np.testing.assert_equal(param.pdf_limits, props['pdf_limits'])



class TestRegressionTestGPC:
    def test_regression_test_gpc(self, sphere3):
        tdcs = TDCSLIST()
        tdcs.currents = [0.001, -0.001]
        tdcs.mesh = sphere3

        electrode = tdcs.add_electrode()
        electrode.channelnr = 1
        electrode.centre = [95, 0, 0]
        electrode.shape = "ellipse"
        electrode.dimensions = [20, 20]
        electrode.thickness = 4

        electrode = tdcs.add_electrode()
        electrode.channelnr = 2
        electrode.centre = [-95, 0, 0]
        electrode.shape = "ellipse"
        electrode.dimensions = [20, 20]
        electrode.thickness = 4

        # Set-up the uncertain conductivities
        # intracranial
        tdcs.cond[2].distribution_type = "beta"
        tdcs.cond[2].distribution_parameters = [3, 3, 0.2, 0.4]
        # bone
        tdcs.cond[3].distribution_type = "beta"
        tdcs.cond[3].distribution_parameters = [3, 3, 0.001, 0.012]
        # scalp
        tdcs.cond[4].distribution_type = "beta"
        tdcs.cond[4].distribution_parameters = [3, 3, 0.4, 0.6]

        # Run the UQ calling with intracranial and bone as ROIs and tolerance of 1e-2
        fn_out = tempfile.TemporaryDirectory()
        simnibs_gpc.run_tcs_gpc(tdcs, os.path.join(fn_out.name,'sphere'), tissues=[3, 4], eps=1e-2)

        with h5py.File(os.path.join(fn_out.name,'sphere_gpc.hdf5'), "r") as f:
            assert len(f.keys()) == 8
            
            hlpVar = f['random_var_samples'][()]
            assert hlpVar.shape[0] <= 16

            hlpVar=f['gpc_object']['poly_idx'][()]
            assert hlpVar.shape == (8,3)
            assert np.all(hlpVar == np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                                            [0, 2, 0], [0, 3, 0], [0, 1, 1], [0, 0, 2]]))
            
            hlpVar = f['mesh_roi/data_matrices/E_samples'][()]
            assert hlpVar.shape[0] <= 16
            assert hlpVar.shape[1:] == (14435, 3)
            
            hlpVar = f['mesh_roi/elmdata/'].keys()
            assert len(hlpVar) == 32
            for k in ['E_mean', 'E_sensitivity_3', 'E_sensitivity_4', 'E_sensitivity_5', 
                    'E_sobol_3', 'E_sobol_4', 'E_sobol_5', 'E_std', 'J_mean',
                    'J_sensitivity_3', 'J_sensitivity_4', 'J_sensitivity_5', 'J_sobol_3',
                    'J_sobol_4', 'J_sobol_5', 'J_std', 'magnE_mean', 'magnE_sensitivity_3', 
                    'magnE_sensitivity_4', 'magnE_sensitivity_5', 'magnE_sobol_3', 'magnE_sobol_4',
                    'magnE_sobol_5', 'magnE_std', 'magnJ_mean', 'magnJ_sensitivity_3', 
                    'magnJ_sensitivity_4', 'magnJ_sensitivity_5', 'magnJ_sobol_3', 'magnJ_sobol_4', 
                    'magnJ_sobol_5', 'magnJ_std']:
                assert k in hlpVar
            
            hlpVar = f['mesh_roi/elmdata/E_mean'][()]
            assert hlpVar.shape == (14435, 3)
            assert np.isclose(np.max(hlpVar),2.5408,rtol=5e-03, atol=5e-03)
            assert np.isclose(np.min(hlpVar),-17.595,rtol=5e-03, atol=5e-03)
            
            hlpVar = f['mesh_roi/elmdata/E_std'][()]
            assert hlpVar.shape == (14435, 3)
            assert np.isclose(np.max(hlpVar),2.081,rtol=5e-03, atol=5e-03)
            
            hlpVar = f['mesh_roi/elmdata/E_sobol_3'][()]
            assert hlpVar.shape == (14435, 3)
            assert np.isclose(np.max(hlpVar),0.02125,rtol=5e-03, atol=1e-04)
            assert np.isclose(np.mean(hlpVar),0.0002517,rtol=5e-03, atol=1e-07)
            
            hlpVar = f['mesh_roi/elmdata/E_sensitivity_4'][()]
            assert hlpVar.shape == (14435, 3)
            assert np.isclose(np.max(hlpVar),4.9689,rtol=5e-03, atol=5e-03)
            assert np.isclose(np.min(hlpVar),-1.6340,rtol=5e-03, atol=5e-03)
            
            hlpVar = f['mesh_roi/elmdata/magnJ_mean'][()]
            assert hlpVar.shape == (14435, )
            assert np.isclose(np.max(hlpVar),0.1110,rtol=5e-03, atol=5e-04)
            
            hlpVar = f['mesh_roi/elmdata/magnJ_std'][()]
            assert hlpVar.shape == (14435, )
            assert np.isclose(np.max(hlpVar),0.0264,rtol=5e-03, atol=5e-04)
            
            hlpVar = f['mesh_roi/elmdata/magnJ_sobol_5'][()]
            assert hlpVar.shape == (14435, )
            assert np.isclose(np.max(hlpVar), 3.2435e-05,rtol=5e-03, atol=1e-08)
            assert np.isclose(np.mean(hlpVar),1.0668e-06,rtol=5e-03, atol=1e-09)
            
            hlpVar = f['mesh_roi/elmdata/magnJ_sensitivity_3'][()]
            assert hlpVar.shape == (14435, )
            assert np.isclose(np.max(hlpVar),0.003164,rtol=5e-03, atol=5e-05)
            assert np.isclose(np.min(hlpVar),-0.00015512,rtol=5e-03, atol=5e-07)
            
            hlpVar = f['mesh_roi/elmdata/E_mean'][()]
            hlpVar2 = f['mesh_roi/elmdata/magnE_mean'][()]
            assert hlpVar.shape == (14435, 3)
            assert hlpVar2.shape == (14435,)
            assert np.allclose(np.sqrt(np.sum(hlpVar**2,1)), hlpVar2, rtol=5e-03, atol=5e-03)
            
            hlpVar = f['mesh_roi/elmdata/J_mean'][()]
            hlpVar2 = f['mesh_roi/elmdata/magnJ_mean'][()]
            assert hlpVar.shape == (14435, 3)
            assert hlpVar2.shape == (14435,)
            assert np.allclose(np.sqrt(np.sum(hlpVar**2,1)), hlpVar2, rtol=5e-03, atol=5e-03)
                    
        if os.path.exists(fn_out.name):
                fn_out.cleanup()
                
                
    def test_regression_test_gpc_custom_QoI(self, sphere3):     
        
        # define custom QoI function (J^2 computed from v and cond)
        def _calc_J_sq(v, parameters, tdcslist, identifiers):
            tdcslist = deepcopy(tdcslist)
            for i, iden in enumerate(identifiers):
                tdcslist.cond[iden-1].value = parameters[f"cond_{iden}"]
            
            cond = tdcslist.cond2elmdata(mesh=v.mesh, logger_level=10)
            m = fem.calc_fields(v, 'J', cond=cond)
            
            return m.field['J'].value**2

        tdcs = TDCSLIST()
        tdcs.currents = [0.001, -0.001]
        tdcs.mesh = sphere3
        
        electrode = tdcs.add_electrode()
        electrode.channelnr = 1
        electrode.centre = [95, 0, 0]
        electrode.shape = "ellipse"
        electrode.dimensions = [20, 20]
        electrode.thickness = 4
        
        electrode = tdcs.add_electrode()
        electrode.channelnr = 2
        electrode.centre = [-95, 0, 0]
        electrode.shape = "ellipse"
        electrode.dimensions = [20, 20]
        electrode.thickness = 4
        
        # Set-up the uncertain conductivities
        # intracranial
        tdcs.cond[2].distribution_type = "beta"
        tdcs.cond[2].distribution_parameters = [3, 3, 0.2, 0.4]
        # bone
        tdcs.cond[3].distribution_type = "beta"
        tdcs.cond[3].distribution_parameters = [3, 3, 0.001, 0.012]
                
        fn_out = tempfile.TemporaryDirectory()
        
        tdcs._prepare()
        mesh_w_elect, elect_surface_tags = tdcs._place_electrodes()
        parameters, random_vars = simnibs_gpc.prep_gpc(tdcs)
        J_sq = functools.partial(_calc_J_sq, tdcslist=tdcs, identifiers=random_vars) # J_sq has only 2 variables: v, cond_values
        
        sampler = simnibs_gpc.TDCSgPCSampler(mesh_w_elect,
                                            tdcs,
                                            os.path.join(fn_out.name,'tst.hdf5'),
                                            elect_surface_tags,
                                            tdcs.currents,
                                            roi=[3, 4]
                                            )
        sampler.create_hdf5()
        sampler.qoi_function = {'J_sq': J_sq, 'E': sampler._calc_E}
        
        algorithm = simnibs_gpc.setup_gpc_algorithm(sampler=sampler,
                                                    parameters=parameters,
                                                    data_poly_ratio=2,
                                                    eps=1e-3, # newer version's stopping criteria is seemingly k-fold CV 
                                                    n_cpus=1,
                                                    min_iter=2,
                                                    regularization_factors=np.array([0]),
                                                    order_end=10,
                                                    interaction_order=len(random_vars)
                                                    )
        algorithm.options['error_type'] = 'loocv'
        gpc_session, _ , _ = algorithm.run()
            
        gpc_reg = simnibs_gpc.gPC_regression(problem=gpc_session.problem, regularization_factors=[0],
                                             multi_indices=gpc_session.basis.multi_indices,coords_norm=gpc_session.grid.coords_norm,
                                             sim_type='TCS', data_file=os.path.join(fn_out.name,'tst.hdf5'), n_cpu=1)
        gpc_reg.save_hdf5()
        gpc_reg.postprocessing('J')
        
        assert len(gpc_session.relative_error_loocv)
        assert gpc_session.relative_error_loocv[-1] < 1e-3
        assert gpc_session.grid.coords.shape[0] < 23
        
        assert gpc_reg.multi_indices.shape == (11,2)
        assert np.all(gpc_reg.multi_indices == np.array([[0, 0], [1, 0], [0, 1], [0, 2], [0, 3],
                                                         [2, 0], [1, 1], [1, 2], [0, 4], [3, 0], [2, 1]]))
        
        with h5py.File(os.path.join(fn_out.name,'tst.hdf5'), "r") as f:
            assert 'J_sq_samples' in  f['mesh_roi']['data_matrices']
            data = f['mesh_roi']['data_matrices']['J_sq_samples'][()]
            assert data.shape[0] <= 23
            assert data.shape[1:] == (14435, 3)
             
            data_dims = data.shape[1:]
            if data.ndim == 3:
                data = data.reshape(data.shape[0], -1)
            
            coeffs, error = gpc_reg.expand(data)
            assert error < 0.00034
            
            # mean
            mean = gpc_reg.get_mean(coeffs=coeffs).reshape(data_dims)
            assert 'J_mean' in  f['mesh_roi']['elmdata']
            assert f['mesh_roi']['elmdata']['J_mean'][()].shape == (14435, 3)
            assert np.allclose(mean, f['mesh_roi']['elmdata']['J_mean'][()]**2, rtol=5e-03, atol=5e-03)
        
            std = gpc_reg.get_std(coeffs=coeffs)
            assert std.shape == (43305,)
            assert np.isclose(np.max(std),6.0182e-3,rtol=1e-05, atol=1e-05)
                        
        if os.path.exists(fn_out.name):
                fn_out.cleanup()
