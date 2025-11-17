import os
import numpy as np
import pytest
import scipy
import random

from pathlib import Path
from copy import deepcopy

from simnibs.optimization.tes_flex_optimization.tes_flex_optimization import (
    TesFlexOptimization,
)
from simnibs.utils.matlab_read import dict_from_matlab
from simnibs.mesh_tools.mesh_io import read_msh
from simnibs.mesh_tools import surface, mesh_io, gmsh_view
from simnibs.optimization.tes_flex_optimization.tes_flex_optimization import (
    valid_skin_region,
    write_visualization,
    make_summary_text,
)
from simnibs.optimization.tes_flex_optimization.ellipsoid import Ellipsoid
from simnibs.utils.file_finder import Templates
from simnibs.utils.region_of_interest import RegionOfInterest
from simnibs.optimization.tes_flex_optimization.electrode_layout import (
    ElectrodeArrayPair,
)


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for all tests"""
    random.seed(42)
    np.random.seed(42)


class TestToFromDict:
    def test_write_read_mat(self, tmp_path: Path):
        opt = TesFlexOptimization()
        opt.subpath = "m2m_ernie"
        opt.output_folder = "tes_optimze_4x1tes_focality"
        opt.seed = 42  # Add seed for reproducibility

        """ Set up goal function """
        opt.goal = "focality"
        opt.threshold = [0.1, 0.2]
        opt.e_postproc = "magn"

        electrode = opt.add_electrode_layout("CircularArray")
        electrode.radius_inner = 10
        electrode.radius_outer = 10
        electrode.distance_bounds = [25, 100]
        electrode.n_outer = 4
        electrode.dirichlet_correction = False
        electrode.dirichlet_correction_detailed = False
        electrode.current = [0.002, -0.002 / 4, -0.002 / 4, -0.002 / 4, -0.002 / 4]

        roi = opt.add_roi()
        roi.method = "surface"
        roi.surface_type = "central"
        roi.roi_sphere_center_space = "subject"
        roi.roi_sphere_center = [-41.0, -13.0, 66.0]
        roi.roi_sphere_radius = 20

        roi = opt.add_roi()
        roi.method = "surface"
        roi.surface_type = "central"
        roi.roi_sphere_center_space = "subject"
        roi.roi_sphere_center = [-41.0, -13.0, 66.0]
        roi.roi_sphere_radius = 25
        roi.roi_sphere_operator = "difference"

        mat_path = os.path.join(tmp_path, "test.mat")

        scipy.io.savemat(mat_path, opt.to_dict())
        tes_flex_opt_loaded = TesFlexOptimization(
            dict_from_matlab(scipy.io.loadmat(mat_path))
        )

        np.testing.assert_equal(opt.to_dict(), tes_flex_opt_loaded.to_dict())

        dict_before = opt.__dict__.copy()
        dict_after = tes_flex_opt_loaded.__dict__.copy()
        del dict_before["_ff_templates"]
        del dict_before["roi"]
        del dict_before["_ellipsoid"]
        del dict_before["electrode"]

        del dict_after["_ff_templates"]
        del dict_after["roi"]
        del dict_after["_ellipsoid"]
        del dict_after["electrode"]

        np.testing.assert_equal(dict_before, dict_after)
        np.testing.assert_equal(
            opt.roi[0].__dict__, tes_flex_opt_loaded.roi[0].__dict__
        )
        np.testing.assert_equal(
            opt.roi[1].__dict__, tes_flex_opt_loaded.roi[1].__dict__
        )
        np.testing.assert_equal(
            opt.electrode[0].__dict__, tes_flex_opt_loaded.electrode[0].__dict__
        )

    @pytest.mark.slow
    def test_write_read_mat_after_prepare(self, tmp_path: Path, example_dataset):
        opt = TesFlexOptimization()
        # path of m2m folder containing the headmodel
        opt.subpath = os.path.join(example_dataset, "m2m_ernie")
        opt.seed = 42  # seed optimizer for reproducibility

        # output folder
        opt.output_folder = os.path.join(tmp_path, "tes_optimize_ti_intensity")

        # type of goal function
        opt.goal = "mean"

        # postprocessing function of e-fields
        # "max_TI": maximize envelope of e-field magnitude
        # "dir_TI_normal": maximize envelope of e-field normal component
        # "dir_TI_tangential": maximize envelope of e-field tangential component
        opt.e_postproc = "max_TI"

        # define first pair of electrodes
        electrode = opt.add_electrode_layout("ElectrodeArrayPair")
        electrode.center = [
            [0, 0],  # electrode center in reference electrode space (x-y plane)
            [0, 20],
        ]
        electrode.radius = [10, 10]  # radius of electrodes
        electrode.dirichlet_correction_detailed = (
            False  # node wise dirichlet correction
        )
        electrode.current = [0.002, 0.002, -0.002, -0.002]  # electrode currents

        # define second pair of electrodes
        electrode = opt.add_electrode_layout("ElectrodeArrayPair")
        electrode.center = [
            [0, 0],  # electrode center in reference electrode space (x-y plane)
            [0, 20],
        ]
        electrode.radius = [10, 10]  # radius of electrodes
        electrode.dirichlet_correction_detailed = (
            False  # node wise dirichlet correction
        )
        electrode.current = [0.002, 0.002, -0.002, -0.002]  # electrode currents

        # define ROI
        roi = opt.add_roi()
        roi.method = "surface"
        roi.surface_type = "central"

        # center of spherical ROI in subject space (in mm)
        roi.roi_sphere_center_space = "subject"
        roi.roi_sphere_center = [-41.0, -13.0, 66.0]

        # radius of spherical ROI (in mm)
        roi.roi_sphere_radius = 20

        # prepare optimization
        opt._prepare()

        # save opt instance as .mat structure
        mat_path = os.path.join(tmp_path, "test.mat")
        scipy.io.savemat(mat_path, opt.to_dict())

        # load .mat structure and initialize new opt instance
        tes_flex_opt_loaded = TesFlexOptimization(
            dict_from_matlab(scipy.io.loadmat(mat_path))
        )

        # prepare loaded opt instance
        tes_flex_opt_loaded._prepare()

        # check if both opt instances are equal
        np.testing.assert_equal(opt.to_dict(), tes_flex_opt_loaded.to_dict())

    @pytest.mark.slow
    def test_run_opt_from_mat(self, tmp_path: Path, example_dataset):
        opt = TesFlexOptimization()
        opt.seed = 42  # seed optimizer for reproducibility
        opt.open_in_gmsh = False
        opt.detailed_results = True

        # path of m2m folder containing the headmodel
        opt.subpath = os.path.join(example_dataset, "m2m_ernie")

        # output folder
        opt.output_folder = os.path.join(tmp_path, "tes_optimize_tes_intensity_org")

        # type of goal function
        opt.goal = "mean"

        # postprocessing function of e-fields
        opt.e_postproc = "magn"

        # define first pair of electrodes
        electrode = opt.add_electrode_layout("ElectrodeArrayPair")
        electrode.center = [
            [0, 0]
        ]  # electrode center in reference electrode space (x-y plane)
        electrode.radius = [10]  # radius of electrodes
        electrode.dirichlet_correction_detailed = (
            False  # node wise dirichlet correction
        )
        electrode.current = [0.002, -0.002]  # electrode currents

        # define ROI
        roi = opt.add_roi()
        roi.method = "surface"
        roi.surface_type = "central"
        roi.roi_sphere_center_space = "subject"
        roi.roi_sphere_center = [-41.0, -13.0, 66.0]
        roi.roi_sphere_radius = 20

        # prepare optimization
        opt._prepare()

        # save opt instance as .mat structure
        mat_path = os.path.join(tmp_path, "opt.mat")
        scipy.io.savemat(mat_path, opt.to_dict())

        # load .mat structure and initialize new opt instance
        tes_flex_opt_loaded = TesFlexOptimization(
            dict_from_matlab(scipy.io.loadmat(mat_path))
        )
        tes_flex_opt_loaded.output_folder = os.path.join(
            tmp_path, "tes_optimize_tes_intensity_mat"
        )

        # prepare loaded opt instance
        tes_flex_opt_loaded._prepare()

        # run optimization (original)
        opt.run()

        # run optimization (.mat)
        tes_flex_opt_loaded.run()

        # compare results
        msh_org = read_msh(
            os.path.join(opt.output_folder, "ernie_tes_flex_opt_surface_mesh.msh")
        )
        msh_mat = read_msh(
            os.path.join(
                tes_flex_opt_loaded.output_folder, "ernie_tes_flex_opt_surface_mesh.msh"
            )
        )

        np.testing.assert_allclose(
            msh_org.field["magnE"].value, msh_mat.field["magnE"].value
        )

    @pytest.mark.slow
    def test_compute_goal(self, tmp_path: Path, example_dataset):
        opt = TesFlexOptimization()
        opt.seed = 42  # Add seed for reproducibility

        # path of m2m folder containing the headmodel
        opt.subpath = os.path.join(example_dataset, "m2m_ernie")

        # output folder
        opt.output_folder = os.path.join(tmp_path, "tes_optimize_compute_goal")

        # postprocessing function of e-fields
        opt.e_postproc = "magn"

        # type of goal function
        opt.goal = "mean"

        # define first pair of electrodes
        electrode = opt.add_electrode_layout("ElectrodeArrayPair")
        electrode.center = [
            [0, 0]
        ]  # electrode center in reference electrode space (x-y plane)
        electrode.radius = [10]  # radius of electrodes
        electrode.dirichlet_correction_detailed = (
            False  # node wise dirichlet correction
        )
        electrode.current = [0.002, -0.002]  # electrode currents

        # define ROI
        roi = opt.add_roi()
        roi.method = "surface"
        roi.surface_type = "central"
        roi.roi_sphere_center_space = "subject"
        roi.roi_sphere_center = [-41.0, -13.0, 66.0]
        roi.roi_sphere_radius = 20

        # prepare optimization
        opt._prepare()

        # mean
        opt.goal = ["mean"]
        e_test = [[np.array([1, 2, 3])]]
        goal_mean = opt.compute_goal(e_test)
        np.testing.assert_equal(goal_mean, -2)

        # neg_mean
        opt.goal = ["neg_mean"]
        e_test = [[np.array([1, 2, 3])]]
        goal_neg_mean = opt.compute_goal(e_test)
        np.testing.assert_equal(goal_neg_mean, 2)

        # mean_abs
        opt.goal = ["mean_abs"]
        e_test = [[np.array([-1, 2, 3])]]
        goal_mean_abs = opt.compute_goal(e_test)
        np.testing.assert_equal(goal_mean_abs, -2)

        # max
        opt.goal = ["max"]
        e_test = [[np.array([1, 2, 3])]]
        goal_max = opt.compute_goal(e_test)
        np.testing.assert_allclose(goal_max, -3, rtol=1e-2)

        # neg_max
        opt.goal = ["neg_max"]
        e_test = [[np.array([1, 2, 3])]]
        goal_neg_max = opt.compute_goal(e_test)
        np.testing.assert_allclose(goal_neg_max, 3, rtol=1e-2)

        # max_abs
        opt.goal = ["max_abs"]
        e_test = [[np.array([1, 2, -3])]]
        goal_max_abs = opt.compute_goal(e_test)
        np.testing.assert_allclose(goal_max_abs, -3, rtol=1e-2)

        # focality (one threshold)
        opt.goal = ["focality"]
        opt.threshold = [2]
        e_test = [[np.array([3, 4, 5]), np.array([0, 1, 2, 3])]]
        goal_focality_threshold_1 = opt.compute_goal(e_test)
        np.testing.assert_equal(np.round(goal_focality_threshold_1), -91)

        # focality (two thresholds)
        opt.goal = ["focality"]
        opt.threshold = [2, 5]
        e_test = [[np.array([3, 4, 5]), np.array([0, 1, 2, 3])]]
        goal_focality_threshold_2 = opt.compute_goal(e_test)
        np.testing.assert_equal(np.round(goal_focality_threshold_2), -58)

        # focality_inv (one threshold)
        opt.goal = ["focality_inv"]
        opt.threshold = [2]
        e_test = [[np.array([3, 4, 5]), np.array([0, 1, 2, 3])]]
        goal_focality_inv_threshold_1 = opt.compute_goal(e_test)
        np.testing.assert_equal(np.round(goal_focality_inv_threshold_1), -112)

        # focality_inv (two thresholds)
        opt.goal = ["focality_inv"]
        opt.threshold = [2, 5]
        e_test = [[np.array([3, 4, 5]), np.array([0, 1, 2, 3])]]
        goal_focality_inv_threshold_2 = opt.compute_goal(e_test)
        np.testing.assert_equal(np.round(goal_focality_inv_threshold_2), -60)

    @pytest.mark.slow
    def test_valid_skin_region(self, tmp_path: Path, example_dataset):
        ellipsoid = Ellipsoid()
        fn_electrode_mask = Templates().mni_volume_upper_head_mask

        mesh = read_msh(os.path.join(example_dataset, "m2m_ernie", "ernie.msh"))

        # relabel internal air
        mesh_relabel = mesh.relabel_internal_air()

        # make final skin surface including some additional distance
        skin_surface = surface.Surface(mesh=mesh_relabel, labels=1005)
        skin_surface = valid_skin_region(
            skin_surface=skin_surface,
            fn_electrode_mask=fn_electrode_mask,
            mesh=mesh_relabel,
            additional_distance=0,
        )

        np.testing.assert_equal(skin_surface.nodes.shape[0], 5427)
        np.testing.assert_allclose(
            skin_surface.nodes[0, :], [9.80649432, 104.20439836, 58.96501272], rtol=1e-6
        )

        # fit optimal ellipsoid to valid skin points
        ellipsoid.fit(points=skin_surface.nodes)

        np.testing.assert_allclose(
            ellipsoid.radii, [113.60465987, 105.76692858, 86.19796459], rtol=1e-6
        )
        np.testing.assert_allclose(
            ellipsoid.center, [1.47022674, 16.43981549, -4.79728615], rtol=1e-6
        )
        np.testing.assert_allclose(
            ellipsoid.rotmat,
            np.array(
                [
                    [-0.02356704, -0.06572103, -0.99755969],
                    [-0.77416654, -0.63015041, 0.05980489],
                    [0.63254309, -0.77368676, 0.03602824],
                ]
            ),
            rtol=1e-6,
        )


class Test_Write_Visualization:
    def test_volume_rois(self, sphere3_msh: mesh_io.Msh, tmp_path):
        input_mesh = deepcopy(sphere3_msh)
        # Create Roi 1
        roi_1 = RegionOfInterest()
        roi_1.load_mesh(input_mesh)
        roi_1.tissues = [3]
        roi_1.method = "volume"
        roi_1.roi_sphere_center_space = "subject"
        roi_1.roi_sphere_center = [0.0, 0.0, 80.0]
        roi_1.roi_sphere_radius = 20
        roi_1._prepare()

        roi_2 = RegionOfInterest()
        roi_2.load_mesh(input_mesh)
        roi_2.tissues = [3]
        roi_2.method = "volume"
        roi_2.roi_sphere_center_space = "subject"
        roi_2.roi_sphere_center = [0.0, 0.0, -80.0]
        roi_2.roi_sphere_radius = 20
        roi_2._prepare()

        # Create result mesh 1
        result_mesh_1_file_path = os.path.join(tmp_path, "test1_TMS_scalar.msh")
        result_mesh_1 = deepcopy(sphere3_msh)
        opt_1 = gmsh_view.Visualization(result_mesh_1)

        opt_1.add_view(ColormapNumber=1)
        result_mesh_1.add_element_field(
            np.tile(np.arange(result_mesh_1.elm.nr), 3).reshape(
                (result_mesh_1.elm.nr, 3)
            ),
            "E",
        )
        opt_1.add_view(ColormapNumber=2)
        result_mesh_1.add_element_field(np.arange(result_mesh_1.elm.nr) * 2, "magnE")
        opt_1.add_view(ColormapNumber=3)
        geo_file_path = os.path.join(tmp_path, "test1_TMS_coil_pos.geo")
        opt_1.add_merge(geo_file_path)
        mesh_io.write_geo_lines(
            [[1, 3, 4], [8, 9, 10]],
            [[5, 6, 7], [11, 12, 13]],
            geo_file_path,
            name="lines",
        )
        opt_1.add_view(ColormapNumber=4)
        result_mesh_1.write(result_mesh_1_file_path)
        opt_1.write_opt(result_mesh_1_file_path)

        # Create result mesh 2
        result_mesh_2_file_path = os.path.join(tmp_path, "test2_TMS_scalar.msh")
        result_mesh_2 = deepcopy(sphere3_msh)
        opt_2 = gmsh_view.Visualization(result_mesh_2)

        opt_2.add_view(ColormapNumber=5)
        result_mesh_2.add_element_field(
            np.tile(np.arange(result_mesh_2.elm.nr) * 3, 3).reshape(
                (result_mesh_2.elm.nr, 3)
            ),
            "E",
        )
        opt_2.add_view(ColormapNumber=6)
        result_mesh_2.add_element_field(np.arange(result_mesh_2.elm.nr) * 4, "magnE")
        opt_2.add_view(ColormapNumber=7)
        geo_file_path = os.path.join(tmp_path, "test2_TMS_coil_pos.geo")
        opt_2.add_merge(geo_file_path)
        mesh_io.write_geo_lines(
            [[1, 3, 4], [8, 9, 10]],
            [[5, 6, 7], [11, 12, 13]],
            geo_file_path,
            name="lines",
        )
        opt_2.add_view(ColormapNumber=8)
        result_mesh_2.write(result_mesh_2_file_path)
        opt_2.write_opt(result_mesh_2_file_path)

        # write visualization
        roi_list = [roi_1, roi_2]
        results_list = [result_mesh_1_file_path, result_mesh_2_file_path]
        base_file_name = "volmask"
        e_postproc = ["max_TI", "max_TI", "dir_TI", "tangential", "normal", "magn"]
        goal_list = ["mean"]

        write_visualization(
            tmp_path, base_file_name, roi_list, results_list, e_postproc, goal_list
        )

        fn_vis_head = os.path.join(tmp_path, "volmask_head_mesh.msh")
        fn_vis_surf = os.path.join(tmp_path, "volmask_surface_mesh.msh")
        assert os.path.exists(fn_vis_head)
        assert not os.path.exists(fn_vis_surf)

        if os.path.exists(fn_vis_head):
            m = mesh_io.read_msh(fn_vis_head)
            field_names = list(m.field.keys())
            field_names_ref = [
                "ROI_0",
                "ROI_1",
                "channel_0__magnE",
                "channel_1__magnE",
                "average__magnE",
                "max_TI",
            ]

            assert "E" not in field_names
            assert "magnE" not in field_names
            assert len(set(field_names_ref).difference(field_names)) == 0

    def test_surface_rois(self, sphere3_msh: mesh_io.Msh, tmp_path):
        input_mesh = deepcopy(sphere3_msh)
        surf_mesh = input_mesh.crop_mesh(tags=1003)
        surf_mesh_file_path = os.path.join(tmp_path, "surf_roi.msh")
        mesh_io.write_msh(surf_mesh, surf_mesh_file_path)

        # Create Roi 1
        roi_1 = RegionOfInterest()
        roi_1.method = "surface"
        roi_1.surface_type = "custom"
        roi_1.surface_path = surf_mesh_file_path
        roi_1._prepare()

        # Create Roi 2
        roi_2 = RegionOfInterest()
        roi_2.method = "surface"
        roi_2.surface_type = "custom"
        roi_2.surface_path = surf_mesh_file_path
        roi_2._prepare()

        # Create result mesh 1
        result_mesh_1_file_path = os.path.join(tmp_path, "test1_TMS_scalar.msh")
        result_mesh_1 = deepcopy(sphere3_msh)
        opt_1 = gmsh_view.Visualization(result_mesh_1)

        opt_1.add_view(ColormapNumber=1)
        result_mesh_1.add_element_field(
            np.tile(np.arange(result_mesh_1.elm.nr), 3).reshape(
                (result_mesh_1.elm.nr, 3)
            ),
            "E",
        )
        opt_1.add_view(ColormapNumber=2)
        result_mesh_1.add_element_field(np.arange(result_mesh_1.elm.nr) * 2, "magnE")
        opt_1.add_view(ColormapNumber=3)
        geo_file_path = os.path.join(tmp_path, "test1_TMS_coil_pos.geo")
        opt_1.add_merge(geo_file_path)
        mesh_io.write_geo_lines(
            [[1, 3, 4], [8, 9, 10]],
            [[5, 6, 7], [11, 12, 13]],
            geo_file_path,
            name="lines",
        )
        opt_1.add_view(ColormapNumber=4)
        result_mesh_1.write(result_mesh_1_file_path)
        opt_1.write_opt(result_mesh_1_file_path)

        # Create result mesh 2
        result_mesh_2_file_path = os.path.join(tmp_path, "test2_TMS_scalar.msh")
        result_mesh_2 = deepcopy(sphere3_msh)
        opt_2 = gmsh_view.Visualization(result_mesh_2)

        opt_2.add_view(ColormapNumber=5)
        result_mesh_2.add_element_field(
            np.tile(np.arange(result_mesh_2.elm.nr) * 3, 3).reshape(
                (result_mesh_2.elm.nr, 3)
            ),
            "E",
        )
        opt_2.add_view(ColormapNumber=6)
        result_mesh_2.add_element_field(np.arange(result_mesh_2.elm.nr) * 4, "magnE")
        opt_2.add_view(ColormapNumber=7)
        geo_file_path = os.path.join(tmp_path, "test2_TMS_coil_pos.geo")
        opt_2.add_merge(geo_file_path)
        mesh_io.write_geo_lines(
            [[1, 3, 4], [8, 9, 10]],
            [[5, 6, 7], [11, 12, 13]],
            geo_file_path,
            name="lines",
        )
        opt_2.add_view(ColormapNumber=8)
        result_mesh_2.write(result_mesh_2_file_path)
        opt_2.write_opt(result_mesh_2_file_path)

        # write visualization
        roi_list = [roi_1, roi_2]
        results_list = [result_mesh_1_file_path, result_mesh_2_file_path]
        base_file_name = "surfmask"
        e_postproc = ["max_TI", "max_TI", "dir_TI", "tangential", "normal", "magn"]
        goal_list = ["mean"]

        write_visualization(
            tmp_path, base_file_name, roi_list, results_list, e_postproc, goal_list
        )

        fn_vis_head = os.path.join(tmp_path, "surfmask_head_mesh.msh")
        fn_vis_surf = os.path.join(tmp_path, "surfmask_surface_mesh.msh")
        assert not os.path.exists(fn_vis_head)
        assert os.path.exists(fn_vis_surf)

        if os.path.exists(fn_vis_surf):
            m = mesh_io.read_msh(fn_vis_surf)
            field_names = list(m.field.keys())
            field_names_ref = [
                "ROI_0",
                "ROI_1",
                "channel_0__magnE",
                "channel_1__magnE",
                "average__magnE",
                "channel_0__normal",
                "channel_1__normal",
                "average__normal",
                "channel_0__tangential",
                "channel_1__tangential",
                "average__tangential",
                "max_TI",
                "dir_TI",
            ]

            assert "E" not in field_names
            assert "magnE" not in field_names
            assert len(set(field_names_ref).difference(field_names)) == 0

    def test_surface_and_volume_rois(self, sphere3_msh: mesh_io.Msh, tmp_path):
        input_mesh = deepcopy(sphere3_msh)
        surf_mesh = input_mesh.crop_mesh(tags=1003)
        surf_mesh_file_path = os.path.join(tmp_path, "surf_roi.msh")
        mesh_io.write_msh(surf_mesh, surf_mesh_file_path)

        # Create Roi 1
        roi_1 = RegionOfInterest()
        roi_1.method = "surface"
        roi_1.surface_type = "custom"
        roi_1.surface_path = surf_mesh_file_path
        roi_1._prepare()

        # Create Roi 2
        roi_2 = RegionOfInterest()
        roi_2.load_mesh(input_mesh)
        roi_2.tissues = [3]
        roi_2.method = "volume"
        roi_2.roi_sphere_center_space = "subject"
        roi_2.roi_sphere_center = [0.0, 0.0, -80.0]
        roi_2.roi_sphere_radius = 20
        roi_2._prepare()

        # Create result mesh 1
        result_mesh_1_file_path = os.path.join(tmp_path, "test1_TMS_scalar.msh")
        result_mesh_1 = deepcopy(sphere3_msh)
        opt_1 = gmsh_view.Visualization(result_mesh_1)

        opt_1.add_view(ColormapNumber=1)
        result_mesh_1.add_element_field(
            np.tile(np.arange(result_mesh_1.elm.nr), 3).reshape(
                (result_mesh_1.elm.nr, 3)
            ),
            "E",
        )
        opt_1.add_view(ColormapNumber=2)
        result_mesh_1.add_element_field(np.arange(result_mesh_1.elm.nr) * 2, "magnE")
        opt_1.add_view(ColormapNumber=3)
        geo_file_path = os.path.join(tmp_path, "test1_TMS_coil_pos.geo")
        opt_1.add_merge(geo_file_path)
        mesh_io.write_geo_lines(
            [[1, 3, 4], [8, 9, 10]],
            [[5, 6, 7], [11, 12, 13]],
            geo_file_path,
            name="lines",
        )
        opt_1.add_view(ColormapNumber=4)
        result_mesh_1.write(result_mesh_1_file_path)
        opt_1.write_opt(result_mesh_1_file_path)

        # Create result mesh 2
        result_mesh_2_file_path = os.path.join(tmp_path, "test2_TMS_scalar.msh")
        result_mesh_2 = deepcopy(sphere3_msh)
        opt_2 = gmsh_view.Visualization(result_mesh_2)

        opt_2.add_view(ColormapNumber=5)
        result_mesh_2.add_element_field(
            np.tile(np.arange(result_mesh_2.elm.nr) * 3, 3).reshape(
                (result_mesh_2.elm.nr, 3)
            ),
            "E",
        )
        opt_2.add_view(ColormapNumber=6)
        result_mesh_2.add_element_field(np.arange(result_mesh_2.elm.nr) * 4, "magnE")
        opt_2.add_view(ColormapNumber=7)
        geo_file_path = os.path.join(tmp_path, "test2_TMS_coil_pos.geo")
        opt_2.add_merge(geo_file_path)
        mesh_io.write_geo_lines(
            [[1, 3, 4], [8, 9, 10]],
            [[5, 6, 7], [11, 12, 13]],
            geo_file_path,
            name="lines",
        )
        opt_2.add_view(ColormapNumber=8)
        result_mesh_2.write(result_mesh_2_file_path)
        opt_2.write_opt(result_mesh_2_file_path)

        # write visualization
        roi_list = [roi_1, roi_2]
        results_list = [result_mesh_1_file_path, result_mesh_2_file_path]
        base_file_name = "bothmasks"
        e_postproc = ["max_TI", "max_TI", "dir_TI", "tangential", "normal", "magn"]
        goal_list = ["mean"]

        write_visualization(
            tmp_path, base_file_name, roi_list, results_list, e_postproc, goal_list
        )

        fn_vis_head = os.path.join(tmp_path, "bothmasks_head_mesh.msh")
        fn_vis_surf = os.path.join(tmp_path, "bothmasks_surface_mesh.msh")
        assert os.path.exists(fn_vis_head)
        assert os.path.exists(fn_vis_surf)

        if os.path.exists(fn_vis_head):
            m = mesh_io.read_msh(fn_vis_head)
            field_names = list(m.field.keys())
            field_names_ref = [
                "ROI_1",
                "channel_0__magnE",
                "channel_1__magnE",
                "average__magnE",
                "max_TI",
            ]

            assert "E" not in field_names
            assert "magnE" not in field_names
            assert len(set(field_names_ref).difference(field_names)) == 0

        if os.path.exists(fn_vis_surf):
            m = mesh_io.read_msh(fn_vis_surf)
            field_names = list(m.field.keys())
            field_names_ref = [
                "ROI_0",
                "channel_0__magnE",
                "channel_1__magnE",
                "average__magnE",
                "channel_0__normal",
                "channel_1__normal",
                "average__normal",
                "channel_0__tangential",
                "channel_1__tangential",
                "average__tangential",
                "max_TI",
                "dir_TI",
            ]

            assert "E" not in field_names
            assert "magnE" not in field_names
            assert len(set(field_names_ref).difference(field_names)) == 0


class Test_Make_Summary_Text:
    def test_m_head(self, sphere3_msh: mesh_io.Msh):
        m_head = sphere3_msh.crop_mesh(elm_type=4)
        m_head.elm.tag1[m_head.elm.tag1 == 4] = 2
        m_head.elm.tag2 = m_head.elm.tag1

        m_surf = None

        m_head.add_element_field(
            np.tile(np.arange(m_head.elm.nr), 3).reshape((m_head.elm.nr, 3)), "E"
        )
        m_head.add_element_field(np.ones(m_head.elm.nr), "max_TO")
        m_head.add_element_field(np.ones(m_head.elm.nr), "magnE")
        m_head.add_element_field(np.ones(m_head.elm.nr), "average__magnE")
        m_head.add_element_field(m_head.elm.tag1 == 2, "ROI")
        m_head.add_element_field(np.ones(m_head.elm.nr), "ROI_1")
        m_head.add_element_field(np.ones(m_head.elm.nr), "non-ROI")
        m_head.add_element_field(np.ones(m_head.elm.nr), " ROI_2")

        summary_txt = make_summary_text(m_surf, m_head)

        assert summary_txt.count("|magnE") == 3
        assert summary_txt.count("|average__magnE") == 3
        assert summary_txt.count("|E") == 0
        assert summary_txt.count("max_TO") == 0

        assert summary_txt.count("|ROI_1 ") == 1
        assert summary_txt.count("|non-ROI ") == 1
        assert summary_txt.count("|ROI ") == 1
        assert summary_txt.count("ROI_2 ") == 0

    def test_m_surf(self, sphere3_msh: mesh_io.Msh, tmp_path):
        m_head = None
        m_surf = sphere3_msh.crop_mesh(elm_type=2)

        m_surf.add_node_field(
            np.tile(np.arange(m_surf.nodes.nr), 3).reshape((m_surf.nodes.nr, 3)), "E"
        )
        m_surf.add_node_field(np.ones(m_surf.nodes.nr), "max_TO")
        m_surf.add_node_field(np.ones(m_surf.nodes.nr), "magnE")
        m_surf.add_node_field(np.ones(m_surf.nodes.nr), "average__magnE")
        m_surf.add_node_field(np.ones(m_surf.nodes.nr), "ROI")
        m_surf.add_node_field(np.ones(m_surf.nodes.nr), "ROI_1")
        m_surf.add_node_field(np.ones(m_surf.nodes.nr), "non-ROI")
        m_surf.add_node_field(np.ones(m_surf.nodes.nr), " ROI_2")

        summary_txt = make_summary_text(m_surf, m_head)

        assert summary_txt.count("|magnE") == 3
        assert summary_txt.count("|average__magnE") == 3
        assert summary_txt.count("|E") == 0
        assert summary_txt.count("max_TO") == 0

        assert summary_txt.count("|ROI_1 ") == 1
        assert summary_txt.count("|non-ROI ") == 1
        assert summary_txt.count("|ROI ") == 1
        assert summary_txt.count("ROI_2 ") == 0

    def test_m_head_and_m_surf(self, sphere3_msh: mesh_io.Msh):
        m_head = sphere3_msh.crop_mesh(elm_type=4)
        m_head.elm.tag1[m_head.elm.tag1 == 4] = 2
        m_head.elm.tag2 = m_head.elm.tag1

        m_surf = sphere3_msh.crop_mesh(elm_type=2)

        m_head.add_element_field(
            np.tile(np.arange(m_head.elm.nr), 3).reshape((m_head.elm.nr, 3)), "E"
        )
        m_head.add_element_field(np.ones(m_head.elm.nr), "max_TO")
        m_head.add_element_field(np.ones(m_head.elm.nr), "magnE")
        m_head.add_element_field(np.ones(m_head.elm.nr), "ROI_1")
        m_surf.add_node_field(np.ones(m_surf.nodes.nr), "average__magnE")
        m_surf.add_node_field(np.ones(m_surf.nodes.nr), "non-ROI")

        summary_txt = make_summary_text(m_surf, m_head)

        assert summary_txt.count("|magnE") == 3
        assert summary_txt.count("|average__magnE") == 3
        assert summary_txt.count("|E") == 0
        assert summary_txt.count("max_TO") == 0
        assert summary_txt.count("|ROI_1 ") == 1
        assert summary_txt.count("|non-ROI ") == 1


class TestElectrodeMapping:
    """Tests for the electrode mapping functionality"""

    @pytest.fixture
    def simple_optimization(self, tmp_path):
        """Create a simple TesFlexOptimization setup for testing"""
        output_dir = tmp_path / "test_output"
        output_dir.mkdir(exist_ok=True)

        opt = TesFlexOptimization()
        opt.fnamehead = str(tmp_path / "mock_ernie.msh")
        opt.output_folder = str(output_dir)
        opt.subpath = str(tmp_path / "mock_m2m_ernie")
        opt.seed = 42
        opt.goal = "mean"

        # Set up electrode array
        electrode = ElectrodeArrayPair()
        electrode.radius = [4]
        electrode.center = [[0, 0]]
        electrode.current = [0.002, -0.002]
        electrode._prepare()
        opt.electrode = [electrode]

        # Set up ROI
        roi = opt.add_roi()
        roi.method = "surface"
        roi.surface_type = "central"
        roi.roi_sphere_center_space = "subject"
        roi.roi_sphere_center = [0, 0, 0]
        roi.roi_sphere_radius = 5

        # Mock optimization results
        opt.electrode_pos_opt = [
            [np.array([1.0, -1.5, 0.0]), np.array([1.2, 1.7, 0.0])]
        ]

        # Set up position matrices
        for i_channel_stim in range(len(opt.electrode)):
            for i_array, electrode_array in enumerate(
                opt.electrode[i_channel_stim]._electrode_arrays
            ):
                for electrode in electrode_array.electrodes:
                    if i_array == 0:
                        electrode.posmat = np.array(
                            [
                                [1.0, 0.0, 0.0, -60.0],
                                [0.0, 1.0, 0.0, 70.0],
                                [0.0, 0.0, 1.0, 35.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    else:
                        electrode.posmat = np.array(
                            [
                                [1.0, 0.0, 0.0, 80.0],
                                [0.0, 1.0, 0.0, 40.0],
                                [0.0, 0.0, 1.0, -5.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )

        # Create mock EEG net file
        eeg_positions_dir = output_dir / "eeg_positions"
        eeg_positions_dir.mkdir(exist_ok=True)
        net_file = eeg_positions_dir / "mock_eeg_net.csv"

        with open(net_file, "w") as f:
            f.write("Electrode,-58.0,75.0,37.0,F5\n")
            f.write("Electrode,79.0,41.0,9.0,FT8\n")
            f.write("Electrode,0.0,0.0,100.0,Cz\n")

        opt.net_electrode_file = str(net_file)
        return opt

    @pytest.mark.slow
    def test_map_to_nearest_net_electrodes(self, simple_optimization):
        """Test mapping electrodes to the nearest positions in an EEG net"""
        opt = simple_optimization
        opt.map_to_net_electrodes = True

        def mock_prepare():
            pass

        opt._prepare = mock_prepare

        mapping_result = opt.map_to_nearest_net_electrodes(opt.net_electrode_file)

        # Verify mapping structure
        assert isinstance(mapping_result, dict)
        assert all(
            key in mapping_result
            for key in [
                "optimized_positions",
                "mapped_positions",
                "mapped_labels",
                "distances",
                "channel_array_indices",
            ]
        )

        # Verify mapping results
        assert len(mapping_result["mapped_labels"]) == 2
        assert mapping_result["mapped_labels"][0] == "F5"
        assert mapping_result["mapped_labels"][1] == "FT8"

        # Verify positions
        np.testing.assert_allclose(
            mapping_result["mapped_positions"][0], [-58.0, 75.0, 37.0], rtol=1e-5
        )
        np.testing.assert_allclose(
            mapping_result["mapped_positions"][1], [79.0, 41.0, 9.0], rtol=1e-5
        )

        # Verify distances
        expected_distance_1 = np.linalg.norm(
            np.array([-60.0, 70.0, 35.0]) - np.array([-58.0, 75.0, 37.0])
        )
        expected_distance_2 = np.linalg.norm(
            np.array([80.0, 40.0, -5.0]) - np.array([79.0, 41.0, 9.0])
        )

        np.testing.assert_allclose(
            mapping_result["distances"][0], expected_distance_1, rtol=1e-5
        )
        np.testing.assert_allclose(
            mapping_result["distances"][1], expected_distance_2, rtol=1e-5
        )
