import pytest
from pathlib import Path
import copy
import numpy as np
import requests
import shutil
import zipfile
import tempfile
import functools

from simnibs.mesh_tools import mesh_io
from simnibs.mesh_tools.mesh_io import Elements, Msh, Nodes
from simnibs.simulation.tms_coil.tms_coil import TmsCoil
from simnibs.simulation.tms_coil.tms_coil_deformation import (
    TmsCoilDeformationRange,
    TmsCoilRotation,
    TmsCoilTranslation,
)
from simnibs.simulation.tms_coil.tms_coil_element import (
    LineSegmentElements,
)
from simnibs.simulation.tms_coil.tms_coil_model import TmsCoilModel
from simnibs.simulation.tms_coil.tms_stimulator import TmsStimulator


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_addoption(parser):
    """
    Adds the --skip-slow command-line option to pytest.
    """
    parser.addoption(
        "--skip-slow",
        action="store_true",
        default=False,
        help="skip tests marked as slow",
    )


def pytest_collection_modifyitems(config, items):
    """
    Modifies the collected test items based on the --skip-slow flag.
    """
    if not config.getoption("--skip-slow"):
        return

    skip_slow = pytest.mark.skip(reason="skipped because --skip-slow was specified")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def test_data_dir():
    """Returns the absolute path to the 'resources' directory."""
    return Path(__file__).parent / "resources"


@pytest.fixture(scope="session")
def examples_dir():
    """Returns the absolute path to the 'examples' directory."""
    return Path(__file__).parent / ".." / "examples"


@pytest.fixture(scope="session")
def _base_sphere3_msh(test_data_dir):
    return mesh_io.read_msh(test_data_dir / "sphere3.msh")


@pytest.fixture(scope="function")
def sphere3_msh(_base_sphere3_msh):
    return copy.deepcopy(_base_sphere3_msh)


@pytest.fixture(scope="module")
def example_dataset():
    url = "https://github.com/simnibs/example-dataset/releases/download/v4.0-lowres/ernie_lowres_V2.zip"

    fn_folder = tempfile.mkdtemp()

    with tempfile.NamedTemporaryFile(suffix=".zip") as tmp_zip:
        with requests.get(url, stream=True) as r:
            r.raw.read = functools.partial(r.raw.read, decode_content=True)
            shutil.copyfileobj(r.raw, tmp_zip)

        tmp_zip.flush()

        with zipfile.ZipFile(tmp_zip.name) as z:
            z.extractall(fn_folder)

    yield fn_folder

    try:
        shutil.rmtree(fn_folder)
    except:
        print(f"Could not remove example dataset folder: {fn_folder}")


@pytest.fixture(scope="session")
def rdm():
    """
    Utility function to calculate the Relative Difference Measure.
    """

    def _rdm(a, b):
        return np.linalg.norm(a / np.linalg.norm(a) - b / np.linalg.norm(b))

    return _rdm


@pytest.fixture(scope="session")
def mag():
    """
    Utility function to calculate the magnitude difference in log space.
    """

    def _mag(a, b):
        return np.abs(np.log(np.linalg.norm(a) / np.linalg.norm(b)))

    return _mag


@pytest.fixture(scope="module")
def small_functional_3_element_coil() -> TmsCoil:
    casings = [
        TmsCoilModel(
            Msh(
                Nodes(
                    np.array(
                        [
                            [-20, -20, 0],
                            [-20, 20, 0],
                            [20, -20, 0],
                            [20, 20, 0],
                            [0, 0, 20],
                        ]
                    )
                ),
                Elements(
                    triangles=np.array(
                        [
                            [0, 1, 2],
                            [3, 2, 1],
                            [0, 1, 4],
                            [1, 3, 4],
                            [2, 3, 4],
                            [2, 0, 4],
                        ]
                    )
                    + 1
                ),
            ),
            None,
        ),
        TmsCoilModel(
            Msh(
                Nodes(
                    np.array(
                        [
                            [-20, -60, 0],
                            [-20, -20, 0],
                            [20, -60, 0],
                            [20, -20, 0],
                            [0, -40, 20],
                        ]
                    )
                ),
                Elements(
                    triangles=np.array(
                        [
                            [0, 1, 2],
                            [3, 2, 1],
                            [0, 1, 4],
                            [1, 3, 4],
                            [2, 3, 4],
                            [2, 0, 4],
                        ]
                    )
                    + 1
                ),
            ),
            None,
        ),
        TmsCoilModel(
            Msh(
                Nodes(
                    np.array(
                        [
                            [-20, 20, 0],
                            [-20, 60, 0],
                            [20, 20, 0],
                            [20, 60, 0],
                            [0, 40, 20],
                        ]
                    )
                ),
                Elements(
                    triangles=np.array(
                        [
                            [0, 1, 2],
                            [3, 2, 1],
                            [0, 1, 4],
                            [1, 3, 4],
                            [2, 3, 4],
                            [2, 0, 4],
                        ]
                    )
                    + 1
                ),
            ),
            None,
        ),
    ]
    deformations = [
        TmsCoilRotation(
            TmsCoilDeformationRange(0, (0, 90)),
            np.array([0, -20, 0]),
            np.array([40, -20, 0]),
        ),
        TmsCoilRotation(
            TmsCoilDeformationRange(0, (0, 90)),
            np.array([40, 20, 0]),
            np.array([0, 20, 0]),
        ),
    ]
    stimulator = TmsStimulator("SimNIBS-Stimulator")
    coil = TmsCoil(
        [
            LineSegmentElements(
                stimulator,
                np.array([[0, -10, 0], [10, 10, 0], [-10, 10, 0]]),
                casing=casings[0],
            ),
            LineSegmentElements(
                stimulator,
                np.array([[0, -50, 0], [10, -30, 0], [-10, -30, 0]]),
                casing=casings[1],
                deformations=[deformations[0]],
            ),
            LineSegmentElements(
                stimulator,
                np.array([[0, 30, 0], [10, 50, 0], [-10, 50, 0]]),
                casing=casings[2],
                deformations=[deformations[1]],
            ),
        ],
        limits=np.array([[-200, 200], [-200, 200], [-200, 200]]),
        resolution=np.array([10, 10, 10]),
    )

    return coil


@pytest.fixture(scope="module")
def small_self_intersecting_2_element_coil() -> TmsCoil:
    casings = [
        TmsCoilModel(
            Msh(
                Nodes(
                    np.array(
                        [
                            [-20, -20, 0],
                            [-20, 20, 0],
                            [20, -20, 0],
                            [20, 20, 0],
                            [0, 0, 20],
                        ]
                    )
                ),
                Elements(
                    triangles=np.array(
                        [
                            [0, 1, 2],
                            [3, 2, 1],
                            [0, 1, 4],
                            [1, 3, 4],
                            [2, 3, 4],
                            [2, 0, 4],
                        ]
                    )
                    + 1
                ),
            ),
            None,
        ),
        TmsCoilModel(
            Msh(
                Nodes(
                    np.array(
                        [[-2, -2, 0], [-2, 2, 0], [2, -2, 0], [2, 2, 0], [0, 0, 2]]
                    )
                    * 3
                    + np.array([0, 0, 15])
                ),
                Elements(
                    triangles=np.array(
                        [
                            [0, 1, 2],
                            [3, 2, 1],
                            [0, 1, 4],
                            [1, 3, 4],
                            [2, 3, 4],
                            [2, 0, 4],
                        ]
                    )
                    + 1
                ),
            ),
            None,
        ),
    ]
    deformations = [
        TmsCoilTranslation(TmsCoilDeformationRange(0, (-50, 50)), 0),
        TmsCoilTranslation(TmsCoilDeformationRange(0, (-50, 50)), 2),
        TmsCoilTranslation(TmsCoilDeformationRange(0, (-50, 50)), 0),
        TmsCoilTranslation(TmsCoilDeformationRange(0, (-50, 50)), 2),
    ]
    stimulator = TmsStimulator("SimNIBS-Stimulator")
    coil = TmsCoil(
        [
            LineSegmentElements(
                stimulator,
                np.array([[0, -10, 0], [10, 10, 0], [-10, 10, 0]]),
                casing=casings[0],
                deformations=[deformations[0], deformations[1]],
            ),
            LineSegmentElements(
                stimulator,
                np.array([[0, -1, 10], [1, 1, 10], [-1, 1, 10]]),
                casing=casings[1],
                deformations=[deformations[2], deformations[3]],
            ),
        ],
        limits=np.array([[-200, 200], [-200, 200], [-200, 200]]),
        resolution=np.array([10, 10, 10]),
        self_intersection_test=[[1, 2]],
    )

    return coil
