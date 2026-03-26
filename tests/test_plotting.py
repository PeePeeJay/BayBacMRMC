import pytest
from mrmc_baybac.plotting import psa_result


@pytest.fixture
def psa_result_dir():
    """Fixture to create a temporary directory for PSA results."""
    return "/Users/ppjacobs/PythonProjects/BayBacMRMC/tests/psa_interaction"


@pytest.fixture
def psa_plot_arguments(psa_result_dir):
    """Fixture to provide arguments for the psa_result plotting function."""
    return {
        "psa_result_dir": psa_result_dir,
        # Add other necessary arguments here
    }


def test_psa_result_raises_when_path_is_not_a_directory(
    psa_result_dir,
):
    fig = psa_result(
        psa_result_dir,
        case_interaction=True
    )
    assert fig is not None
