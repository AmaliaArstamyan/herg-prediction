import numpy
import pandas
import matplotlib
import sklearn
import streamlit
import rdkit


def test_core_dependencies_import():
    assert numpy.__version__
    assert pandas.__version__
    assert matplotlib.__version__
    assert sklearn.__version__
    assert streamlit.__version__
    assert rdkit.__version__
