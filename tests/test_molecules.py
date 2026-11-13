import pytest

from src.molecules import (
    calculate_descriptors,
    morgan_fingerprint,
    parse_smiles,
    scaffold_key,
)


def test_parse_smiles_accepts_valid_molecule():
    molecule = parse_smiles("CCO")

    assert molecule is not None


@pytest.mark.parametrize("smiles", ["", "not-a-smiles"])
def test_calculate_descriptors_rejects_invalid_smiles(smiles):
    with pytest.raises(ValueError, match="Invalid SMILES"):
        calculate_descriptors(smiles)


def test_calculate_descriptors_returns_expected_properties():
    descriptors = calculate_descriptors("CCO")

    assert descriptors["Molecular weight (Da)"] == pytest.approx(46.07)
    assert descriptors["Hydrogen-bond donors"] == 1


def test_morgan_fingerprint_has_requested_size():
    fingerprint = morgan_fingerprint("CCO", n_bits=128)

    assert fingerprint.shape == (128,)
    assert set(fingerprint).issubset({0, 1})


def test_scaffold_key_groups_molecules_with_the_same_ring_scaffold():
    assert scaffold_key("c1ccccc1") == scaffold_key("Cc1ccccc1")
