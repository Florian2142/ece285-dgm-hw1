import pytest
import torch

from dgm.evaluation.sampling import matched_class_indices


def test_class_sorted_training_data_matches_reference_without_duplicates_or_rng_changes():
    labels = torch.tensor([0] * 30 + [1] * 70)
    reference = torch.tensor([0] * 8 + [1] * 12)
    state = torch.random.get_rng_state().clone()
    indices = matched_class_indices(labels, reference, seed=0)
    assert len(indices.unique()) == len(reference)
    assert torch.bincount(labels[indices]).tolist() == [8, 12]
    assert torch.equal(torch.random.get_rng_state(), state)
    assert torch.equal(indices, matched_class_indices(labels, reference, seed=0))
    assert not torch.equal(indices, matched_class_indices(labels, reference, seed=1))


@pytest.mark.parametrize("labels,reference", [([0, 0], [1]), ([0, 1], [1, 1])])
def test_missing_or_insufficient_class_is_rejected(labels, reference):
    with pytest.raises(ValueError, match="need"):
        matched_class_indices(torch.tensor(labels), torch.tensor(reference))


@pytest.mark.parametrize("labels,reference", [([], []), ([[0]], [0]), ([0], [[0]])])
def test_empty_reference_or_wrong_label_shape_is_rejected(labels, reference):
    with pytest.raises(ValueError, match="one-dimensional"):
        matched_class_indices(torch.tensor(labels), torch.tensor(reference))
