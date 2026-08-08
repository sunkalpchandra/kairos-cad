

def test_collate_trims_to_the_batch_maximum():
    """Requirements use at most 51 of 64 padded positions, and the language
    encoder is most of the model's forward cost."""
    import torch

    from kairos.training.bc_dataset import collate

    def row(real_len):
        mask = torch.zeros(64, dtype=torch.long)
        mask[:real_len] = 1
        return {
            "token_ids": torch.randint(0, 50, (64,)),
            "token_values": torch.zeros(64),
            "token_mask": mask,
            "numeric": torch.zeros(4),
            "history": torch.zeros(8, dtype=torch.long),
            "operation_mask": torch.ones(38, dtype=torch.long),
            "operation": torch.tensor(0),
            "parameters": torch.zeros(6),
        }

    batch = collate([row(20), row(37), row(11)])
    assert batch.token_ids.shape[1] == 37
    assert batch.token_mask.shape[1] == 37
    assert batch.token_values.shape[1] == 37
    # Non-text tensors are untouched.
    assert batch.numeric.shape[1] == 4


def test_collate_never_trims_below_the_longest_sequence():
    """A constant cap would silently truncate a longer requirement mid-spec."""
    import torch

    from kairos.training.bc_dataset import collate

    mask = torch.zeros(64, dtype=torch.long)
    mask[:64] = 1
    row = {
        "token_ids": torch.randint(0, 50, (64,)),
        "token_values": torch.zeros(64),
        "token_mask": mask,
        "numeric": torch.zeros(4),
        "history": torch.zeros(8, dtype=torch.long),
        "operation_mask": torch.ones(38, dtype=torch.long),
        "operation": torch.tensor(0),
        "parameters": torch.zeros(6),
    }
    assert collate([row]).token_ids.shape[1] == 64


def test_collate_keeps_at_least_one_column():
    """An all-padding batch must not produce a zero-width tensor."""
    import torch

    from kairos.training.bc_dataset import collate

    row = {
        "token_ids": torch.zeros(64, dtype=torch.long),
        "token_values": torch.zeros(64),
        "token_mask": torch.zeros(64, dtype=torch.long),
        "numeric": torch.zeros(4),
        "history": torch.zeros(8, dtype=torch.long),
        "operation_mask": torch.ones(38, dtype=torch.long),
        "operation": torch.tensor(0),
        "parameters": torch.zeros(6),
    }
    assert collate([row]).token_ids.shape[1] == 1
