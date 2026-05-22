import torch


def _index_on_tensor_device(indices: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    return indices.to(device=tensor.device, dtype=torch.long, non_blocking=True)


def get_first_token_from_combo_tokens(
    tensor: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    start_indices = torch.cumsum(
        torch.cat((lengths.new_zeros(1), lengths[:-1])), dim=0
    )
    return tensor.index_select(0, _index_on_tensor_device(start_indices, tensor))


def get_last_token_from_combo_tokens(
    tensor: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    end_indices = torch.cumsum(lengths, dim=0) - 1
    return tensor.index_select(0, _index_on_tensor_device(end_indices, tensor))
