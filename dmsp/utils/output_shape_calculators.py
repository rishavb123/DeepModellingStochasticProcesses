import torch

def calc_encoder_output_size(module_lst, input_sizes, aggregator):
    print("Calling function to calculate output shape:")
    print(module_lst, input_sizes)

    assert len(module_lst) == len(input_sizes), "Length of module lists and input sizes must be the same."

    import pdb; pdb.set_trace()
    inp_tensors = [torch.zeros((1, inp_size)) for inp_size in input_sizes]
    out_tensor = aggregator([module(inp_tensor) for module, inp_tensor in zip(module_lst, inp_tensors)], dim=-1)

    out_size = out_tensor.flatten().shape[0]

    return out_size