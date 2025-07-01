def print_all_modules(module, prefix=''):
    for name, submodule in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        print(full_name, '->', submodule)
        print_all_modules(submodule, full_name)


def print_state_dict_info(module):
    print("\nState Dict:")
    for key, tensor in module.state_dict().items():
        print(f"{key}: shape={tuple(tensor.shape)}, num_elements={tensor.numel()}")