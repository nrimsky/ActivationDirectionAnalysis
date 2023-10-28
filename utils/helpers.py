def make_tensor_save_suffix(layer, model_name_path, dataset_name):
    return f'{layer}_{model_name_path.split("/")[-1]}_{dataset_name}'
