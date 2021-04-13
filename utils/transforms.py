def normalize(data, mean, std):
    data = (data - mean) / std
    return data


def denormalize(data, mean, std):
    data = (data * std) + mean
    return data
