from add_noise import add_noise


NOISE_SIZE = 0.1


def hider(ori_data):
    return add_noise(ori_data, NOISE_SIZE)
