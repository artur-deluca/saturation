import imageio
import os
import re

def create_gif(path):
    pretrain = natural_sort([x for x in os.listdir(path) if x.endswith(".png") and "Pretrain" in x])
    iteration = natural_sort([x for x in os.listdir(path) if x.endswith(".png") and "Iteration" in x])
    duration = [0.2 for x in pretrain + iteration]
    duration[0] = 1.5
    duration[len(pretrain)] = 1.5
    duration[-1] = 1.5
    with imageio.get_writer("{}/animated.gif".format(path), mode="I", duration=duration) as writer:

        for filename in pretrain:
            image = imageio.imread(os.path.join(path, filename))
            writer.append_data(image)

        for filename in iteration:
            image = imageio.imread(os.path.join(path, filename))
            writer.append_data(image)


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)