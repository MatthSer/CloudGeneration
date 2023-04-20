import os
import imageio as iio
import numpy as np
import tifffile
from CloudPerlin import CloudPerlin


# if you need to access a file next to the source code, use the variable ROOT
# for example:
#    torch.load(os.path.join(ROOT, 'weights.pth'))


def main(input, res, octave):
    u = tifffile.imread(input)

    cloud, mask = CloudPerlin.cloud_generation((1024, 1024), (res, res), octave)
    cloudy = CloudPerlin.cloud_copy(u, cloud)

    if not os.path.exists('output/'):
        os.mkdir('output/')

    # Rescale image on 8 bits for visualization
    u_8bits = CloudPerlin.convert_float32_to_uint8(u)
    cloudy_8bits = CloudPerlin.convert_float32_to_uint8(cloudy)

    # Save output
    tifffile.imwrite('output/background.png', u_8bits)
    tifffile.imwrite('output/cloud.png', (cloud * 255).astype(np.uint8))
    tifffile.imwrite('output/cloudy.png', cloudy_8bits)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', dest='input', type=str, required=True, help='input image')
    parser.add_argument('-r', '--res', dest='res', type=int, default=2, help="res noise parameter")
    parser.add_argument('-o', '--octave', dest='octave', type=int, default=7, help="octave noise parameter")
    args = parser.parse_args()

    main(args.input, args.res, args.octave)
