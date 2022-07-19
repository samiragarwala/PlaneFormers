import imageio
import glob
import os
import shutil
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--output', type=str, default="render", help='output folder')
parser.add_argument('--input', type=str, default="", help='alternatively import from config')
args = parser.parse_args()
print(args)
writer = imageio.get_writer(args.output, fps=40)
imgs = list(sorted(glob.glob(os.path.join(args.input, '*'))))
for file in imgs:
    im = imageio.imread(file)
    writer.append_data(im)
imgs.reverse()
for file in imgs:
    im = imageio.imread(file)
    writer.append_data(im)
writer.close()
shutil.rmtree(args.input)