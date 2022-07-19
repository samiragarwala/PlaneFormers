import imageio
from glob import glob
import os
import shutil
import argparse
import subprocess
from os.path import join
from tqdm import tqdm
# examples = [f for f in sorted(list(glob('/Pool1/users/jinlinyi/public_html/p-multi-surface/eccv/v05_video/*/*'))) if os.path.isdir(f)]
examples = [f for f in sorted(list(glob('/Users/jinlinyi/Downloads/v05_video/*/*'))) if os.path.isdir(f)]

for example in tqdm(examples):
    glbs = [f for f in os.listdir(example) if f.endswith('.glb')]
    for glb in glbs:
        tqdm.write(join(example,glb))
        subprocess.run(f"/Applications/Blender.app/Contents/MacOS/Blender --background template.blend --python render.py -- --output tmp_render/ --glb {join(example, glb)}", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
        # subprocess.run(f"CUDA_VISIBLE_DEVICES=0 /Pool1/users/jinlinyi/local/blender-3.0.1-linux-x64/blender --background template.blend --python render.py -- --output tmp_render/ --glb {join(example, glb)}", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
        subprocess.run(f"python video.py --output {join(example, glb.replace('.glb', '.mp4'))} --input tmp_render", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
        # exit()
    # exit()