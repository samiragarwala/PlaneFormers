import cv2
import os
import sys
from glob import glob
import sys
import pdb
from tqdm import tqdm

def main(input_path):
    target_width = 320
    target_height = 480
    img_paths = sorted(list(glob(os.path.join(input_path, '*/*.png'))))
    for img_path in tqdm(img_paths):
        img = cv2.imread(img_path, -1)
        height = img.shape[0]
        width = img.shape[1]
        # factor = target_width / width
        # target_height = int(height * factor)
        factor = target_height / height
        target_width = int(width * factor)
        img = cv2.resize(img, (target_width, target_height))
        cv2.imwrite(img_path, img)

if __name__=='__main__':
    assert len(sys.argv) == 2
    path = sys.argv[1]
    main(path)