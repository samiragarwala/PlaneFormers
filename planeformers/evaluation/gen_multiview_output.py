import json
import argparse
import pickle
from tqdm import tqdm
from planeformers.utils.misc import *

# generates optimized dict with predictions that is used for eval
def gen_optimized_dict(args, params, ckpt_name):

    mv_inference = MultiViewInference(params, ckpt_name)

    with open(args.json_file, 'r') as f:
        mv_data = json.load(f)['data']

    opt_dict = {}
    for i in tqdm(range(len(mv_data))):
        img_paths = []
        for j in range(args.num_images):
            img_paths.append(mv_data[i][str(j)]['file_name'])

        opt_dict[i] = mv_inference.inference(img_paths)

    with open(args.output_file, 'wb') as f:
        pickle.dump(opt_dict, f)


if __name__=="__main__":

    params = get_default_dataset_config("plane_params")
    
    parser = argparse.ArgumentParser(description='Transformer model for planar embedding correspondences')
    parser.add_argument('--num_images', type=int)
    parser.add_argument('--json_file')
    parser.add_argument('--output_file')
    parser.add_argument('--ckpt', help="checkpoint for generating eval file")

    args = parser.parse_args()
    gen_optimized_dict(args, params, args.ckpt)



