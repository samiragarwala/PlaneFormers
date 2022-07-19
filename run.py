from planeformers.models.inference import *
from planeformers.figure_factory.visualize_mv import PlaneFormerInferenceVisualization
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument( "--imgs",  nargs="*",  type=str, help='List of images for inference')
parser.add_argument( "--output", type=str, default='output.pkl', help='Output file name')
args = parser.parse_args()

# loading model and checkpoint
params = get_default_dataset_config("plane_params")
ckpt = "./models/planeformers_eccv.pt"
mv_inference = MultiViewInference(params, ckpt)

# making predictions
preds = mv_inference.inference(args.imgs)

viz_info = {}
viz_info['preds'] = preds
viz_info['imgs'] = args.imgs

# saving predictions
with open(args.output, 'wb') as f:
    pickle.dump(viz_info, f)