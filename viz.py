import sys
import os
import pickle
from planeformers.figure_factory.visualize_mv import PlaneFormerInferenceVisualization
import argparse


parser = argparse.ArgumentParser()
parser.add_argument( "--pred-file", type=str, default='output.pkl')
parser.add_argument( "--output-dir", type=str, default='output')
args = parser.parse_args()

with open(args.pred_file, 'rb') as f:
    viz_info = pickle.load(f)

preds = viz_info['preds']
img_list = viz_info['imgs']

# visualizing reconstruction
visualizer = PlaneFormerInferenceVisualization((img_list))
save_folder = args.output_dir
os.makedirs(save_folder, exist_ok=True)
visualizer.save_input(save_folder, 'original')
visualizer.save_multiple_objects(tran_topk=-2, rot_topk=-2, output_dir=save_folder,
            rcnn_output=preds[0], prefix='planeformers', pred_camera=preds[1], \
            show_camera=True, corr_list=preds[3], webvis=False)
visualizer.save_matching(rcnn_output=preds[0], assignment=preds[3], \
            output_dir=os.path.join(save_folder, 'corr'), prefix='planeformers', \
            paper_img=True, vertical=False)