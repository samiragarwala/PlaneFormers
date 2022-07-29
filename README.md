## PlaneFormers: From Sparse View Planes to 3D Reconstruction

<h4>
Samir Agarwala, Linyi Jin, Chris Rockwell, David F. Fouhey
</br>
<span style="font-size: 14pt; color: #555555">
University of Michigan
</span>
</br>
ECCV 2022
</h4>
<hr>



## 1. Environment Setup

```bash
git clone https://github.com/samiragarwala/multi-surface-recon 
cd multi-surface-recon

# setting up conda environment
conda env create -f environment.yml
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.10/index.html
pip install -e .
git submodule update --init
cd SparsePlanes/sparsePlane
pip install -e .
```

## 2. Training Documentation 

```bash
python train.py --train --batch_size 40 --num_epochs 50 --model_name plane_camera_corr --use_l1_res_loss --transformer_on --d_model 899 --print_freq 250 --val_freq 1000 --emb_format balance_cam --optimizer sgdm --scheduler cos_annealing --t_max 40000 --use_plane_mask --nlayers 5 --use_plane_mask \
--use_appearance_embeddings --use_plane_params \
--kmeans_rot ./SparsePlanes/sparsePlane/models/kmeans_rots_32.pkl \
--kmeans_trans ./SparsePlanes/sparsePlane/models/kmeans_trans_32.pkl \
--sparseplane_config ./SparsePlanes/sparsePlane/tools/demo/config.yaml \
--json_path <Directory containing train/val/test jsons from sparseplanes> \
--path <Directory containing plane embedding dataset>
```

## 3. Inference Documentation 

### Inference Setup

- Please download the weights of the SparsePlanes model as per their [documentation](https://github.com/jinlinyi/SparsePlanes/blob/main/docs/demo.md) and save the weights under `PlaneFormers/models` 
- Download our pre-trained [model weights](https://drive.google.com/file/d/1KwOSdGisabu1rhASvf5-QQGDyZkzxdgS/view?usp=sharing) and save it under the `PlaneFormers/models` directory

### Running Inference

- To generate predictions for an arbitrary scene, please run the following command to generate a pickle file containing PlaneFormer predictions:

```bash
python run.py --output <output file name> --imgs img1_path img2_path ... imgN_path
```

### Visualizing Results

- Our current visualization code requires a different environment than that needed for the PlaneFormers model. Please install the entire SparsePlanes environment as a separate environment in conda as per their [documentation](https://github.com/jinlinyi/SparsePlanes/blob/main/docs/environment.md) to support visualization of our output.
- Run the following code to visualize plane correspondences and mesh objects for PlaneFormer predictions:

```bash
conda activate sparseplane
python viz.py --pred-file <output file name from inference step> --output-dir <directory to save visualizations>
```

## 4. Citation
If you find this code useful in your research, please consider citing:

```text
@inproceedings{agarwala2022planes,
      title={PlaneFormers: From Sparse View Planes to 3D Reconstruction}, 
      author={Samir Agarwala and Linyi Jin and Chris Rockwell and David F. Fouhey},
      booktitle = {ECCV},
      year={2022}
}
```

## 5. Acknowledgements
This work was supported by the DARPA Machine Common Sense Program. We would like to thank Richard Higgins and members of the Fouhey lab for helpful discussions and feedback.
