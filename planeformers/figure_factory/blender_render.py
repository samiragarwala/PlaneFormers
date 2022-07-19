"""
Tested on Blender 2.8.1.
blender -b --python render_arti.py
"""
import bpy
import mathutils
from math import pi
from glob import glob
import os
import json
import mathutils
from mathutils import Matrix, Vector
import random
import pdb
import sys
from math import tan
import numpy as np
import hashlib
import socket
file_path = os.path.realpath(__file__)
sys.path.insert(0, os.path.dirname(file_path))
#import camera_utils

scene = bpy.context.scene

# render engine and device
scene.render.engine = 'CYCLES'
prefs = bpy.context.preferences.addons['cycles'].preferences
print(prefs.get_devices())
prefs.compute_device_type = "CUDA"
print(prefs.devices[0])
prefs.devices[0].use = True


cameras = [
    # side view 1
    {
        'location': (0, -5, 2),
        'rotation_euler': (70 / 180 * pi, 0, 0),
    },
    # side view 2 (right)
    {
        'location': (5, -5, 2),
        'rotation_euler': (70 / 180 * pi, 0, 45 / 180 * pi),
    },
    # side view 3 (left)
    {
        'location': (-5, -5, 2),
        'rotation_euler': (70 / 180 * pi, 0, - 45 / 180 * pi),
    },
    # top view 1
    {
        'location': (0, 0, 5),
        'rotation_euler': (20 / 180 * pi, 0, 0),
    },
    # top view 2
    {
        'location': (0, 0, 8),
        'rotation_euler': (20 / 180 * pi, 0, 0),
    }
]

def initialize(camera, output_path):
    # Set the device and feature set
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.cycles.feature_set = "SUPPORTED"

    # render resolution
    #scene.render.resolution_x = 640
    #scene.render.resolution_y = 480
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 960
    scene.render.film_transparent = True

    # render layers
    # we need render Normal and Vector pass to get surface normal and optical flow
    #scene.render.use_single_layer = True
    #scene.view_layers["View Layer"].use_pass_normal = True
    #scene.view_layers["View Layer"].use_pass_vector = True

    # frame
    scene.frame_start = 1
    scene.frame_end = 1
    #scene.frame_current = scene.frame_start

    # remove objects other than Camera and Light
    # by default, there are ['Camera', 'Cube', 'Light']
    # we need remove the cube.
    bpy.ops.object.select_all(action='DESELECT') # deselect all
    for obj_name in bpy.data.objects.keys():
        if obj_name == 'Camera':# or obj_name == 'Light':
            continue
        bpy.data.objects[obj_name].select_set(True)
    bpy.ops.object.delete() 

    # add light
    #bpy.ops.object.light_add(type='SUN', location=(0, -2, 0))
    #bpy.data.objects['Sun'].rotation_euler = (pi/2, 0, 0)
    bpy.ops.object.light_add(type='SUN', location=(5, -5, 2))
    bpy.data.objects['Sun'].rotation_euler = (70 / 180 * pi, 0, 45 / 180 * pi)
    bpy.data.objects['Sun'].data.energy = 5
    bpy.ops.object.light_add(type='SUN', location=(-5, -5, 2))
    bpy.data.objects['Sun.001'].rotation_euler = (70 / 180 * pi, 0, - 45 / 180 * pi)
    bpy.data.objects['Sun.001'].data.energy = 5
    #bpy.data.objects['Sun'].rotation_euler = (70 / 180 * pi, 0, 45 / 180 * pi)
    #bpy.data.objects['Sun'].data.energy = 5

    # set up the camera
    bpy.data.objects['Camera'].data.lens_unit = 'FOV'
    bpy.data.objects['Camera'].data.angle = 1.10724
    bpy.data.objects['Camera'].location = camera['location']
    bpy.data.objects['Camera'].rotation_euler = camera['rotation_euler']
    bpy.context.view_layer.update()
    cam = bpy.data.objects['Camera']
    
    # output settings
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = output_path

    return None


def list_objects():
    # list all objects
    # by default, it's ['Camera', 'Cube', 'Light']
    return bpy.data.objects.keys()


def load_obj(filepath):
    """
    load .obj file into current scene.
    """
    # load from obj file
    imported_object = bpy.ops.import_scene.obj(filepath=filepath)
    objs = bpy.context.selected_objects

    if len(objs) > 1:
        # merge objects
        bpy.context.view_layer.objects.active = objs[0]
        bpy.ops.object.join()
        print(bpy.context.selected_objects)

    # return the merged object
    obj = bpy.context.selected_objects[0]
    return obj


def rotate_obj(obj, angle, axis):
    rot_mat = Matrix.Rotation(angle, 4, axis)

    # decompose world_matrix's components, and from them assemble 4x4 matrices
    orig_loc, orig_rot, orig_scale = obj.matrix_world.decompose()
    orig_loc_mat = Matrix.Translation(orig_loc)
    orig_rot_mat = orig_rot.to_matrix().to_4x4()
    orig_scale_mat = Matrix.Scale(orig_scale[0],4,(1,0,0)) @ Matrix.Scale(orig_scale[1],4,(0,1,0)) @ Matrix.Scale(orig_scale[2],4,(0,0,1))

    # assemble the new matrix
    obj.matrix_world = orig_loc_mat @ rot_mat @ orig_rot_mat @ orig_scale_mat


def translate_obj(obj, dis, axis):
    loc_mat = Matrix.Translation(dis * axis)

    # decompose world_matrix's components, and from them assemble 4x4 matrices
    orig_loc, orig_rot, orig_scale = obj.matrix_world.decompose()
    orig_loc_mat = Matrix.Translation(orig_loc)
    orig_rot_mat = orig_rot.to_matrix().to_4x4()
    orig_scale_mat = Matrix.Scale(orig_scale[0],4,(1,0,0)) @ Matrix.Scale(orig_scale[1],4,(0,1,0)) @ Matrix.Scale(orig_scale[2],4,(0,0,1))

    # assemble the new matrix
    obj.matrix_world = loc_mat @ orig_loc_mat @ orig_rot_mat @ orig_scale_mat 


def transform_obj(obj, transform):
    obj.matrix_world = transform @ obj.matrix_world


def random_transform():
    """
    Generate an initial random transformation for 
    """
    mat_trans = Matrix.Translation((random.random() * 4 - 2,  random.random() * 4 - 2, random.random() * 4 - 2))
    axis = (random.random(), random.random(), random.random())
    angle = random.random() - 0.5
    mat_rot = Matrix.Rotation(angle, 4, axis)
    mat_scale = Matrix.Scale(random.random() + 0.5, 4)

    # identity transformation
    #mat_trans = Matrix.Translation((0, 0, 0))
    #mat_rot = Matrix.Identity(4)
    #mat_scale = Matrix.Scale(1, 4)

    m = mat_trans @ mat_rot @ mat_scale
    return m, mat_rot


def project_rot_axis(P, center, direction):
    pt1_3d = center
    pt2_3d = center.to_3d() + direction.to_3d() * 0.1
    pt1 = P @ pt1_3d
    pt2 = P @ pt2_3d
    pt1 /= pt1[2]
    pt2 /= pt2[2]
    pt1 = [pt1[0], pt1[1]]
    pt2 = [pt2[0], pt2[1]]
    pt2[0] = pt1[0] + (pt2[0] - pt1[0]) * 100
    pt2[1] = pt1[1] + (pt2[1] - pt1[1]) * 100
    pt1 = (int(pt1[0]), int(pt1[1]))
    pt2 = (int(pt2[0]), int(pt2[1]))
    return pt1, pt2


def main():
    vis_dir = sys.argv[-1]
    print("visualizing {}".format(vis_dir))

    vis_dir = sys.argv[-1]
    obj_files = glob(os.path.join(vis_dir, '*/*.obj'))
    obj_files.sort()
    #obj_files = obj_files[:1]
    
    for obj_file in obj_files:
        print(obj_file)
        for camera_id, camera in enumerate(cameras):
            obj_dir, obj_fn = os.path.split(obj_file)
            prefix = obj_fn.split('.')[0] + '_{:0>4}_'.format(camera_id)
            output_path = os.path.join(obj_dir, prefix)

            # initialize
            initialize(camera, output_path)

            # load obj file
            obj = load_obj(obj_file)

            # render it
            bpy.ops.render.render(animation=True)



if __name__=='__main__':
    main()