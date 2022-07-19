import numpy as np
import sys
import pathlib
import json
import os
import time
import random
import itertools
from collections import namedtuple, OrderedDict

import bpy
from mathutils import Vector
import bpy_extras
import argparse



import sys
 
import subprocess



# Don't forget:
# - Set the renderer to Cycles
# - A ground plane set as shadow catcher
# - The compositing nodes should be [Image, RenderLayers] -> AlphaOver -> Composite 
# - The world shader nodes should be Sky Texture -> Background -> World Output
# - Set a background image node


if ".blend" in os.path.realpath(__file__):
    # If the .py has been packaged in the .blend
    curdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
else:
    curdir = os.path.dirname(os.path.realpath(__file__))


def setScene():
    bpy.data.scenes["Scene"].cycles.film_transparent = True
    try:
        bpy.data.scenes[0].render.engine = "CYCLES"

        # Set the device_type
        bpy.context.preferences.addons[
            "cycles"
        ].preferences.compute_device_type = "CUDA" # or "OPENCL"

        # Set the device and feature set
        bpy.context.scene.cycles.device = "GPU"

        # get_devices() to let Blender detects GPU device
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
        for d in bpy.context.preferences.addons["cycles"].preferences.devices:
            d["use"] = 1 # Using all devices, include GPU and CPU
            print(d["name"], d["use"])
    except:
        print('CPU render!')
    
    bpy.data.scenes["Scene"].cycles.samples = 128


def getVertices(obj, world=False):
    """Get the vertices of the object."""
    vertices = []
    if obj.data:
        if world:
            vertices.append([obj.matrix_world @ x.co for x in obj.data.vertices])
        else:
            vertices.append([x.co for x in obj.data.vertices])
    for child in obj.children:
        vertices.extend(getVertices(child, world=world))
    return vertices


def getObjBoundaries(obj):
    """Get the object boundary in image space."""
    cam = bpy.data.objects['Camera']
    scene = bpy.context.scene
    list_co = []
    vertices = getVertices(obj, world=True)
    for coord_3d in itertools.chain(*vertices):
        co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, coord_3d)
        list_co.append([x for x in co_2d])
    list_co = np.asarray(list_co)[:,:2]
    retval = list_co.min(axis=0).tolist()
    retval.extend(list_co.max(axis=0).tolist())
    return retval


def changeVisibility(obj, hide):
    """Hide or show object in render."""
    obj.hide_render = hide
    for child in obj.children:
        changeVisibility(child, hide)


def setCamera(pitch, roll, hfov, imh, imw, cam_pos=(0, 0, 1.6)):
    # Set camera parameters
    # uses a 35mm camera sensor model
    bpy.data.cameras["Camera"].sensor_width = 36
    cam = bpy.data.objects['Camera']
    cam.location = Vector(cam_pos)
    cam.rotation_euler[0] = -pitch + 90.*np.pi/180
    cam.rotation_euler[1] = -roll
    cam.rotation_euler[2] = 0
    bpy.data.cameras["Camera"].angle = hfov
    bpy.data.scenes["Scene"].render.resolution_x = imw
    bpy.data.scenes["Scene"].render.resolution_y = imh
    bpy.data.scenes["Scene"].render.resolution_percentage = 100
    bpy.context.view_layer.update()


def setObjectToImagePosition(object_name, ipv, iph):
    """insertion point vertical and horizontal (ipv, iph) in relative units."""

    bpy.context.view_layer.update()

    cam = bpy.data.objects['Camera']

    # Get the 3D position of the 2D insertion point
    # Get the viewpoint 3D coordinates
    frame = cam.data.view_frame()
    frame = [cam.matrix_world @ corner for corner in frame]

    # Perform bilinear interpolation
    top_vec = frame[0] - frame[3]
    bottom_vec = frame[1] - frame[2]
    top_pt = frame[3] + top_vec*iph
    bottom_pt = frame[2] + bottom_vec*iph
    vertical_vec = bottom_pt - top_pt
    unit_location = top_pt + vertical_vec*ipv

    # Find the intersection with the ground plane
    obj_direction = unit_location - cam.location
    length = -cam.location[2]/obj_direction[2]
    no_ground = False
    if length < 0:
        length = random.randint(2,10)
        no_ground = True

    # Set the object location
    bpy.data.objects[object_name].location = cam.location + obj_direction*length
    print(bpy.data.objects[object_name].location)

    bpy.context.view_layer.update()

    print("Object location: {}".format(bpy.data.objects[object_name].location))
    return no_ground


def changeBackgroundImage(bgpath, size):
    if "background" in bpy.data.images:
        previous_background = bpy.data.images["background"]
        bpy.data.images.remove(previous_background)

    img = bpy.data.images.load(bgpath)
    img.name = "background"

    bpy.data.images["background"].scale(*size)

    tree = bpy.context.scene.node_tree
    for node in tree.nodes:
        if isinstance(node, bpy.types.CompositorNodeImage):
            node.image = img
            break
    else:
        raise Exception("Could not find the background image node!")


def setParametricSkyLighting(theta, phi, t):
    """Use the Hosek-Wilkie sky model"""

    # Compute lighting direction
    x = np.sin(theta)*np.sin(phi)
    y = np.sin(theta)*np.cos(phi)
    z = np.cos(theta)

    # Remove previous link to Background and link it with Sky Texture
    link = bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].links[0]
    bpy.data.worlds["World"].node_tree.links.remove(link)
    bpy.data.worlds["World"].node_tree.links.new(
        bpy.data.worlds["World"].node_tree.nodes["Sky Texture"].outputs["Color"],
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0]
    )
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 1.0

    # Set Hosek-Wilkie sky texture
    bpy.data.worlds["World"].node_tree.nodes["Sky Texture"].sky_type = "HOSEK_WILKIE"
    bpy.data.worlds["World"].node_tree.nodes["Sky Texture"].sun_direction = Vector((x, y, z))
    bpy.data.worlds["World"].node_tree.nodes["Sky Texture"].turbidity = t
    bpy.data.worlds["World"].node_tree.nodes["Sky Texture"].ground_albedo = 0.3

    bpy.data.objects["Sun"].rotation_euler = Vector((theta, 0, -phi + np.pi))
    bpy.data.lights["Sun"].angle = 10*np.pi/180.
    bpy.data.lights["Sun"].energy = 4

    #bpy.data.objects["Sun"].hide = False
    bpy.data.objects["Sun"].hide_render = False


def setIBL(path, phi):
    """Use an IBL to light the scene"""

    # Remove previous IBL
    if "envmap" in bpy.data.images:
        previous_background = bpy.data.images["envmap"]
        bpy.data.images.remove(previous_background)

    img = bpy.data.images.load(path)
    img.name = "envmap"

    # Remove previous link to Background and link it with Environment Texture
    link = bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].links[0]
    bpy.data.worlds["World"].node_tree.links.remove(link)
    bpy.data.worlds["World"].node_tree.links.new(
        bpy.data.worlds["World"].node_tree.nodes["Environment Texture"].outputs["Color"],
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0]
    )
    bpy.data.worlds["World"].node_tree.nodes["Environment Texture"].image = img
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 2.0
    bpy.data.worlds["World"].node_tree.nodes["Mapping"].inputs[2].default_value[2] = phi + np.pi/2

    #bpy.data.objects["Sun"].hide = True
    bpy.data.objects["Sun"].hide_render = True


def performRendering(subfolder="render"):
# def performRendering(k, suffix="", subfolder="render"):

    # Flush scene modifications
    bpy.context.view_layer.update()

    os.makedirs(subfolder, exist_ok=True)

    # redirect output to log file
    logfile = 'blender_render.log'
    open(logfile, 'a').close()
    old = os.dup(1)
    sys.stdout.flush()
    os.close(1)
    os.open(logfile, os.O_WRONLY)

    # do the rendering
    # imgpath = os.path.join(curdir, '{}/{}{}.png'.format(subfolder, k, suffix))
    # #assert not os.path.isfile(imgpath)
    # bpy.data.scenes["Scene"].render.filepath = imgpath
    # bpy.ops.render.render(write_still=True)
    print(bpy.data.scenes["Scene"].render.filepath)
    bpy.data.scenes["Scene"].render.filepath = subfolder
    print(bpy.data.scenes["Scene"].render.filepath)
    # bpy.ops.render.render()
    bpy.ops.render.render(animation=True, write_still=True)
    # disable output redirection
    os.close(1)
    os.dup(old)
    os.close(old)

def setShadow(shadow_on):
    bpy.data.objects["Plane"].hide_render = not shadow_on

def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)

def set_obj(glb):
    bpy.ops.import_scene.gltf(filepath=glb)
    name = [o.name for o in bpy.context.selected_objects]
    assert len(name) == 1
    changeVisibility(bpy.data.objects[name[0]], hide=True)
    prev_data = bpy.data.objects['Template'].data
    bpy.data.objects['Template'].data = bpy.data.objects[name[0]].data
    return prev_data
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--output', type=str, default="render", help='output folder')
    parser.add_argument('--glb', type=str, default="", help='alternatively import from config')

    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    args = parser.parse_known_args(argv)[0]

    root = pathlib.Path(__file__).parent.resolve()
    # setScene()

    set_obj(args.glb)
    if True:
        ts = time.time()
        performRendering(subfolder=os.path.join(root, args.output))
        print("Rendering done in {0:0.3f}s".format(time.time() - ts))
        print("------------------------------")
