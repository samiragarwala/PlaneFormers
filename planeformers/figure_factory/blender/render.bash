/Applications/Blender.app/Contents/MacOS/Blender --background template.blend  \
--python render.py -- --output tmp_render/ \
--glb 03_gt.glb
python video.py --output test.mp4 --input tmp_render
