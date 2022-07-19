from setuptools import setup, find_packages

setup(
    name='planeformers',
    version='1.0',
    author='Samir Agarwala, Linyi Jin, Chris Rockwell, David Fouhey',
    description='Code for PlaneFormers: From Sparse View Planes to 3D Reconstruction',
    packages=find_packages(exclude=("multiview_dataset", "scripts")),
    install_requires=['detectron2', 'pytorch3d', 'fvcore', 'torchvision>=0.11'],
)