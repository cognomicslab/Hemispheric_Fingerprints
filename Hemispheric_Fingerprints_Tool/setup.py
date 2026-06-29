from setuptools import setup, find_packages

setup(
    name='hemispheric_fingerprints_tool',          
    version='1.0.0',          
    packages=find_packages(), 
    install_requires=[
        'torch',                    # torch, torch.nn, torch.distributed, torch.nn.parallel
        'numpy',                    # numpy
        'scikit-learn',             # sklearn.model_selection, sklearn.metrics
        'scipy',                    # scipy.interpolate, scipy.spatial.distance
        'captum',                   # captum.attr
        'nilearn',                  # nilearn.plotting, nilearn.image, nilearn.surface, nilearn.datasets
        'openpyxl',                 # openpyxl 
    ],
    author='Zhou Yuying',
    description='brain fingerprint extraction from fMRI using Integrated Gradients',
)