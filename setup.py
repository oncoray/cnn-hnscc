from setuptools import setup, find_packages

setup(
    name='dl_toolbox',
    version="0.1",
    description="useful functions for deep learning with keras and survival analysis",
    long_description="Install via `pip install -r requirements.txt'; 'python setup.py install'",
    url="https://gitlab.hzdr.de/starke88/dl_toolbox",
    author="Sebastian Starke",
    author_email="s.starke@hzdr.de",
    packages=find_packages(),
    zip_safe=False,
    include_package_data=True,
    install_requires=[
        'keras',
        'lifelines',
        'matplotlib',
        'nibabel',
        'numpy',
        'pandas',
        'pillow',
        'statsmodels',
        'tensorflow',
        'keras_vis',
        'innvestigate'
    ],
)
