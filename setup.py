from setuptools import setup, find_packages

setup(
    name='mat2jpeg',
    version='0.1.0',
    description='Converts .mat files to .jpeg files using mel spectrogram',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'librosa',
        'scipy',
        'matplotlib',
        'argparse',
        'concurrent.futures'
    ],

    )
