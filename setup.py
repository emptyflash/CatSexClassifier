from setuptools import setup, find_packages
from pip.req import parse_requirements, download

install_reqs = parse_requirements('requirments.txt', session=download.PipSession())

setup(
    name = "CatSexClassifier",
    version = "0.1",
    packages = find_packages(),
    scripts = ['train_network.py', 'transform_data.py', 'get_dataset.py'],
    install_requires = [str(ir.req) for ir in install_reqs],
    package_data = {
    '': ['*.dat'],
    },

    # metadata for upload to PyPI
    author = "EmptyFlash",
    author_email = "emptyflash@gmail.com",
    description = "Python code to train neural network for identifying the sex of cats",
    license = "PSF",
    keywords = "train neural network cat sex sexer identify",
)
