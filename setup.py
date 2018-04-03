"""
-- User: Ashok Kumar Pant (asokpant@gmail.com)
-- Date: 3/30/18
-- Time: 9:27 PM
"""

from setuptools import setup
from setuptools import find_packages

install_requires = {
    'numpy',
    'scipy',
    'tqdm'
}

setup(
    name="capsbrain",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    author="Ashok Kumar Pant",
    author_email="asokpant@gmail.com",
    url="https://github.com/ashokpant/capsbrain",
    license="Apache-2.0",
    install_requires=install_requires,
    description="capsbrain: TensorFlow capsule networks",
    keywords="matrix capsules, capsule, capsNet, em routing,  deep learning, tensorflow",
    platform=['any']
)
