from setuptools import setup, find_packages
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


__version__ = "0.0.1"

packages = find_packages()
setup(
    name = 'autopt',
    version = __version__,
    packages = packages,
    include_package_data = True,
    py_modules = ['autopt.__init__', 'autopt.utils'],
    url = 'https://github.com/Vahram99/Auto',
    license = "GNU General Public License v3.0",
    author = read('AUTHORS.txt').replace('\n', ', ').replace('-', ''),
    author_email = 'vahram.babadjanyan@gmail.com',
    description = 'Automated hyperparameter optimization',
    long_description = read('README.md'),
    keywords = "machine-learning automated hyperparameter optimization",
)
