import os
from importlib.machinery import SourceFileLoader

from setuptools import setup, find_packages

module_name = 'boseed'

module = SourceFileLoader(
    module_name,
    os.path.join(module_name, '__init__.py')
).load_module()


def load_requirements(fname):
    """ load requirements from a pip requirements file """
    with open(fname) as f:
        line_iter = (line.strip() for line in f.readlines())
        return [
            line for line in line_iter
            if line and line[0] != '#' and not line.startswith('git+git')
        ]

setup(
    name=module_name.replace('_', '-'),
    version=module.__version__,
    author=module.__author__,
    author_email=module.authors_email,
    license=module.__license__,
    description=module.package_info,
    long_description=open("README.md").read(),
    platforms="all",
    classifiers=[
        'Intended Audience :: Developers',
        'Natural Language :: Russian',
        'Operating System :: MacOS',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    python_requires='>=3.6<3.7',
    packages=find_packages(),
    install_requires=load_requirements('requirements.txt'),
    entry_points={
        'console_scripts': [
            '{0} = {0}.main:main'.format(module_name)
        ]
    },
    include_package_data=True,
)
