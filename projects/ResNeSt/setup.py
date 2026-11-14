import io
import os
import subprocess

from setuptools import setup, find_packages

cwd = os.path.dirname(os.path.abspath(__file__))

version = '0.0.6'
try:
    if not os.getenv('RELEASE'):
        from datetime import date
        today = date.today()
        day = today.strftime("b%Y%m%d")
        version += day
except Exception:
    pass

def create_version_file():
    global version, cwd
    print('-- Building version ' + version)
    version_path = os.path.join(cwd, 'resnest', 'version.py')
    with open(version_path, 'w') as f:
        f.write('"""This is resnest version file."""\n')
        f.write("__version__ = '{}'\n".format(version))

requirements = [
    'numpy',
    'tqdm',
    'nose',
    'torch>=1.0.0',
    'Pillow',
    'scipy',
    'requests',
    'iopath',
    'fvcore',
]

if __name__ == '__main__':
    create_version_file()
    setup(
        name="resnest",
        version=version,
        description="ResNeSt",
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
        license='Apache-2.0',
        install_requires=requirements,
        packages=find_packages(exclude=["scripts", "examples", "tests"]),
    )

