# -*- coding: utf-8 -*-                                                                                                                                                                                                                                              
from setuptools import setup, find_packages

# with open("/home/amarchal/GASKAP/Readme.md") as f:
#         readme = f.read()
	
# with open("/home/amarchal/GASKAP/LICENSE") as f:
# 	license = f.read()

setup(
    name='SPARK',
    version='0.1.0',
    description='SPARK python package for GASKAP users',
    # long_description=readme,
    classifiers=[
        'Development status :: 1 - Alpha',
        'License :: CC-By-SA2.0',
        'Programming Language :: Python',
        'Topic :: Data Analysis'
    ],
    author='Antoine Marchal w/ GASKAP',
    author_email='amarchal@cita.utoronto.ca',
    url='https://github.com/GASKAP/SPARK',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
            'numpy',
    ],
    include_package_data=True
)
