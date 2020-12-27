#!/usr/bin/env python
import os
from setuptools import find_packages, setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


if __name__ == '__main__':
    setup(
        name='pardet-pytorch',
        version="v0.1",
        description='pardet-pytorch for Pedestrian Attribute Recognition.',
        long_description=read('README.md'),
        long_description_content_type='text/markdown',
        author='wduo',
        author_email='wduo2017@163.com',
        keywords='computer vision, pedestrian attribute recognition',
        url='https://github.com/wduo/pardet-pytorch',
        packages=find_packages(exclude=('configs', 'tools', 'demo')),
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
        license='Apache License 2.0',

        ext_modules=[],
        zip_safe=False)
