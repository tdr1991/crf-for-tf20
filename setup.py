from setuptools import setup
from setuptools import find_packages
import os


setup(name='crf',
      version='0.0.1',
      description='CRF layer for tensorflow-2.x',
      author='caishiqing',
      author_email='caishiqing@tom.com',
      url='https://github.com/caishiqing/crf-for-tf20.git',
      license='MIT',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())
