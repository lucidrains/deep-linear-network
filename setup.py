from setuptools import setup, find_packages

setup(
  name = 'deep-linear-network',
  packages = find_packages(),
  version = '0.0.1',
  license='MIT',
  description = 'Deep Linear Network - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/deep-linear-network',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
  ],
  install_requires=[
    'torch',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)