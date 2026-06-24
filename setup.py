from setuptools import setup, find_packages


setup(
    name='mlearn', 
    version='0.0.2', 
    packages=find_packages(),
    description='classic machine learning algorithm implementation with minimum lines of code',
    install_requires = ['numpy', 'loguru'],
    extras_require={'torch': ['torch']},
    scripts=[],
    python_requires = '>=3',
    include_package_data=True,
    author='Liu Shengli',
    url='http://github.com/gseismic/mlearn.py',
    zip_safe=False,
    author_email='liushengli203@163.com'
)
