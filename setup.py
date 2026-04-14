from setuptools import setup, find_packages

with open("requirements.txt") as f:
    req_list = f.read().splitlines()

setup(
    name = "MLOPS",
    version = "0.1.0",
    author = 'ZAINUL ABEDEEN',
    author_email = "zainulpasha589@gmail.com",

    packages = find_packages(),
    
    install_requires = req_list,
    python_requires = ">=3.7",
) 