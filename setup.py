from setuptools import setup, find_packages
from codecs import open

# Get the description from the README file
with open("README.md") as file:
    long_description = file.read()

setup(
    name="my-tensors",
    version="0.1.0",
    description="Practice library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RunnersNum40/my-tensors",
    author="Ted Pinkerton",
    license="GNU General Public License v3.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3.0",
        "Operating System :: OS Independent"
    ],
    packages=find_packages(include=['my_tensors']),
    include_package_data=True,
    install_requires=["numpy"],
    setup_requires=["pytest-runner"],
)
