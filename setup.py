
from setuptools import find_packages, setup

setup(
    name="sygnal",
    version="0.1.0",
    packages=find_packages(),
    description="A package for extract signal properties from given signals",
    author="lsy",
    license="MIT",
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    test_suite="tests",
)