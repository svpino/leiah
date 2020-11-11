from setuptools import setup, find_packages

setup(
    name="leiah",
    version="0.0.3",
    description="Leiah",
    url="https://github.com/svpino/leiah",
    author="Santiago L. Valdarrama",
    author_email="svpino@gmail.com",
    packages=find_packages(exclude=["test"]),
    install_requires=["PyYAML==5.3.1", "sagemaker==2.15.0"],
    zip_safe=False,
)
