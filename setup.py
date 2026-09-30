from setuptools import setup, find_packages

setup(
    name="Genox",  # Package name
    version="0.1",   # Initial version
    packages=find_packages(),  # Automatically find subpackages
    install_requires=[  # If your package has dependencies, list them here
        "pandas",
        "ase",
        "networkx",
        "spglib",
        "pot",
        "pymatgen",
        "optuna",
        "torch",
        "numpy",
        "scikit-learn",
        "hdbscan",
        "dscribe"
    ],
    author="Helmi_Risan",
    author_email="helmi.risansyauqi@tuwien.ac.at",
    description="A Python package for metal oxide analysis",
    long_description=open("README.md").read(),
)