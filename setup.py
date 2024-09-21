import setuptools
from src.pyfem import __version__

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyfem",
    version=__version__,
    author="Jingyu Sun",
    author_email="sun.jingyu@outlook.com",
    description="Python Finite Element Method",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sunwhale/pyfem",
    project_urls={
        "Bug Tracker": "https://github.com/sunwhale/pyfem/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.9",
    entry_points={
        'console_scripts': [
            'pyfem=pyfem.__main__:main'
        ]
    },
    install_requires=[
        'numpy',
        'scipy',
        'meshio',
        'tomli',
        'colorlog',
        'sympy',
        'tomli_w',
        'h5py'
    ],
)