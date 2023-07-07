#generate setup.py file
from setuptools import setup, find_packages

#TODO more professional. Add other details and improve existent ones
# extras_require={
    #     "dev": ["pytest>=7.0", "twine>=4.0.1"],
    # }


setup(
    name="Swarmist",
    version="0.0.1",
    description="A DSL for building metaheuristics",
    author="Kevin Martins",
    url="https://github.com/kafm/swarmist",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Software Development",
            "Topic :: Software Development :: Libraries",
            "Topic :: Software Development :: Libraries :: Python Modules",
            "Programming Language :: Python :: 3",
    ],
    license="BSD-3",
    keywords=["swarmist", "metaheuristics", "optimization", "Global optimization"],
    packages=find_packages(),
    python_requires=">= 3.11.3",
    install_requires=["numpy>=1.24.3", "numba>=0.57.0", "scipy>=1.10.1", "lark>=1.1.5", "PyMonad>=2.4.0"],
)