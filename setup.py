from setuptools import setup, find_packages

setup(
    name="Swarmist",
    version="0.0.01",
    description="A DSL for building metaheuristics",
    author="Kevin Martins",
    url="https://github.com/kafm/swarmist",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    license="MIT",
    keywords=["swarmist", "metaheuristics", "optimization", "dsl"],
    packages=find_packages(),
    python_requires=">= 3.10",
    install_requires=["numpy>=1.24.3", "scipy>=1.10.1", "lark>=1.1.5", "PyMonad>=2.4.0", "optuna>=3.2.0"],
)
