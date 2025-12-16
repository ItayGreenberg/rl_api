from setuptools import setup, find_packages

setup(
    name="rl_api",
    version="0.2.0",
    packages=find_packages(include=["rl_api", "rl_api.*"]),
    install_requires=[
        "numpy",
        "torch",
        "tensorboard"
    ],
    python_requires=">=3.10",
)
