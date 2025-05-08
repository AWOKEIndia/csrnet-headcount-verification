from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="csrnet-headcount",
    version="1.0.0",
    author="AWOKE India",
    author_email="it@awokeindia.com",
    description="Headcount verification system based on CSRNet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/awokeindia/csrnet-headcount-verification",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "csrnet-train=src.csrnet_implementation:main",
            "csrnet-prepare=src.dataset_preparation:main",
            "csrnet-evaluate=src.model_evaluation:main",
            "csrnet-video=src.video_processing:main",
            "csrnet-gui=src.gui_application:main",
        ],
    },
)
