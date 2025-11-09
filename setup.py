from setuptools import setup, find_packages
   
setup(
    name="bovw_gcn",
    version="0.1.0",
    author="Montassar Mastour",
    author_email="mastourmontassar@gmail.com",
    description="BoVW-GCN for Remote Sensing Classification",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
    ],
)
