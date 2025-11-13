from setuptools import setup, find_packages
   
setup(
    name="bovw_gcn",
    version="0.1.0",
    author="Montassar Mastour",
    author_email="mastourmontassar@gmail.com",
    description="BoVW-GCN for Remote Sensing Classification",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=open("requirements.txt").read().splitlines(),
)
