from setuptools import setup, find_packages
from pathlib import Path

here = Path(__file__).parent

with open(here / 'requirements.txt', 'r') as f:
    REQUIRED = f.readlines()

setup(
    name='GnssIntTorusAI',
    version='0.0.0dev',
    packages=find_packages(where='./gnssint'),
    install_requires=REQUIRED
)