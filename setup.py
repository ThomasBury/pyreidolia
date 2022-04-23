import os.path
from setuptools import setup, find_packages

# The directory containing this file
HERE = os.path.abspath(os.path.dirname(__file__))

# The text of the README file
with open(os.path.join(HERE, "README.md")) as fid:
    README = fid.read()

with open(os.path.join(HERE, "requirements.txt")) as req:
    REQUIREMENTS = req.read().splitlines()


EXTRAS_REQUIRE = {'plotting': ['matplotlib>=2.2.0', 'jupyter'], 'setup': ['pytest-runner', 'flake8'], 'test': ['pytest']}

KEYWORDS = "pareidolia, segmentation, clouds, meteorology"

setup(
    name="pyreidolia",
    version="0.0.2",
    description="cloudy regions segmentation",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Thomas Bury, Afonso Alves, Daniel Staudegger",
    author_email="bury.thomas@gmail.com",
    packages=find_packages(),
    zip_safe=False,  # the package can run out of an .egg file
    install_requires=REQUIREMENTS,
    extras_require=EXTRAS_REQUIRE,
    python_requires=">=3.7",
    license="MIT",
    keywords=KEYWORDS,
    url='https://www.kaggle.com/c/understanding_cloud_organization'
    #package_data={'': ['data/*.png', 'data/*.npz', 'data/*.gz', 'data/*.jpg']},
)
