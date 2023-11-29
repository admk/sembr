from setuptools import find_packages, setup


setup(
    name="sembr",
    version="0.1",
    description="A semantic linebreaker",
    author="Xitong Gao",
    author_email="",
    platforms=["any"],
    url="http://github.com/admk/sembr",
    packages=find_packages(),
    entry_points = {
        'console_scripts': [
            'sembr=sembr.cli:main'
        ],
    },
    install_requires=[
        'torch',
        'datasets',
        'flask',
        'transformers',
    ],
)
