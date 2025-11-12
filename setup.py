from setuptools import setup, find_packages

install_requires = ["numpy",
                    "joblib>=0.17.0",
                    "scipy>=1.7.1",
                    "numba"]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    # these rarely change
    name="QdpMC",
    description='A package for pricing OTC option via Monte Carlo',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='derivatives, finance',
    license='Free for non-commercial use',
    author='Yield Chain Developers',
    author_email='luyudso@gmail.com',
    url='http://yulu0131.github.io/',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.8',
    install_requires=install_requires,
)
