from setuptools import setup, find_packages

install_requires = ["numpy",
                    "joblib>=1.5.2",
                    "scipy>=1.7.1",
                    "numba"]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="QdpMC",
    description='A package for pricing OTC options using a vectorized Monte Carlo method.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='derivatives, finance',
    license='Free for non-commercial use',
    author='YieldChain Developers',
    author_email='dev@yieldchain.com',
    url='http://www.yieldchain.com/qdpmc/',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.8',
    install_requires=install_requires,
)
