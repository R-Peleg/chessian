from setuptools import setup, find_packages

setup(
    name='chessian',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'transformers==4.51.1',
        'chess==1.9.3',
        'requests==2.31.0',
    ],
    entry_points={
        'console_scripts': [
            'chessian=chessian.main:main',
        ],
    },
)
