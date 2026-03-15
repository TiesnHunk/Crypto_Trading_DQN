"""
Setup script - tạo để cài đặt dễ dàng hơn
"""

from setuptools import setup, find_packages

setup(
    name='crypto-trading-rl',
    version='1.0.0',
    description='Cryptocurrency Trading System with Q-Learning',
    author='Your Name',
    packages=find_packages(),
    install_requires=[
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'python-binance>=1.0.19',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'scikit-learn>=1.3.0',
        'tqdm>=4.65.0',
    ],
    python_requires='>=3.8',
)

