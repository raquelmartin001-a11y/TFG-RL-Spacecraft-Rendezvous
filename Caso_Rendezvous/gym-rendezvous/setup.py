from setuptools import setup, find_packages
 
setup(
    name='gym_rendezvous',
    version='2.0.0',
    description='Rendezvous environment compatible with Gymnasium and Stable-Baselines3',
    python_requires='>=3.9',
    packages=find_packages(),
    install_requires=[
        'gymnasium>=0.26.0',
        'stable-baselines3>=2.0.0',
        'numpy>=1.21.0',
        'torch>=1.13.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'matplotlib>=3.5.0',
        ]
    },
)