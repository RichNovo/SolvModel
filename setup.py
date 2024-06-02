from setuptools import setup, find_packages

setup(
    name="atom_model",
    version="0.1",
    packages=['atom_model'],
    install_requires=[
        # Add your dependencies here
        # 'some_package',
    ],
    entry_points={
        'console_scripts': [
            # Define any command line scripts here
            # 'my_command=my_package.module:function',
        ],
    },
)