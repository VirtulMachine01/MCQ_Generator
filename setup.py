from setuptools import find_packages, setup

setup(
    name = 'mcqgenerator',
    version='0.0.1',
    author='Meet Patel',
    author_email='meetnpatel101112@gmail.com',
    install_requires=["langchain", "streamlit", "python-dotenv", "PyPDF2"],
    packages=find_packages()
)