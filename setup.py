from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="robot_world",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A brief description of your robot world project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/robot_world",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        # Add your project dependencies here, for example:
        # "numpy>=1.20.0",
        # "pandas>=1.3.0",
    ],
)