from setuptools import setup, find_packages

setup(
    name="robot_world",  # package name with hyphen
    version="0.1.0",
    description="A brief description of your robot world project",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.7",
)