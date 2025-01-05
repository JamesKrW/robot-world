from setuptools import setup, find_packages

setup(
    name="robot_world",  
    version="0.1.0",
    description="robot world project",
    author="Jameskrw",
    author_email="kangruiwang.cs@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "torchvision",
        "einops",
        "diffusers",
        "timm",
        "av"
    ],
    dependency_links=[
        "https://download.pytorch.org/whl/cu121"  # CUDA 12.1 specific PyTorch wheels
    ]
)