from setuptools import setup, find_packages

setup(
    name="robot_world",  
    description="robot world project",
    author="Jameskrw",
    author_email="kangruiwang.cs@gmail.com",
    packages=["robot_world", "octo"],  
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