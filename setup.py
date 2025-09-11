from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="multimodalrag",
    version="0.1.0",
    author="Ali Janloo",
    author_email="mahmoodjanlooali@gmail.com",
    description="A Multi-Modal RAG application for intelligently processing and indexing unstructured medical PDF documents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Alijanloo/MultiModalRag",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)