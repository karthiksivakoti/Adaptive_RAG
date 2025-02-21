# risk_rag_system/setup.py

from setuptools import setup, find_packages

setup(
    name="risk_rag_system",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "pydantic>=2.0.0",
        "loguru>=0.7.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        "pytest-timeout>=2.1.0",
        "pytest-cov>=4.1.0",
        "aiohttp>=3.8.0",
        "networkx>=3.0",
        "chromadb>=0.4.0",
        "PyMuPDF>=1.22.0",
        "python-docx>=0.8.11",
        "pytesseract>=0.3.10",
        "Pillow>=10.0.0",
        "openpyxl>=3.1.0"
    ],
    extras_require={
        "dev": [
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.4.0",
            "pytest-mock>=3.11.0",
            "pytest-xdist>=3.3.0",
            "pytest-env>=0.8.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-timeout>=2.1.0",
            "pytest-cov>=4.1.0",
        ],
    },
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="Risk RAG System with RAPTOR-based indexing",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="rag, nlp, risk-analysis, raptor",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)