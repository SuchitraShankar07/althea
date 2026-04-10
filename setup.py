"""
setup.py
Package configuration for failure_aware_rag.
"""

from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="failure-aware-rag",
    version="0.1.0",
    description=(
        "Failure-Aware RAG: Hallucination diagnosis at the claim level "
        "with metric-guided QLoRA fine-tuning."
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Failure-Aware RAG Team",
    python_requires=">=3.10",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "rag-demo=scripts.demo:main",
            "rag-infer=scripts.run_inference:main",
            "rag-train=scripts.train:main",
            "rag-eval=scripts.evaluate:main",
            "rag-index=scripts.build_index:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
