import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="hipporag-nyabot",
    version="2.0.0-alpha.2",
    author="Bernal Jimenez Gutierrez",
    author_email="jimenezgutierrez.1@osu.edu",
    description="A powerful graph-based RAG framework that enables LLMs to identify and leverage connections within new knowledge for improved retrieval.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/OSU-NLP-Group/HippoRAG",
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    python_requires=">=3.10",
    install_requires=[
        "openai==1.78.1",
        "vllm==0.8.5.post1",
        "gritlm==1.0.2",
        "torch==2.6.0",
        "transformers==4.51.3",
        "networkx==3.4.2",
        "pydantic==2.10.4",
        "python_igraph==0.11.8",
        "tenacity==9.1.2",
        "tiktoken==0.9.0",
        "einops", # No version specified
        "tqdm", # No version specified
    ]
)