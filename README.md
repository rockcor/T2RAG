
# T2RAG: Triplet-driven Thinking RAG
Retrieval-augmented generation (RAG) is critical for reducing hallucinations and incorporating external knowledge into Large Language Models (LLMs).
However, advanced RAG systems face a trade-off between performance and efficiency. Multi-round RAG approaches achieve strong reasoning but incur excessive LLM calls and token costs, while Graph RAG methods suffer from computationally expensive, error-prone graph construction and retrieval redundancy. To address these challenges, we propose T2RAG, a novel framework that operates on a simple, graph-free knowledge base of atomic triplets. T2RAG leverages an LLM to decompose questions into searchable triplets with placeholders, which it then iteratively resolves by retrieving evidence from the triplet database. Empirical results show that T2RAG significantly outperforms state-of-the-art multi-round and Graph RAG methods, achieving an average performance gain of up to 11\% across six datasets while reducing retrieval costs by up to 45\%.

## Environment Setup
Assuming you have an environment with torch>=2.0.
```sh
pip install -r requirements.txt
```
you may need manually download some packages based on your torch version.

## API Initialization

```sh
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=<path to Huggingface home directory>
export OPENAI_API_KEY=<your openai api key>   # if you want to use OpenAI model
export GOOGLE_API_KEY=<your gemini api key>
```

## Quick Start

```sh
python main.py --dataset 2wikimultihopqa --method trag
```
## Supporting parameters
```sh
--dataset [popqa, 2wikimultihopqa, musique, hotpotqa, story, medical]
--method [trag, standard, hippo]
```
## Results
Final and intermediate results are saved in `outputs/<dataset>_<method>`.
