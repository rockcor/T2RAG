import os
from typing import List
import json

from src.HippoRAG import HippoRAG
from src.StandardRAG import StandardRAG
from src.TRAG import TRAG
from src.utils.misc_utils import string_to_bool
from src.utils.config_utils import BaseConfig

import argparse


# os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
import warnings

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="modeling_nvembed")


# os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging

def get_gold_docs(samples: List, dataset_name: str = None) -> List:
    gold_docs = []
    for sample in samples:
        if 'supporting_facts' in sample:  # hotpotqa, 2wikimultihopqa
            gold_title = set([item[0] for item in sample['supporting_facts']])
            gold_title_and_content_list = [item for item in sample['context'] if item[0] in gold_title]
            if dataset_name.startswith('hotpotqa'):
                gold_doc = [item[0] + '\n' + ''.join(item[1]) for item in gold_title_and_content_list]
            else:
                gold_doc = [item[0] + '\n' + ' '.join(item[1]) for item in gold_title_and_content_list]
        elif 'contexts' in sample:
            gold_doc = [item['title'] + '\n' + item['text'] for item in sample['contexts'] if item['is_supporting']]
        else:
            assert 'paragraphs' in sample, "`paragraphs` should be in sample, or consider the setting not to evaluate retrieval"
            gold_paragraphs = []
            for item in sample['paragraphs']:
                if 'is_supporting' in item and item['is_supporting'] is False:
                    continue
                gold_paragraphs.append(item)
            gold_doc = [item['title'] + '\n' + (item['text'] if 'text' in item else item['paragraph_text']) for item in gold_paragraphs]

        gold_doc = list(set(gold_doc))
        gold_docs.append(gold_doc)
    return gold_docs


def get_gold_answers(samples):
    gold_answers = []
    for sample_idx in range(len(samples)):
        gold_ans = None
        sample = samples[sample_idx]

        if 'answer' in sample or 'gold_ans' in sample:
            gold_ans = sample['answer'] if 'answer' in sample else sample['gold_ans']
        elif 'reference' in sample:
            gold_ans = sample['reference']
        elif 'obj' in sample:
            gold_ans = set(
                [sample['obj']] + [sample['possible_answers']] + [sample['o_wiki_title']] + [sample['o_aliases']])
            gold_ans = list(gold_ans)
        assert gold_ans is not None
        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]
        assert isinstance(gold_ans, list)
        gold_ans = set(gold_ans)
        if 'answer_aliases' in sample:
            gold_ans.update(sample['answer_aliases'])

        gold_answers.append(gold_ans)

    return gold_answers

def main():
    parser = argparse.ArgumentParser(description="HippoRAG retrieval and QA")
    parser.add_argument('--dataset', type=str, default='musique', help='Dataset name')
    parser.add_argument('--llm_base_url', type=str, default='https://api.openai.com/v1', help='LLM base URL')
    parser.add_argument('--llm_name', type=str, default='gpt-4o-mini', help='LLM name')
    parser.add_argument('--embedding_name', type=str, default='nvidia/NV-Embed-v2', help='embedding model name')
    parser.add_argument('--api_key', type=str, default='', help='Gemini API key')
    parser.add_argument('--force_index_from_scratch', type=str, default='false',
                        help='If set to True, will ignore all existing storage files and graph data and will rebuild from scratch.')
    parser.add_argument('--force_openie_from_scratch', type=str, default='false', help='If set to False, will try to first reuse openie results for the corpus if they exist.')
    parser.add_argument('--openie_mode', choices=['online', 'offline'], default='online',
                        help="OpenIE mode, offline denotes using VLLM offline batch mode for indexing, while online denotes")
    parser.add_argument('--save_dir', type=str, default='outputs', help='Save directory')
    parser.add_argument('--method', type=str, default='standard', help='Method name (standard, hippo, trag)')
    parser.add_argument('--recall_eval_top_k', type=int, default=200, help='Number of top retrieval results to select for recall evaluation')
    parser.add_argument('--qa_top_k', type=int, default=5, help='Number of top QA results to select for CGRAG reasoning')
    parser.add_argument('--max_qa_steps', type=int, default=3, help='Maximum number of steps for QA')
    args = parser.parse_args()
    base_save_dir = args.save_dir
    dataset_name = args.dataset
    dataset_name_univ = dataset_name + '_univ'
    
    if base_save_dir == 'outputs':
        # Reusable directory for embeddings/OpenIE
        reusable_save_dir = base_save_dir + '/' + dataset_name_univ
        # Method-specific working directory
        method_save_dir = base_save_dir + '/' + dataset_name + '_' + args.method
    else:
        # Reusable directory for embeddings/OpenIE
        reusable_save_dir = base_save_dir + '_' + dataset_name_univ
        # Method-specific working directory
        method_save_dir = base_save_dir + '_' + dataset_name + '_' + args.method

    llm_base_url = args.llm_base_url
    llm_name = args.llm_name
    api_key = args.api_key

    # Set environment variable for Gemini API key
    os.environ["GOOGLE_API_KEY"] = api_key

    corpus_path = f"reproduce/dataset/{dataset_name}_corpus.json"
    with open(corpus_path, "r") as f:
        corpus = json.load(f)

    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]

    force_index_from_scratch = string_to_bool(args.force_index_from_scratch)
    force_openie_from_scratch = string_to_bool(args.force_openie_from_scratch)

    # Prepare datasets and evaluation
    samples = json.load(open(f"reproduce/dataset/{dataset_name}.json", "r"))
    all_queries = [s['question'] for s in samples]

    gold_answers = get_gold_answers(samples)

    try:
        gold_docs = get_gold_docs(samples, dataset_name)
        assert len(all_queries) == len(gold_docs) == len(gold_answers), "Length of queries, gold_docs, and gold_answers should be the same."
    except:
        gold_docs = None

    # Config for method-specific components - uses actual LLM name
    config = BaseConfig(
        save_dir=method_save_dir,  # Use method-specific directory for LLM caches
        llm_base_url=llm_base_url,
        llm_name=llm_name,  # Actual LLM name for method-specific caches
        dataset=dataset_name_univ,
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=force_index_from_scratch,
        force_openie_from_scratch=force_openie_from_scratch,
        rerank_dspy_file_path="src/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        recall_eval_top_k=args.recall_eval_top_k,
        linking_top_k=5,
        max_qa_steps=args.max_qa_steps,
        qa_top_k=args.qa_top_k,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=8,
        max_new_tokens=None,
        corpus_len=len(corpus),
        openie_mode=args.openie_mode,
    )

    # Set up config with new directory structure
    config.save_dir = method_save_dir  # Method-specific working directory
    config.exclude_llm_from_paths = True  
    config.embedding_base_dir = reusable_save_dir  # Universal directory for embeddings/OpenIE
    
    # Add LLM cache directory setup
    llm_cache_dir = method_save_dir + '/' + args.llm_name.replace(':', '-').replace('/', '-') + '_cache'
    config.llm_cache_dir = llm_cache_dir

    logging.basicConfig(level=logging.INFO)

    # Choose between HippoRAG (with KG) and StandardRAG (direct vector search)
    if args.method == 'standard':
        print("Using StandardRAG (direct vector search without knowledge graph)")
        rag_system = StandardRAG(global_config=config)
    elif args.method == 'hippo':
        print("Using HippoRAG (with knowledge graph)")
        rag_system = HippoRAG(global_config=config)
    elif args.method == 'trag':
        print("Using TRAG (with knowledge graph)")
        rag_system = TRAG(global_config=config)
    
    print(f"Using embedding directory: {rag_system.embedding_dir}")
    if hasattr(rag_system, 'openie_results_path'):
        print(f"Using OpenIE results path: {rag_system.openie_results_path}")
    else:
        print("StandardRAG does not use OpenIE")

    rag_system.index(docs)

    # Retrieval and QA
    logging.info(f"Running QA for {len(all_queries)} queries and {len(gold_answers)} gold answers")
    results = rag_system.rag_qa(queries=all_queries, gold_docs=gold_docs, gold_answers=gold_answers)
    
if __name__ == "__main__":
    main() 
