import json
import os
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from datetime import datetime
import re
from tqdm import tqdm

from .llm import _get_llm_class, BaseLLM
from .embedding_model import _get_embedding_model_class, BaseEmbeddingModel
from .embedding_store import EmbeddingStore
from .information_extraction import OpenIE
from .utils.config_utils import BaseConfig
from .utils.misc_utils import QuerySolution
from .evaluation.qa_eval import QAExactMatch, QAF1Score

logger = logging.getLogger(__name__)

class TRAG:
    def __init__(self, 
                 global_config=None,
                 save_dir=None,
                 llm_model_name=None,
                 embedding_model_name=None,
                 llm_base_url=None,
                 azure_endpoint=None,
                 azure_embedding_endpoint=None,
                 max_qa_steps: int = 3):
        
        if global_config is None:
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config

        # Overwriting Configuration if Specified
        if save_dir is not None:
            self.global_config.save_dir = save_dir

        if llm_model_name is not None:
            self.global_config.llm_name = llm_model_name

        if embedding_model_name is not None:
            self.global_config.embedding_model_name = embedding_model_name

        if llm_base_url is not None:
            self.global_config.llm_base_url = llm_base_url

        if azure_endpoint is not None:
            self.global_config.azure_endpoint = azure_endpoint

        if azure_embedding_endpoint is not None:
            self.global_config.azure_embedding_endpoint = azure_embedding_endpoint

        self.max_qa_steps = max_qa_steps
        
        # Extract base dataset name from config
        dataset_name = getattr(self.global_config, 'dataset', '2wikimultihopqa')
        self.base_dataset_name = dataset_name.replace('_univ', '') if dataset_name.endswith('_univ') else dataset_name
        
        # Set up directories following the same pattern as other RAG classes
        llm_label = self.global_config.llm_name.replace("/", "_")
        embedding_label = self.global_config.embedding_model_name.replace("/", "_")
        
        # Handle different directory structures
        if getattr(self.global_config, 'exclude_llm_from_paths', False):
            # New structure: working_dir is method-specific, embedding_dir is universal
            self.working_dir = self.global_config.save_dir  # Method-specific directory
            # Embedding dir should be in the universal directory (passed via separate parameter)
            if hasattr(self.global_config, 'embedding_base_dir'):
                self.embedding_dir = os.path.join(self.global_config.embedding_base_dir, embedding_label)
            else:
                self.embedding_dir = os.path.join(self.global_config.save_dir, embedding_label)
        else:
            # Traditional structure
            self.working_dir = os.path.join(self.global_config.save_dir, f"{llm_label}_{embedding_label}")
            self.embedding_dir = self.working_dir

        # TRAG-specific directories
        self.intermediate_results_dir = os.path.join(self.working_dir, "intermediate_results")
        
        # Create directories
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.intermediate_results_dir, exist_ok=True)
        os.makedirs(self.embedding_dir, exist_ok=True)
        
        # Initialize LLM and embedding model
        self.llm_model: BaseLLM = _get_llm_class(self.global_config)
        self.embedding_model: BaseEmbeddingModel = _get_embedding_model_class(
            embedding_model_name=self.global_config.embedding_model_name)(
            global_config=self.global_config,
            embedding_model_name=self.global_config.embedding_model_name)
        
        # Initialize embedding store for chunks (reuse universal embeddings)
        self.chunk_embedding_store = EmbeddingStore(
            self.embedding_model,
            os.path.join(self.embedding_dir, "chunk_embeddings"),
            self.global_config.embedding_batch_size, 
            'chunk'
        )
        
        # Load pre-existing OpenIE results from universal directory
        if hasattr(self.global_config, 'embedding_base_dir'):
            self.openie_results_path = os.path.join(self.global_config.embedding_base_dir, 'openie_results_ner.json')
        else:
            self.openie_results_path = os.path.join(self.embedding_dir, '..', 'openie_results_ner.json')
        
        self.openie_results = None
        
        # New proposition embedding components
        self.proposition_embeddings = None
        self.proposition_to_passage = {}
        self.all_propositions = []
        
        if os.path.exists(self.openie_results_path):
            logger.info(f"Loading OpenIE results from {self.openie_results_path}")
            with open(self.openie_results_path, 'r') as f:
                self.openie_results = json.load(f)
            self._prepare_proposition_embeddings()
        else:
            logger.warning(f"OpenIE results not found at {self.openie_results_path}")
            self.openie_results = {}
        
        logger.info(f"TRAG initialized with max_qa_steps={max_qa_steps}")
        logger.info(f"Working directory: {self.working_dir}")
        logger.info(f"Embedding directory: {self.embedding_dir}")

    def _prepare_proposition_embeddings(self):
        """Prepare proposition embeddings for semantic triple retrieval from OpenIE results"""
        logger.info("Preparing proposition embeddings for semantic triple retrieval")
        
        docs = self.openie_results.get('docs', [])
        logger.info(f"Processing {len(docs)} OpenIE documents for proposition embedding")
        
        # Check if embeddings are already cached
        proposition_embeddings_path = os.path.join(self.embedding_dir, "proposition_embeddings.pkl")
        proposition_mapping_path = os.path.join(self.embedding_dir, "proposition_mapping.json")
        
        # Check cache compatibility by examining the mapping format
        cache_compatible = False
        if os.path.exists(proposition_embeddings_path) and os.path.exists(proposition_mapping_path):
            try:
                with open(proposition_mapping_path, 'r') as f:
                    cached_mapping = json.load(f)
                    # Check if any proposition has the new 'passage_id' format
                    sample_prop = next(iter(cached_mapping.values()), {})
                    if 'passage_id' in sample_prop and 'embedding_idx' not in sample_prop:
                        cache_compatible = True
                        logger.info("Found compatible cached proposition embeddings")
                    else:
                        logger.info("Found old format cached embeddings, will regenerate")
            except Exception as e:
                logger.warning(f"Error checking cache compatibility: {e}")
        
        if cache_compatible:
            logger.info("Loading cached proposition embeddings")
            try:
                with open(proposition_embeddings_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.proposition_embeddings = cache_data['embeddings']
                    self.all_propositions = cache_data['propositions']
                
                with open(proposition_mapping_path, 'r') as f:
                    self.proposition_to_passage = json.load(f)
                
                logger.info(f"Loaded {len(self.all_propositions)} cached proposition embeddings")
                return
            except Exception as e:
                logger.warning(f"Failed to load cached embeddings: {e}, regenerating...")
        
        # Generate new embeddings
        self.all_propositions = []
        self.proposition_to_passage = {}
        
        # Debug: Check first few doc_idx values
        sample_docs = docs[:3]
        for i, doc in enumerate(sample_docs):
            doc_idx = doc.get('idx', '')
            logger.info(f"Sample doc {i}: doc_idx='{doc_idx}', type={type(doc_idx)}")
        
        # Extract all triples from OpenIE results and create propositions
        propositions_to_embed = []
        for doc in docs:
            doc_idx = doc.get('idx', '')
            passage = doc.get('passage', '')
            extracted_triples = doc.get('extracted_triples', [])
            
            # Debug empty doc_idx
            if not doc_idx:
                logger.warning(f"Empty doc_idx found for passage: {passage[:100]}...")
                continue
                
            for triple in extracted_triples:
                if len(triple) == 3:
                    subject, predicate, obj = triple
                    # Create proposition by concatenating subject, predicate, object
                    proposition = f"{subject} {predicate} {obj}"
                    
                    self.all_propositions.append(proposition)
                    propositions_to_embed.append(proposition)
                    
                    # Store mapping to source passage using doc_idx as passage_id
                    self.proposition_to_passage[proposition] = {
                        'passage': passage,
                        'passage_id': str(doc_idx),  # Ensure passage_id is string
                        'triple': triple
                    }
        
        # Debug: Check first few propositions
        sample_props = list(self.proposition_to_passage.keys())[:3]
        for prop in sample_props:
            mapping = self.proposition_to_passage[prop]
            logger.info(f"Sample proposition: '{prop}' -> passage_id: '{mapping['passage_id']}'")
        
        # Embed all propositions
        if propositions_to_embed:
            logger.info(f"Embedding {len(propositions_to_embed)} propositions")
            from .prompts.linking import get_query_instruction
            
            # Use batch encoding for efficiency
            self.proposition_embeddings = self.embedding_model.batch_encode(
                propositions_to_embed,
                instruction=get_query_instruction('query_to_passage'),  # Use same instruction as query
                norm=True
            )
            
            # Cache the embeddings
            try:
                cache_data = {
                    'embeddings': self.proposition_embeddings,
                    'propositions': self.all_propositions
                }
                with open(proposition_embeddings_path, 'wb') as f:
                    pickle.dump(cache_data, f)
                
                with open(proposition_mapping_path, 'w') as f:
                    json.dump(self.proposition_to_passage, f, indent=2)
                
                logger.info(f"Cached proposition embeddings to {proposition_embeddings_path}")
            except Exception as e:
                logger.warning(f"Failed to cache proposition embeddings: {e}")
            
            logger.info(f"Proposition embeddings prepared with {len(self.all_propositions)} propositions")
        else:
            logger.warning("No propositions found in OpenIE results for embedding")
            self.proposition_embeddings = np.array([])

    def index(self, corpus: List[str], **kwargs) -> None:
        """Index the corpus for retrieval"""
        logger.info("Indexing corpus for TRAG...")
        
        # Index documents with chunk embedding store
        self.chunk_embedding_store.insert_strings(corpus)
        
        logger.info("TRAG indexing completed")

    def reason_and_form_triples(self, query: str) -> Dict[str, Any]:
        """Step 1: Reasoning the query and forming triples with ? placeholders"""
        logger.info(f"Step 1: Reasoning query and forming triples: {query}")
        
        prompt = f"""You are tasked with reasoning about a question and extracting the necessary knowledge triples to answer it.

Question: {query}

Instructions:
1. Think step by step about what information is needed to answer this question
2. Form triples in the format: subject | predicate | object
3. Use "?" as placeholder for unknown entities
4. For comparative questions involving multiple entities, use distinct placeholders like ?entityA, ?directorA, ?directorB
5. Extract multiple triples if the question requires complex reasoning

Examples:
- Question: "What is the capital of France?"
  Reasoning: To answer this, I need to know what France's capital is.
  Triple: France | has capital | ?

- Question: "Who directed the movie that won Best Picture in 2020?"
  Reasoning: To answer this, I need to know which movie won Best Picture in 2020, and who directed that movie.
  Triples: ? | won Best Picture | 2020
           ? | is directed by | ?


- Question: "Which film whose director was born first, MovieA or MovieB?"
  Reasoning: To answer this, I need to know the director of each movie, and the birth year of each director to compare them.
  Triples: MovieA | is directed by | ?directorA
           MovieB | is directed by | ?directorB
           ?directorA | was born in | ?
           ?directorB | was born in | ?


Now analyze this question:

Question: {query}

Provide your response in this format:
Reasoning: [Your step-by-step reasoning about what information is needed]

Triples:
[List each triple on a new line in format: subject | predicate | object]

Propositions: [Convert triples to natural language sentences, keeping ? placeholders]
To answer the question, I need to know [list propositions like "A is directed by ?", "? is born in ?"]
"""

        try:
            # Format message for LLM
            formatted_message = [{"role": "user", "content": prompt}]
            response = self.llm_model.infer(formatted_message)
            llm_response = response[0] if isinstance(response, tuple) else response
            
            # Parse the response
            reasoning = ""
            triples = []
            propositions = ""
            
            lines = llm_response.strip().split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('Reasoning:'):
                    current_section = 'reasoning'
                    reasoning = line.replace('Reasoning:', '').strip()
                elif line.startswith('Triples:'):
                    current_section = 'triples'
                elif line.startswith('Propositions:'):
                    current_section = 'propositions'
                    propositions = line.replace('Propositions:', '').strip()
                elif current_section == 'reasoning' and line:
                    reasoning += " " + line
                elif current_section == 'triples' and '|' in line:
                    parts = [part.strip() for part in line.split('|')]
                    if len(parts) >= 3:
                        triple = {
                            'subject': parts[0],
                            'predicate': parts[1],
                            'object': parts[2]
                        }
                        triples.append(triple)
                elif current_section == 'propositions' and line:
                    propositions += " " + line

            # If no triples extracted, retry with simpler format
            if not triples:
                logger.warning("No triples extracted from LLM response, retrying with simpler prompt")
                triples = self._retry_triple_extraction_simple(query)
                if not triples:
                    logger.error(f"Failed to extract meaningful triples from query: {query}")
                    raise RuntimeError(f"Could not extract meaningful triples from query: {query}")
                reasoning = f"Simple extraction identified key relationships in the query."
                propositions = self._triples_to_propositions(triples)

        except Exception as e:
            logger.error(f"Error in reasoning and forming triples: {e}")
            # Try simple extraction as last resort
            try:
                triples = self._retry_triple_extraction_simple(query)
                if not triples:
                    raise RuntimeError(f"Could not extract meaningful triples from query: {query}")
                reasoning = f"Fallback extraction for query analysis."
                propositions = self._triples_to_propositions(triples)
            except:
                logger.error(f"Complete failure to extract triples from query: {query}")
                raise RuntimeError(f"TRAG cannot process query without meaningful triples: {query}")

        # Deduplicate triples before categorization
        unique_triples = []
        seen_triples = set()
        
        for triple in triples:
            # Create a normalized representation for comparison
            triple_key = (
                triple.get('subject', '').strip().lower(),
                triple.get('predicate', '').strip().lower(), 
                triple.get('object', '').strip().lower()
            )
            
            if triple_key not in seen_triples:
                seen_triples.add(triple_key)
                unique_triples.append(triple)
            else:
                logger.debug(f"Removing duplicate triple: {triple}")
        
        logger.info(f"Deduplication: {len(triples)} -> {len(unique_triples)} triples")
        triples = unique_triples

        # Categorize triples by type based on number of ? placeholders
        fuzzy_clues = []      # Type 1: 2+ ? placeholders
        traceable_clues = []  # Type 2: 1 ? placeholder  
        resolved_clues = []   # Type 3: 0 ? placeholders
        
        for triple in triples:
            question_count = sum(value.count('?') for value in triple.values())
            if question_count >= 2:
                fuzzy_clues.append(triple)
            elif question_count == 1:
                traceable_clues.append(triple)
            else:
                resolved_clues.append(triple)

        result = {
            'reasoning': reasoning,
            'all_triples': triples,
            'fuzzy_clues': fuzzy_clues,      # Type 1: 2+ ?
            'traceable_clues': traceable_clues,  # Type 2: 1 ?
            'resolved_clues': resolved_clues,    # Type 3: 0 ?
            'propositions': propositions
        }
        
        logger.info(f"Extracted {len(triples)} total triples: {len(fuzzy_clues)} fuzzy, {len(traceable_clues)} traceable, {len(resolved_clues)} resolved")
        return result

    def retrieve_and_double_check(self, traceable_resolved_clues: List[Dict[str, str]], query: str) -> Dict[str, Any]:
        """Step 2: Retrieve passages using proposition embeddings from OpenIE"""
        logger.info(f"Step 2: Retrieving passages using proposition embeddings for {len(traceable_resolved_clues)} traceable/resolved clues")
        
        # Check if we have proposition embeddings available
        if self.proposition_embeddings is None or len(self.proposition_embeddings) == 0:
            logger.warning("No proposition embeddings available, falling back to empty results")
            return {
                'retrieved_passages': [],
                'retrieved_propositions': [],
                'passage_count': 0,
                'proposition_count': 0
            }
        
        # Step 2.1: Create propositions from query triples
        logger.info("Step 2.1: Creating propositions from query triples")
        query_propositions = []
        for clue in traceable_resolved_clues:
            # Create proposition by concatenating subject, predicate, object
            # Skip placeholders when creating search propositions
            search_parts = []
            if clue.get('subject') and not clue['subject'].startswith('?'):
                search_parts.append(clue['subject'])
            if clue.get('predicate') and not clue['predicate'].startswith('?'):
                search_parts.append(clue['predicate'])
            if clue.get('object') and not clue['object'].startswith('?'):
                search_parts.append(clue['object'])
            
            if search_parts:
                proposition = ' '.join(search_parts)
                query_propositions.append(proposition)
                logger.debug(f"Created query proposition: '{proposition}' from clue: {clue}")
        
        # Add the original query as a proposition too (but filter out question words)
        # Remove question words and ?
        query_filtered = query
        question_words = ['what', 'who', 'where', 'when', 'why', 'how', 'which', 'whose']
        for qword in question_words:
            query_filtered = query_filtered.lower().replace(qword, '').strip()
        query_filtered = query_filtered.replace('?', '').strip()
        
        if query_filtered:
            query_propositions.append(query_filtered)
        
        if not query_propositions:
            logger.warning("No query propositions created from clues")
            return {
                'retrieved_passages': [],
                'retrieved_propositions': [],
                'passage_count': 0,
                'proposition_count': 0
            }
        
        # Step 2.2: Embed query propositions
        logger.info(f"Step 2.2: Embedding {len(query_propositions)} query propositions")
        from .prompts.linking import get_query_instruction
        
        query_proposition_embeddings = self.embedding_model.batch_encode(
            query_propositions,
            instruction=get_query_instruction('query_to_passage'),
            norm=True
        )
        
        # Step 2.3: Find similar propositions using cosine similarity
        logger.info("Step 2.3: Finding similar propositions using cosine similarity")
        
        # Calculate similarities between all query propositions and all stored propositions
        similarities = np.dot(self.proposition_embeddings, query_proposition_embeddings.T)
        
        # Get max similarity for each stored proposition across all query propositions
        max_similarities = np.max(similarities, axis=1)
        
        # Get top propositions by similarity
        top_indices = np.argsort(max_similarities)[::-1]
        
        # Step 2.4: Retrieve passages until we have k unique passages
        target_passage_count = getattr(self.global_config, 'qa_top_k', 5)  # Use qa_top_k instead of retrieval_triplets_top_k
        logger.info(f"Step 2.4: Retrieving propositions until {target_passage_count} unique passages")
        
        retrieved_passages = []
        retrieved_propositions = []
        seen_passage_ids = set()  # Track unique passage IDs (doc_idx)
        
        for idx in top_indices:
            if len(retrieved_passages) >= target_passage_count:
                break
                
            if max_similarities[idx] <= 0:  # Skip propositions with no similarity
                continue
                
            proposition = self.all_propositions[idx]
            proposition_info = self.proposition_to_passage.get(proposition, {})
            
            # Debug: Check proposition mapping
            if not proposition_info:
                logger.warning(f"No mapping found for proposition: '{proposition}'")
                continue
                
            passage_text = proposition_info.get('passage', '')
            passage_id = proposition_info.get('passage_id', 'unknown')
            
            # Debug: Check passage_id
            if passage_id == 'unknown':
                logger.warning(f"Unknown passage_id for proposition: '{proposition}', mapping: {proposition_info}")
            
            # Add proposition info
            retrieved_propositions.append({
                'proposition': proposition,
                'similarity': float(max_similarities[idx]),
                'passage_id': passage_id,
                'passage_text': passage_text
            })
            
            # Add passage if not already seen (based on passage_id/doc_idx)
            if passage_text and passage_id not in seen_passage_ids and passage_id != 'unknown':
                retrieved_passages.append({
                    'text': passage_text,
                    'passage_id': passage_id,
                    'score': float(max_similarities[idx])
                })
                seen_passage_ids.add(passage_id)
        
        logger.info(f"Retrieved {len(retrieved_passages)} unique passages from {len(retrieved_propositions)} propositions")
        
        result = {
            'retrieved_passages': retrieved_passages,
            'retrieved_propositions': retrieved_propositions,
            'passage_count': len(retrieved_passages),
            'proposition_count': len(retrieved_propositions)
        }
        
        logger.info(f"Step 2 completed: Retrieved {len(retrieved_passages)} passages using proposition embeddings")
        return result

    def prepare_retrieval_objects(self):
        """Prepare embeddings and keys for fast retrieval"""
        logger.info("Preparing retrieval objects for TRAG")
        
        # Initialize query to embedding mapping
        self.query_to_embedding: Dict = {'triple': {}, 'passage': {}}
        
        # Prepare passage embeddings similar to StandardRAG (if needed for other purposes)
        if hasattr(self, 'chunk_embedding_store'):
            self.passage_node_keys = list(self.chunk_embedding_store.get_all_ids())
            self.passage_embeddings = np.array(self.chunk_embedding_store.get_embeddings(self.passage_node_keys))
            logger.info(f"Prepared {len(self.passage_node_keys)} passage embeddings")
        else:
            logger.info("Chunk embedding store not available, skipping passage embeddings preparation")

    def get_query_embeddings(self, queries: List[str]):
        """Get embeddings for queries and cache them"""
        from .prompts.linking import get_query_instruction
        
        all_query_strings = []
        for query in queries:
            if query not in self.query_to_embedding.get('passage', {}):
                all_query_strings.append(query)

        if len(all_query_strings) > 0:
            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_passage.")
            all_query_embeddings = self.embedding_model.batch_encode(all_query_strings,
                                                                    instruction=get_query_instruction('query_to_passage'),
                                                                    norm=True)
            for i, query in enumerate(all_query_strings):
                self.query_to_embedding['passage'][query] = all_query_embeddings[i]

    def resolve_clues(self, traceable_clues: List[Dict[str, str]], fuzzy_clues: List[Dict[str, str]], retrieval_results: Dict[str, Any], current_resolved_clues: List[Dict[str, str]], query: str) -> Dict[str, List[Dict[str, str]]]:
        """Step 3: Use all retrieved results to resolve both traceable and fuzzy clues
        
        Returns:
        - resolved_clues: Traceable clues that became fully resolved (1 ? -> 0 ?)
        - newly_traceable_clues: Fuzzy clues that became traceable (2+ ? -> 1 ?)
        - remaining_traceable_clues: Traceable clues that couldn't be resolved this iteration
        - remaining_fuzzy_clues: Fuzzy clues that couldn't be made traceable this iteration
        """
        logger.info(f"Step 3: Resolving {len(traceable_clues)} traceable clues and {len(fuzzy_clues)} fuzzy clues using all retrieved data")
        
        if not retrieval_results['retrieved_passages'] and not retrieval_results['retrieved_propositions']:
            logger.info("No retrieved passages or propositions available for resolution")
            return {
                'resolved_clues': [],
                'newly_traceable_clues': [],
                'remaining_traceable_clues': traceable_clues.copy(),
                'remaining_fuzzy_clues': fuzzy_clues.copy()
            }
        
        # Extract context from retrieved propositions
        context_propositions = "\n".join([
            f"- {proposition.get('proposition', '')}" for proposition in retrieval_results['retrieved_propositions']
        ])
        
        # Also prepare context passages for prompt
        top_k = getattr(self.global_config, 'qa_top_k', 5)
        top_passages = retrieval_results.get('retrieved_passages', [])[:top_k]
        context_passages = "\n".join([f"- {p.get('text', '')}" for p in top_passages])
        
        fuzzy_clues_text = ""
        for i, fuzzy_clue in enumerate(fuzzy_clues):
            fuzzy_clues_text += f"""
            Fuzzy Clue {i+1}:
            Subject: {fuzzy_clue['subject']}
            Predicate: {fuzzy_clue['predicate']}
            Object: {fuzzy_clue['object']}
            """
        resolved_clues_context = ""
        if current_resolved_clues:
            resolved_clues_context = "\n\nPreviously Resolved Clues (use these to replace ? placeholders):\n" + "\n".join([
                f"- {clue.get('subject', '')} | {clue.get('predicate', '')} | {clue.get('object', '')}"
                for clue in current_resolved_clues
            ])
        

        traceable_clues_text = ""
        for i, clue in enumerate(traceable_clues):
            traceable_clues_text += f"""
Traceable Clue {i+1}:
Subject: {clue['subject']}
Predicate: {clue['predicate']}
Object: {clue['object']}
"""
        
        combined_prompt = f"""Example:
        Context Propositions:
        {context_propositions[:1000]}
        
        Fully Resolved Clue 1:
        Subject: Lothair II
        Predicate: has mother
        Object: Ermengarde of Tours
        
        Newly Traceable Clue 1:
        Subject: Ermengarde of Tours
        Predicate: died on
        Object: 20 March 851
        
        ---

        Now apply the same process to the following clues:
        Use the context passages and propositions to resolve any '?' placeholders with as much detail as possible, grounding your answers in the passage content.
        Instructions:
        1. For traceable clues (one '?'), replace '?' with the correct entity to fully resolve it, including any relevant attributes.
        2. For fuzzy clues (multiple '?'), generate a Newly Traceable Clue by replacing one of the placeholders with the correct entity, including any relevant context.

        Original Query: {query}

        Traceable Clues:
        {traceable_clues_text}
        
        Fuzzy Clues:
        {fuzzy_clues_text}

        Context Passages:
        {context_passages}

        Context Propositions:
        {context_propositions}

        {resolved_clues_context}

        Return two lists in this format:
        Fully Resolved Clue 1:
        Subject: ...
        Predicate: ...
        Object: ...

        Fully Resolved Clue 2:
        Subject: ...
        Predicate: ...
        Object: ...

        Newly Traceable Clue 1:
        Subject: ...
        Predicate: ...
        Object: ...

        Newly Traceable Clue 2:
        Subject: ...
        Predicate: ...
        Object: ...

        (Continue numbering accordingly)"""
        
        formatted_message = [{"role": "user", "content": combined_prompt}]
        response = self.llm_model.infer(formatted_message)
        llm_response = response[0] if isinstance(response, tuple) else response
        
        # Parse combined response
        fully_resolved, newly_traceable = self._parse_combined_clue_response(llm_response)
        
        # Compute remaining clues (all original clues processed)
        resolved_clues = fully_resolved.copy()
        newly_traceable_clues = newly_traceable.copy()
        
        # No unresolved clues remain after combined resolution
        remaining_traceable_clues: List[Dict[str, str]] = []
        remaining_fuzzy_clues: List[Dict[str, str]] = []
        
        logger.info(f"Combined resolution: {len(resolved_clues)} fully resolved, {len(newly_traceable_clues)} newly traceable")
        
        # Deduplicate and return
        resolved_clues = self._deduplicate_clues(resolved_clues)
        newly_traceable_clues = self._deduplicate_clues(newly_traceable_clues)
        remaining_traceable_clues = self._deduplicate_clues(remaining_traceable_clues)
        remaining_fuzzy_clues = self._deduplicate_clues(remaining_fuzzy_clues)
        
        return {
            'resolved_clues': resolved_clues,
            'newly_traceable_clues': newly_traceable_clues,
            'remaining_traceable_clues': remaining_traceable_clues,
            'remaining_fuzzy_clues': remaining_fuzzy_clues
        }

    def _parse_triple_response(self, response: str, original_triple: Dict[str, str]) -> Dict[str, str]:
        """Parse LLM response to extract resolved triple"""
        try:
            lines = response.strip().split('\n')
            resolved_triple = {}
            
            for line in lines:
                if line.startswith('Subject:'):
                    resolved_triple['subject'] = line.replace('Subject:', '').strip()
                elif line.startswith('Predicate:'):
                    resolved_triple['predicate'] = line.replace('Predicate:', '').strip()
                elif line.startswith('Object:'):
                    resolved_triple['object'] = line.replace('Object:', '').strip()
            
            # Ensure all keys are present
            for key in ['subject', 'predicate', 'object']:
                if key not in resolved_triple:
                    resolved_triple[key] = original_triple[key]
            
            return resolved_triple
            
        except Exception as e:
            logger.error(f"Error parsing triple response: {e}")
            return original_triple

    def _parse_batch_triple_response(self, response: str, original_triples: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Parse LLM response to extract multiple resolved triples from batch processing"""
        try:
            lines = response.strip().split('\n')
            resolved_triples = []
            current_triple = {}
            current_clue_num = 0
            
            for line in lines:
                line = line.strip()
                
                # Check for clue number marker
                if line.startswith('Clue ') and ':' in line:
                    # Save previous triple if complete
                    if current_triple and len(current_triple) == 3:
                        resolved_triples.append(current_triple)
                    current_triple = {}
                    current_clue_num += 1
                
                # Parse triple components
                elif line.startswith('Subject:'):
                    current_triple['subject'] = line.replace('Subject:', '').strip()
                elif line.startswith('Predicate:'):
                    current_triple['predicate'] = line.replace('Predicate:', '').strip()
                elif line.startswith('Object:'):
                    current_triple['object'] = line.replace('Object:', '').strip()
            
            # Don't forget the last triple
            if current_triple and len(current_triple) == 3:
                resolved_triples.append(current_triple)
            
            # Ensure we have the right number of triples, fill missing with originals
            while len(resolved_triples) < len(original_triples):
                idx = len(resolved_triples)
                logger.warning(f"Missing resolved triple {idx+1}, using original")
                resolved_triples.append(original_triples[idx].copy())
            
            # Ensure all triples have all required keys
            final_triples = []
            for i, resolved_triple in enumerate(resolved_triples[:len(original_triples)]):
                final_triple = {}
                original_triple = original_triples[i]
                
                for key in ['subject', 'predicate', 'object']:
                    final_triple[key] = resolved_triple.get(key, original_triple[key])
                
                final_triples.append(final_triple)
            
            logger.info(f"Parsed {len(final_triples)} triples from batch response")
            return final_triples
            
        except Exception as e:
            logger.error(f"Error parsing batch triple response: {e}")
            # Return original triples as fallback
            return [triple.copy() for triple in original_triples]

    def _parse_expansion_response(self, response: str, original_triple: Dict[str, str]) -> List[Dict[str, str]]:
        """Parse LLM response that might contain single or multiple expanded clues"""
        try:
            lines = response.strip().split('\n')
            expanded_clues = []
            current_clue = {}
            clue_count = 0
            
            for line in lines:
                line = line.strip()
                
                # Check for clue number marker (Clue 1:, Clue 2:, etc.)
                if line.startswith('Clue ') and ':' in line:
                    # Save previous clue if complete
                    if current_clue and len(current_clue) == 3:
                        expanded_clues.append(current_clue)
                    current_clue = {}
                    clue_count += 1
                
                # Parse triple components
                elif line.startswith('Subject:'):
                    current_clue['subject'] = line.replace('Subject:', '').strip()
                elif line.startswith('Predicate:'):
                    current_clue['predicate'] = line.replace('Predicate:', '').strip()
                elif line.startswith('Object:'):
                    current_clue['object'] = line.replace('Object:', '').strip()
            
            # Don't forget the last clue
            if current_clue and len(current_clue) == 3:
                expanded_clues.append(current_clue)
            
            # If no numbered clues found, try parsing as single clue
            if not expanded_clues:
                single_clue = self._parse_triple_response(response, original_triple)
                expanded_clues = [single_clue]
            
            # Ensure all clues have all required keys, fallback to original if missing
            final_clues = []
            for clue in expanded_clues:
                final_clue = {}
                for key in ['subject', 'predicate', 'object']:
                    final_clue[key] = clue.get(key, original_triple[key])
                final_clues.append(final_clue)
            
            logger.info(f"Parsed {len(final_clues)} expanded clues from response")
            return final_clues
            
        except Exception as e:
            logger.error(f"Error parsing expansion response: {e}")
            # Return original triple as fallback
            return [original_triple.copy()]

    def _parse_batch_expansion_response(self, response: str, original_triples: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Parse LLM response that might contain expanded clues with sub-numbering (1a, 1b, etc.)"""
        try:
            lines = response.strip().split('\n')
            all_resolved_clues = []
            current_clue = {}
            
            for line in lines:
                line = line.strip()
                
                # Check for clue number marker (Clue 1:, Clue 2a:, Clue 2b:, etc.)
                if line.startswith('Clue ') and ':' in line:
                    # Save previous clue if complete
                    if current_clue and len(current_clue) == 3:
                        all_resolved_clues.append(current_clue)
                    current_clue = {}
                
                # Parse triple components
                elif line.startswith('Subject:'):
                    current_clue['subject'] = line.replace('Subject:', '').strip()
                elif line.startswith('Predicate:'):
                    current_clue['predicate'] = line.replace('Predicate:', '').strip()
                elif line.startswith('Object:'):
                    current_clue['object'] = line.replace('Object:', '').strip()
            
            # Don't forget the last clue
            if current_clue and len(current_clue) == 3:
                all_resolved_clues.append(current_clue)
            
            # If parsing failed, fallback to original method
            if not all_resolved_clues:
                logger.warning("Batch expansion parsing failed, falling back to standard batch parsing")
                return self._parse_batch_triple_response(response, original_triples)
            
            # Ensure all clues have all required keys
            final_clues = []
            for clue in all_resolved_clues:
                final_clue = {}
                # Use the first original triple as template for missing keys
                template = original_triples[0] if original_triples else {'subject': '', 'predicate': '', 'object': ''}
                
                for key in ['subject', 'predicate', 'object']:
                    final_clue[key] = clue.get(key, template[key])
                
                final_clues.append(final_clue)
            
            logger.info(f"Parsed {len(final_clues)} clues from batch expansion response (may include expansions)")
            return final_clues
            
        except Exception as e:
            logger.error(f"Error parsing batch expansion response: {e}")
            # Return original triples as fallback
            return [triple.copy() for triple in original_triples]

    def _parse_combined_clue_response(self, response: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """Parse the combined LLM output into fully resolved and newly traceable clues."""
        fully_resolved: List[Dict[str, str]] = []
        newly_traceable: List[Dict[str, str]] = []
        current_section = None
        current_clue: Dict[str, str] = {}
        for line in response.splitlines():
            line = line.strip()
            if line.startswith("Fully Resolved Clue"):
                # Save previous clue
                if current_section == "resolved" and len(current_clue) == 3:
                    fully_resolved.append(current_clue)
                elif current_section == "traceable" and len(current_clue) == 3:
                    newly_traceable.append(current_clue)
                current_section = "resolved"
                current_clue = {}
            elif line.startswith("Newly Traceable Clue"):
                if current_section == "resolved" and len(current_clue) == 3:
                    fully_resolved.append(current_clue)
                elif current_section == "traceable" and len(current_clue) == 3:
                    newly_traceable.append(current_clue)
                current_section = "traceable"
                current_clue = {}
            elif line.startswith("Subject:"):
                current_clue["subject"] = line.split("Subject:", 1)[1].strip()
            elif line.startswith("Predicate:"):
                current_clue["predicate"] = line.split("Predicate:", 1)[1].strip()
            elif line.startswith("Object:"):
                current_clue["object"] = line.split("Object:", 1)[1].strip()
        # Append last clue
        if current_section == "resolved" and len(current_clue) == 3:
            fully_resolved.append(current_clue)
        elif current_section == "traceable" and len(current_clue) == 3:
            newly_traceable.append(current_clue)
        return fully_resolved, newly_traceable

    def _retry_triple_extraction_simple(self, query: str) -> List[Dict[str, str]]:
        """Retry triple extraction with a simpler, more direct prompt"""
        logger.info("Retrying triple extraction with simplified prompt")
        
        simple_prompt = f"""Extract knowledge triples from this question. Use "?" for unknown information.

Question: {query}

Extract 1-3 triples that represent the knowledge needed to answer this question.
Format: subject | predicate | object

Examples:
- "When did Lothair II's mother die?" -> "Lothair II | has mother | ?" and "? | died in | ?"
- "What nationality is the director of film Blood Brothers?" -> "film Blood Brothers | directed by | ?" and "? | has nationality | ?"

Triples:"""

        try:
            # Format message for LLM
            formatted_message = [{"role": "user", "content": simple_prompt}]
            response = self.llm_model.infer(formatted_message)
            llm_response = response[0] if isinstance(response, tuple) else response
            
            # Parse the response - simpler parsing
            triples = []
            lines = llm_response.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if '|' in line and not line.startswith('Question:') and not line.startswith('Examples:'):
                    # Remove any numbering or leading characters
                    if '. ' in line:
                        line = line.split('. ', 1)[-1]
                    if '- ' in line:
                        line = line.replace('- ', '')
                    
                    parts = [part.strip().strip('"').strip("'") for part in line.split('|')]
                    if len(parts) >= 3:
                        triple = {
                            'subject': parts[0],
                            'predicate': parts[1], 
                            'object': parts[2]
                        }
                        # Only add meaningful triples (not generic ones)
                        if not (parts[0] == '?' and 'related' in parts[1].lower() and len(parts[2]) > 30):
                            triples.append(triple)
            
            logger.info(f"Simple extraction found {len(triples)} triples: {triples}")
            return triples[:3]  # Limit to 3 triples max
            
        except Exception as e:
            logger.error(f"Simple triple extraction also failed: {e}")
            return []

    def _triples_to_propositions(self, triples: List[Dict[str, str]]) -> str:
        """Convert triples to natural language propositions"""
        propositions = []
        for triple in triples:
            prop = f"{triple['subject']} {triple['predicate']} {triple['object']}"
            propositions.append(prop)
        
        return f"To answer the question, I need to know: {', '.join(propositions)}"

    def _deduplicate_clues(self, clues: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Remove duplicate clues based on normalized subject, predicate, object"""
        unique_clues = []
        seen_clues = set()
        
        for clue in clues:
            # Create normalized key for comparison
            clue_key = (
                clue.get('subject', '').strip().lower(),
                clue.get('predicate', '').strip().lower(),
                clue.get('object', '').strip().lower()
            )
            
            if clue_key not in seen_clues:
                seen_clues.add(clue_key)
                unique_clues.append(clue)
            else:
                logger.debug(f"Removing duplicate clue: {clue}")
        
        return unique_clues

    def has_unresolved_clues(self, fuzzy_clues: List[Dict[str, str]], traceable_clues: List[Dict[str, str]]) -> bool:
        """Check if there are any unresolved clues (fuzzy with 2+ ? OR traceable with 1 ?)"""
        # Check for fuzzy clues (2+ ?)
        for clue in fuzzy_clues:
            question_count = sum(value.count('?') for value in clue.values())
            if question_count >= 2:
                return True
        
        # Check for traceable clues (1 ?)
        for clue in traceable_clues:
            question_count = sum(value.count('?') for value in clue.values())
            if question_count == 1:
                return True
        
        return False

    def final_qa(self, query: str, all_traceable_resolved_clues: List[Dict[str, str]], all_retrieval_results: List[Dict[str, Any]]) -> str:
        """Step 4: Final QA using only resolved reasoning clues"""
        logger.info("Step 4: Final QA generation using resolved reasoning clues only")
        
        # Format resolved clues for context
        clue_context = "\n".join([
            f"Clue {i+1}: {clue.get('subject', '')} | {clue.get('predicate', '')} | {clue.get('object', '')}"
            for i, clue in enumerate(all_traceable_resolved_clues)  # Show top 5 clues
        ])
        
        # Create final QA prompt
        final_prompt = f"""Based on the reasoning clues, please answer the following question.

Question: {query}

Key Reasoning Clues:
{clue_context}

Instructions:
1. Analyze the question step by step
2. Use the reasoning clues to understand what information is needed
3. Provide ONLY a concise answer

Answer format requirements:
- For WH questions (who/what/where/when): Provide the exact entity, date, full name, or full place name only
- For yes/no questions: Answer only "yes" or "no"
- No explanations, reasoning, or additional text
- One entity or fact only

Answer:"""
        
        try:
            # Format message for LLM
            formatted_message = [{"role": "user", "content": final_prompt}]
            response = self.llm_model.infer(formatted_message)
            answer = response[0] if isinstance(response, tuple) else response
            
            # Clean up the answer
            if "Answer:" in answer:
                answer = answer.split("Answer:")[-1].strip()
            
            logger.info("Final answer generated successfully using resolved clues")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating final answer: {e}")
            return "Error generating answer"

    def save_iteration_results(self, query_id: str, query: str, iteration_data: Dict[str, Any]) -> None:
        """Save all iteration results to a single JSON file per question"""
        results_file = os.path.join(self.intermediate_results_dir, f"{query_id}_results.json")
        
        # Load existing results if file exists
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                all_results = json.load(f)
        else:
            all_results = {
                'query_id': query_id,
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'step_1_reasoning': None,
                'iterations': [],
                'final_results': None
            }
        
        # Handle different types of data
        if 'step' in iteration_data:
            if iteration_data['step'] == 1:
                # Step 1: Query reasoning and triple formation
                all_results['step_1_reasoning'] = iteration_data
            elif iteration_data['step'] == 'final':
                # Final results
                all_results['final_results'] = iteration_data
            else:
                # Regular iteration
                all_results['iterations'].append(iteration_data)
        else:
            # Regular iteration (backward compatibility)
            all_results['iterations'].append(iteration_data)
        
        # Save updated results
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"Saved results to {results_file}")

    def trag_qa(self, query: str, query_id: str = None, **kwargs) -> Dict[str, Any]:
        """Main TRAG QA pipeline - Retrieval-Augmented Generation with iterative clue resolution"""
        if query_id is None:
            query_id = f"query_{hash(query) % 10000}"
        
        logger.info(f"Starting TRAG QA for query: {query} (ID: {query_id})")
        
        # Step 1: Reasoning and forming triples with ? placeholders
        logger.info("=" * 50)
        logger.info("STEP 1: Reasoning and Forming Triples")
        logger.info("=" * 50)
        
        reasoning_result = self.reason_and_form_triples(query)
        
        # Save Step 1 results
        step1_data = {
            'step': 1,
            'step_name': 'reasoning_and_triple_formation',
            'input_query': query,
            'reasoning': reasoning_result['reasoning'],
            'all_triples': reasoning_result['all_triples'],
            'fuzzy_clues': reasoning_result['fuzzy_clues'],
            'traceable_clues': reasoning_result['traceable_clues'], 
            'resolved_clues': reasoning_result['resolved_clues'],
            'propositions': reasoning_result['propositions']
        }
        self.save_iteration_results(query_id, query, step1_data)
        
        # Initialize tracking variables
        # Start with traceable and resolved clues for retrieval
        current_traceable_clues = reasoning_result['traceable_clues'].copy()
        current_resolved_clues = reasoning_result['resolved_clues'].copy()
        current_fuzzy_clues = reasoning_result['fuzzy_clues'].copy()
        all_double_checked_results = []
        iteration = 0
        
        # Main iteration loop - continue until all clues are fully resolved (0 ?) OR max QA steps reached
        while iteration < self.max_qa_steps and self.has_unresolved_clues(current_fuzzy_clues, current_traceable_clues):
            iteration += 1
            logger.info("=" * 50)
            logger.info(f"ITERATION {iteration}")
            logger.info("=" * 50)
            
            # Step 2: Retrieve passages using proposition embeddings from OpenIE
            logger.info(f"STEP 2: Retrieve Passages Using Proposition Embeddings (Iteration {iteration})")
            logger.info("-" * 30)
            
            # Save input state before retrieval for accurate logging
            input_traceable_clues = current_traceable_clues.copy()
            input_resolved_clues = current_resolved_clues.copy()
            input_fuzzy_clues = current_fuzzy_clues.copy()

            # Only retrieve by traceable clues
            retrieval_results = self.retrieve_and_double_check(input_traceable_clues, query)
            all_double_checked_results.append(retrieval_results)
            
            # Step 3: Resolve clues using retrieved results
            logger.info(f"STEP 3: Resolve Clues (Iteration {iteration})")
            logger.info("-" * 30)
            
            clues_result = self.resolve_clues(current_traceable_clues, current_fuzzy_clues, retrieval_results, current_resolved_clues, query)
            
            # Update tracking variables for next iteration
            # Resolved clues stay resolved (0 ?)
            current_resolved_clues.extend(clues_result['resolved_clues'])

            # Newly traceable clues become the new traceable clues for next iteration
            current_traceable_clues = clues_result['newly_traceable_clues'] + clues_result['remaining_traceable_clues']

            # --- ENFORCE: Only clues with exactly one '?' are traceable, move resolved to resolved_clues ---
            still_traceable = []
            for clue in current_traceable_clues:
                q_count = sum(v.count('?') for v in clue.values())
                if q_count == 1:
                    still_traceable.append(clue)
                elif q_count == 0:
                    current_resolved_clues.append(clue)
            current_traceable_clues = still_traceable

            # Remaining fuzzy clues continue to next iteration
            current_fuzzy_clues = clues_result['remaining_fuzzy_clues']
            
            # Log iteration status
            logger.info(f"Iteration {iteration} results:")
            logger.info(f"  - Fuzzy clues remaining: {len(current_fuzzy_clues)}")
            logger.info(f"  - Traceable clues remaining: {len(current_traceable_clues)}")
            logger.info(f"  - Newly resolved (0 ?): {len(clues_result['resolved_clues'])}")
            logger.info(f"  - Newly traceable (1 ?): {len(clues_result['newly_traceable_clues'])}")
            logger.info(f"  - Total resolved clues: {len(current_resolved_clues)}")
            
            # Prepare iteration data for logging
            iteration_data = {
                'iteration': iteration,
                'step_2_retrieval_and_expansion': {
                    'input_traceable_clues': input_traceable_clues,  # Use saved input state
                    'input_resolved_clues': input_resolved_clues,   # Use saved input state
                    'input_clues_count': len(input_traceable_clues),
                    'retrieved_passages': retrieval_results['retrieved_passages'],
                    'retrieved_propositions': retrieval_results['retrieved_propositions'],
                    'retrieved_count': len(retrieval_results['retrieved_passages']),
                    'proposition_count': len(retrieval_results['retrieved_propositions'])
                },
                'step_3_clue_resolution': {
                    'input_traceable_clues': input_traceable_clues,     # Use saved input state
                    'input_traceable_clues_count': len(input_traceable_clues),
                    'input_fuzzy_clues': input_fuzzy_clues,            # Use saved input state
                    'input_fuzzy_clues_count': len(input_fuzzy_clues),
                    'resolved_clues': clues_result['resolved_clues'],
                    'resolved_count': len(clues_result['resolved_clues']),
                    'newly_traceable_clues': clues_result['newly_traceable_clues'],
                    'newly_traceable_count': len(clues_result['newly_traceable_clues']),
                    'remaining_traceable_clues': clues_result['remaining_traceable_clues'],
                    'remaining_traceable_count': len(clues_result['remaining_traceable_clues']),
                    'remaining_fuzzy_clues': clues_result['remaining_fuzzy_clues'],
                    'remaining_fuzzy_count': len(clues_result['remaining_fuzzy_clues']),
                    'total_resolved_accumulated': len(current_resolved_clues),
                    'total_traceable_accumulated': len(current_traceable_clues)
                },
                'iteration_summary': {
                    'has_remaining_unresolved_clues': self.has_unresolved_clues(current_fuzzy_clues, current_traceable_clues),
                    'has_remaining_fuzzy_clues': len(current_fuzzy_clues) > 0,
                    'has_remaining_traceable_clues': len(current_traceable_clues) > 0,
                    'will_continue': self.has_unresolved_clues(current_fuzzy_clues, current_traceable_clues) and iteration < self.max_qa_steps,
                    'termination_reason': 'max_qa_steps' if iteration >= self.max_qa_steps else 'all_resolved' if not self.has_unresolved_clues(current_fuzzy_clues, current_traceable_clues) else 'continuing'
                }
            }
            
            # Save iteration results
            self.save_iteration_results(query_id, query, iteration_data)
            
            # Break if all clues are resolved (no fuzzy or traceable clues remain)
            if not self.has_unresolved_clues(current_fuzzy_clues, current_traceable_clues):
                logger.info("✅ All clues fully resolved!")
                break
        
        # Check termination reason
        if iteration >= self.max_qa_steps and self.has_unresolved_clues(current_fuzzy_clues, current_traceable_clues):
            logger.info(f"⚠️ Reached max QA steps ({self.max_qa_steps}) with unresolved clues remaining")
            logger.info(f"   - Fuzzy clues (2+ ?): {len(current_fuzzy_clues)}")
            logger.info(f"   - Traceable clues (1 ?): {len(current_traceable_clues)}")
        
        # Deduplicate resolved clues before final QA to remove duplicates
        current_resolved_clues = self._deduplicate_clues(current_resolved_clues)

        # Step 4: Final QA
        logger.info("=" * 50)
        logger.info("STEP 4: Final QA")
        logger.info("=" * 50)
        
        # Combine all resolved and traceable clues for final QA
        all_final_clues = current_resolved_clues + current_traceable_clues
        final_answer = self.final_qa(query, all_final_clues, all_double_checked_results)
        
        # Save final results
        final_data = {
            'step': 'final',
            'step_name': 'final_qa',
            'total_iterations': iteration,
            'all_resolved_clues': current_resolved_clues,
            'resolved_clues_count': len(current_resolved_clues),
            'all_traceable_clues': current_traceable_clues,
            'traceable_clues_count': len(current_traceable_clues),
            'all_final_clues': all_final_clues,
            'all_clues_count': len(all_final_clues),
            'remaining_fuzzy_clues': current_fuzzy_clues,
            'remaining_fuzzy_count': len(current_fuzzy_clues),
            'total_retrieval_results': len(all_double_checked_results),
            'final_answer': final_answer,
            'completion_status': 'all_resolved' if not self.has_unresolved_clues(current_fuzzy_clues, current_traceable_clues) else 'max_qa_steps_reached'
        }
        
        self.save_iteration_results(query_id, query, final_data)
        
        completion_status = "all clues fully resolved" if not self.has_unresolved_clues(current_fuzzy_clues, current_traceable_clues) else f"max iterations reached with unresolved clues remaining (fuzzy: {len(current_fuzzy_clues)}, traceable: {len(current_traceable_clues)})"
        logger.info(f"TRAG QA completed after {iteration} iterations with {len(all_final_clues)} total final clues - {completion_status}")
        
        return {
            'answer': final_answer,
            'all_resolved_clues': current_resolved_clues,
            'all_traceable_clues': current_traceable_clues,
            'all_final_clues': all_final_clues,
            'all_retrieval_results': all_double_checked_results,
            'total_iterations': iteration,
            'remaining_fuzzy_clues': current_fuzzy_clues,
            'completion_status': completion_status
        }

    def _clues_similar(self, clue1: Dict[str, str], clue2: Dict[str, str]) -> bool:
        """Check if two clues are similar (one might be a resolved version of the other)"""
        # Simple similarity check - at least 2 out of 3 components match or are compatible
        matches = 0
        for key in ['subject', 'predicate', 'object']:
            val1, val2 = clue1.get(key, ''), clue2.get(key, '')
            if val1 == val2:
                matches += 1
            elif val1 == '?' or val2 == '?':
                matches += 0.5  # Partial match for ? replacement
            elif val1.lower() in val2.lower() or val2.lower() in val1.lower():
                matches += 0.5  # Partial match for similar terms
        
        return matches >= 1.5  # At least 1.5 matches out of 3

    def rag_qa(self, 
            queries: List[str],
            gold_docs: List[List[str]] = None,
            gold_answers: List[List[str]] = None) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        """RAG QA method compatible with main_gemini.py interface"""
        logger.info(f"TRAG RAG QA started for {len(queries)} queries")
        
        # # Initialize evaluators if needed
        # if gold_answers is not None:
        #     qa_em_evaluator = QAExactMatch(global_config=self.global_config)
        #     qa_f1_evaluator = QAF1Score(global_config=self.global_config)
        
        query_solutions = []
        all_response_messages = []
        all_metadata = []
        
        for i, query in enumerate(queries):
            query_id = f"query_{i}"
            
            try:
                # Run TRAG QA for this query
                result = self.trag_qa(query, query_id=query_id)
                
                # Collect documents from all retrieved results (not just double-checked)
                retrieved_docs = []
                if result['all_retrieval_results']:
                    for retrieval_result in result['all_retrieval_results']:
                        for passage in retrieval_result.get('retrieved_passages', []):
                            if passage.get('text') and passage['text'] not in retrieved_docs:
                                retrieved_docs.append(passage['text'])
                
                # Create QuerySolution object 
                query_solution = QuerySolution(
                    question=query,
                    docs=retrieved_docs,
                    doc_scores=[1.0] * len(retrieved_docs),  # Placeholder scores
                    answer=result['answer']
                )
                
                # Add gold information if available
                if gold_answers is not None and i < len(gold_answers):
                    query_solution.gold_answers = list(gold_answers[i])
                if gold_docs is not None and i < len(gold_docs):
                    query_solution.gold_docs = gold_docs[i]
                
                query_solutions.append(query_solution)
                
                # Extract answer for response messages
                all_response_messages.append(result['answer'])
                
                # Create metadata
                metadata = {
                    'query_id': query_id,
                    'total_iterations': result['total_iterations'],
                    'num_traceable_resolved_clues': len(result['all_final_clues']),
                    'num_retrieval_results': len(result['all_retrieval_results']),
                    'method': 'TRAG'
                }
                all_metadata.append(metadata)
                
            except Exception as e:
                logger.error(f"Error processing query {i}: {e}")
                # Add empty results for failed queries
                query_solution = QuerySolution(
                    question=query,
                    docs=[],
                    doc_scores=[],
                    answer='Error processing query'
                )
                if gold_answers is not None and i < len(gold_answers):
                    query_solution.gold_answers = list(gold_answers[i])
                if gold_docs is not None and i < len(gold_docs):
                    query_solution.gold_docs = gold_docs[i]
                    
                query_solutions.append(query_solution)
                all_response_messages.append('Error processing query')
                all_metadata.append({'query_id': query_id, 'error': str(e), 'method': 'TRAG'})
        
        logger.info(f"TRAG RAG QA completed for {len(queries)} queries")
        
        # Evaluate QA if gold answers provided
        if gold_answers is not None:
 # Use the centralized QA evaluation function
            from .evaluation.qa_eval import calculate_overall_qa_metrics
            
            # Calculate QA scores using the centralized function
            overall_qa_results = calculate_overall_qa_metrics(
                gold_answers=gold_answers,
                predicted_answers=[qa_result.answer for qa_result in query_solutions],
                aggregation_fn=np.max
            )
            
            logger.info(f"Evaluation results for StandardRAG QA: {overall_qa_results}")

            # Save retrieval and QA results
            for idx, q in enumerate(query_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            return query_solutions, all_response_messages, all_metadata, overall_qa_results
        else:
            return query_solutions, all_response_messages, all_metadata

    def clear_proposition_cache(self):
        """Clear proposition embedding cache to force regeneration"""
        proposition_embeddings_path = os.path.join(self.embedding_dir, "proposition_embeddings.pkl")
        proposition_mapping_path = os.path.join(self.embedding_dir, "proposition_mapping.json")
        
        for cache_file in [proposition_embeddings_path, proposition_mapping_path]:
            if os.path.exists(cache_file):
                os.remove(cache_file)
                logger.info(f"Removed cache file: {cache_file}")
        
        logger.info("Proposition embedding cache cleared") 