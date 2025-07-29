# CGRAG LLM Triple Filtering Template

prompt_template = """Given the query: "${query}"

Please carefully analyze which of the following knowledge triples are potentially relevant to answering this query. Consider both direct relevance and indirect connections that might be useful for reasoning.

Knowledge Triples:
${triple_strings}

Instructions:
- Return only the numbers of the relevant triples, separated by commas
- Include triples that mention entities, concepts, or relationships that could help answer the query
- Be generous in inclusion - it's better to include potentially useful triples than to miss important ones

Relevant triple numbers:""" 