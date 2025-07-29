# CGRAG Triple Filtering Template

prompt_template = """Given the query: "${query}"

Please evaluate which of the following knowledge triples are relevant to answering this query. 
Return only the numbers of the relevant triples.

Triples:
${triple_strings}

Relevant triple numbers:""" 