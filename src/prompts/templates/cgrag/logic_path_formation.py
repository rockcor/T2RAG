# CGRAG Logic Path Formation Template

prompt_template = """Given the query: "${query}"

Analyze the following knowledge triples and construct logical reasoning paths that could help answer the query. Each path should connect related triples in a meaningful sequence.

Knowledge Triples:
${triple_strings}

Instructions:
- Create 3-5 logical reasoning paths using the triple numbers
- Each path should connect 2-4 related triples
- Focus on paths that are most relevant to answering the query
- Explain the reasoning behind each path
- Format: Path X: triple_numbers - explanation

Example format:
Path 1: 1, 3, 5 - These triples show the connection between the person and their achievement
Path 2: 2, 4 - These triples establish the location and time context

Reasoning Paths:""" 