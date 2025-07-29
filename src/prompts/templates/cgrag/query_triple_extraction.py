# CGRAG Query Triple Extraction Template

prompt_template = """Given the query: "${query}"

Extract the key knowledge triples that would be needed to answer this query. 
A knowledge triple consists of [subject, relation, object] format.

Instructions:
- Extract at most 8 triples from the query
- Use "?" for unknown/missing parts that need to be found
- Focus on the main entities in the query. Complete the triples with the missing parts by [reasoning concepts].
- Using diverse synonyms to help the following search.
- Format each triple EXACTLY as: subject , relation , object (with spaces around commas)
- Number each triple (1., 2., 3., 4., 5., 6., 7., 8.)

Examples:
Query: "What did Einstein discover in 1905?"
Triples: 
1. Einstein , discovered , [discovery]
2. Einstein , proposed , [theory]
3. Einstein , was famous for , [paper name]
4. Einstein , worked_on , [project]
5. [discovery] , happened_in , 1905

Query: "Who died first, Aivar Kuusmaa or Andy Summers?"
Triples:
1. Andy Summers , died_in , [date]
2. Andy Summers , passed_away_in , [location]
3. Aivar Kuusmaa , died_in , [time]
4. Aivar Kuusmaa , passed_away_in , [country]
5. Andy Summers , got fatal disease in , [year]


Now extract triples for the given query (use EXACT format with spaces around commas):

Triples:""" 