# CGRAG Entity Merging Template

prompt_template = """Analyze the following list of entities and identify groups of entities that refer to the same concept or thing. These might include:
- Different spellings, abbreviations, or variations of the same name
- Synonyms or alternative names for the same entity
- Entities that clearly refer to the same person, place, organization, or concept

Entity List:
${entity_list}

Instructions:
- Group similar entities together under one canonical name
- Choose the most complete or commonly used form as the canonical name
- Output format: canonical_name: variant1, variant2, variant3
- Only include groups with 2 or more entities that are truly synonymous or variations
- Skip entities that don't have clear matches
- Do NOT map unrelated entities, dates, or numbers
- Only output actual entity mappings, no notes or explanations

Entity Mappings:""" 