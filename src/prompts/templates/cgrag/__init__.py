# CGRAG Templates Directory
# Contains all CGRAG-specific prompt templates organized by step

TEMPLATE_MAPPING = {
    'query_triple_extraction': 'query_triple_extraction.py',
    'triple_filtering': 'triple_filtering.py', 
    'llm_triple_filtering': 'llm_triple_filtering.py',
    'entity_merging': 'entity_merging.py',
    'logic_path_formation': 'logic_path_formation.py',
    'final_qa': 'final_qa.py'
}

def load_template(template_name):
    """Load a CGRAG template by name"""
    import os
    import importlib.util
    from string import Template
    
    if template_name not in TEMPLATE_MAPPING:
        raise ValueError(f"Unknown template: {template_name}")
    
    template_file = TEMPLATE_MAPPING[template_name]
    template_path = os.path.join(os.path.dirname(__file__), template_file)
    
    spec = importlib.util.spec_from_file_location(template_name, template_path)
    template_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(template_module)
    
    return Template(template_module.prompt_template) 