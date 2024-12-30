import os
from llama_index.core.prompts import PromptTemplate

class PromptTemplateCache:

    def __init__(self, directory):
        self.directory = directory
        self.cache = {}
        self._load_templates()

    def _load_templates(self):
        for filename in os.listdir(self.directory):
            if filename.endswith('.prompt'):
                filepath = os.path.join(self.directory, filename)
                with open(filepath, 'r') as file:
                    template_name = os.path.splitext(filename)[0]
                    template_content = file.read()
                    self.cache[template_name] = PromptTemplate(template=template_content)

    def get_template(self, template_name) -> PromptTemplate:
        template = self.cache.get(template_name)
        if template is None:
            print(self.cache)
            raise ValueError(f"Template '{template_name}' not found")
        return self.cache.get(template_name)
    
    @staticmethod
    def get_cache_instance():        
        if _cache_instance is None:
            _cache_instance = PromptTemplateCache('assistamt/prompts/templates')
        return _cache_instance