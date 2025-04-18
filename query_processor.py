import json
import logging
import re
from typing import Dict, List, Any, Tuple, Set, Optional, Union
import networkx as nx
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

class QueryProcessor:
    """
    Processes natural language queries about Excel data.
    """
    def __init__(self, llm_agent_interface):
        """
        Initialize the query processor.
        
        Args:
            llm_agent_interface: LLMAgentInterface instance
        """
        self.agent_interface = llm_agent_interface
        self.kg = llm_agent_interface.kg
        
        # Query templates for matching
        self.query_templates = {
            'metadata': [
                r'what sheets (are|does) (in|this) (workbook|excel|file) (have|contain)',
                r'how many (sheets|tables|named ranges) (are|does) (in|this) (workbook|excel|file) (have|contain)',
                r'show me (the|all) (metadata|information) (about|for) (this|the) (workbook|excel|file)',
                r'what is the structure of (this|the) (workbook|excel|file)',
                r'tell me about (this|the) (workbook|excel|file)'
            ],
            'content': [
                r'what (is|does) (the value|cell) ([A-Z]+[0-9]+) (in|on) (sheet|tab) (\w+) (contain|have|show)',
                r'what (is|does) (in|the value of) (cell|the cell) ([A-Z]+[0-9]+)',
                r'show me (the|all) (data|content|values) (in|from) (sheet|tab) (\w+)',
                r'what (does|is in) (sheet|tab) (\w+)',
                r'(give|show) me (the|a) (summary|overview) of (the|all) (data|content)'
            ],
            'structure': [
                r'what (tables|ranges|named ranges) (are|exist) (in|within) (this|the) (workbook|excel|file)',
                r'(are|is) there (any|a) (hierarchy|hierarchies|structure) (in|within) (the|this) data',
                r'show me (the|all) (tables|structure) (in|of) (sheet|tab) (\w+)',
                r'how (is|are) (the|this) (data|sheet|workbook) (structured|organized)',
                r'what (format|structure|layout) (does|is) (this|the) (workbook|excel|file) (have|using)'
            ],
            'formula': [
                r'what (formulas|calculations) (are|is) (used|present) (in|within) (this|the) (workbook|excel|file)',
                r'show me (the|all) (formulas|calculations) (in|from) (sheet|tab) (\w+)',
                r'what (does|is) (the|cell) ([A-Z]+[0-9]+) (calculate|compute)',
                r'what (dependencies|cells) (does|is) (the formula in|cell) ([A-Z]+[0-9]+) (use|depend on|reference)',
                r'(are|is) there (any|complex) (formulas|calculations) (in|within) (this|the) (workbook|excel|file)'
            ],
            'semantic': [
                r'what (does|is) (this|the) (data|workbook|excel|file) (about|represent|mean)',
                r'what (entities|concepts|relationships) (are|exist) (in|within) (this|the) (data|workbook)',
                r'what (is|appears to be) (the|this) (domain|subject|topic) of (this|the) (data|workbook)',
                r'how (are|is) (the|these|this) (entities|data elements) (related|connected)',
                r'what (meaning|semantics|insights) (can you extract|are there) (from|in) (this|the) (data|workbook)'
            ]
        }
        
        # Intent patterns for more specific intents
        self.intent_patterns = {
            'find_value': r'(what is|show me|get|find) (the value|the cell value|cell) ([A-Z]+[0-9]+)',
            'list_sheets': r'(what|which|list|show me) (all|the) sheets',
            'sheet_summary': r'(summarize|summary of|overview of) (sheet|tab) (\w+)',
            'formula_dependencies': r'(what|which) (cells|values) (does|do) (the formula in|cell) ([A-Z]+[0-9]+) (depend on|reference|use)',
            'count_tables': r'how many (tables|ranges) (are|exist)',
            'find_column': r'(find|show me|get|where is) (the|a) column (named|called|with heading) ["\']?([^"\']+)["\']?',
            'column_values': r'(what|list|show me) (values|all values) (in|from) (column|the column) ["\']?([^"\']+)["\']? (in|from) (sheet|tab) (\w+)',
            'semantic_type': r'what (type|kind) of data is (in|contained in) (column|the column) ["\']?([^"\']+)["\']?'
        }
    
    def process_query(self, query_text: str) -> Dict[str, Any]:
        """
        Process a natural language query and extract relevant information.
        
        Args:
            query_text: Natural language query
            
        Returns:
            Dict[str, Any]: Query results
        """
        # Match query against templates to determine category
        query_type = self._match_query_type(query_text)
        
        # Extract parameters based on query type
        params = self._extract_parameters(query_text, query_type)
        
        # Determine specific intent
        intent = self._determine_intent(query_text)
        params['intent'] = intent
        
        # Process the query through the agent interface
        response = self.agent_interface.query(query_text)
        
        # Enhance response with extracted parameters and intent
        if 'extracted_params' not in response:
            response['extracted_params'] = params
        
        if 'intent' not in response:
            response['intent'] = intent
        
        # Generate a natural language answer
        nl_answer = self._generate_nl_answer(response, query_text, intent)
        response['nl_answer'] = nl_answer
        
        return response
    
    def _match_query_type(self, query_text: str) -> str:
        """
        Match the query text against templates to determine its type.
        
        Args:
            query_text: Natural language query
            
        Returns:
            str: Query type
        """
        query_text = query_text.lower()
        
        # Check each template category
        best_match = None
        best_score = 0
        
        for category, templates in self.query_templates.items():
            for template in templates:
                # Convert template to a fuzzy matching pattern
                fuzzy_pattern = template.replace(r'\w+', '\\w+').replace(r'\d+', '\\d+')
                fuzzy_pattern = re.sub(r'\([^)]+\)', '.+', fuzzy_pattern)
                
                # Try regex match first
                if re.search(template, query_text, re.IGNORECASE):
                    return category
                
                # Try fuzzy matching
                score = fuzz.partial_ratio(fuzzy_pattern, query_text)
                
                if score > best_score:
                    best_score = score
                    best_match = category
        
        # If we have a good fuzzy match, use it
        if best_score > 70:
            return best_match
        
        # Default to content query for general questions
        return 'content'
    
    def _extract_parameters(self, query_text: str, query_type: str) -> Dict[str, Any]:
        """
        Extract parameters from the query text based on its type.
        
        Args:
            query_text: Natural language query
            query_type: Type of query
            
        Returns:
            Dict[str, Any]: Extracted parameters
        """
        params = {'text': query_text}
        query_text = query_text.lower()
        
        # Extract cell references
        cell_refs = re.findall(r'([A-Z]+[0-9]+)', query_text, re.IGNORECASE)
        if cell_refs:
            params['cell_ref'] = cell_refs[0]
        
        # Extract sheet names
        sheet_matches = re.findall(r'(sheet|tab)\s+(\w+)', query_text, re.IGNORECASE)
        if sheet_matches:
            params['sheet'] = sheet_matches[0][1]
        else:
            # Try to find sheet name without 'sheet' or 'tab' keyword
            sheet_nodes = [self.kg.graph.nodes[n].get('name') 
                          for n, attrs in self.kg.graph.nodes(data=True) 
                          if attrs.get('type') == 'sheet']
            
            for sheet_name in sheet_nodes:
                if sheet_name.lower() in query_text:
                    params['sheet'] = sheet_name
                    break
        
        # Extract column names/headers
        column_matches = re.findall(r'column\s+["\']?([^"\']+)["\']?', query_text, re.IGNORECASE)
        if column_matches:
            params['column'] = column_matches[0]
        
        # Handle specific query types
        if query_type == 'metadata':
            if 'sheet' in query_text:
                params['entity'] = 'sheet'
            elif 'table' in query_text:
                params['entity'] = 'table'
            elif 'named range' in query_text or 'range' in query_text:
                params['entity'] = 'named_range'
            else:
                params['entity'] = 'workbook'
                
        elif query_type == 'formula':
            if 'dependenc' in query_text:
                params['aspect'] = 'dependencies'
            elif 'complex' in query_text:
                params['aspect'] = 'complex'
            else:
                params['aspect'] = 'all'
        
        return params
    
    def _determine_intent(self, query_text: str) -> str:
        """
        Determine the specific intent of the query.
        
        Args:
            query_text: Natural language query
            
        Returns:
            str: Query intent
        """
        query_text = query_text.lower()
        
        # Check each intent pattern
        for intent, pattern in self.intent_patterns.items():
            if re.search(pattern, query_text, re.IGNORECASE):
                return intent
        
        # Fall back to general intent based on query keywords
        if any(kw in query_text for kw in ['formula', 'calculate', 'compute']):
            return 'formula_query'
        elif any(kw in query_text for kw in ['sheet', 'tab']):
            return 'sheet_query'
        elif any(kw in query_text for kw in ['cell', 'value']):
            return 'value_query'
        elif any(kw in query_text for kw in ['table', 'structure']):
            return 'structure_query'
        elif any(kw in query_text for kw in ['meaning', 'semantic', 'about']):
            return 'semantic_query'
        
        # Default to general query
        return 'general_query'
    
    def _generate_nl_answer(self, response: Dict[str, Any], 
                          query_text: str, intent: str) -> str:
        """
        Generate a natural language answer based on the query response.
        
        Args:
            response: Query response
            query_text: Original query text
            intent: Query intent
            
        Returns:
            str: Natural language answer
        """
        if response.get('status') == 'error':
            return f"I couldn't answer that question: {response.get('message')}"
        
        # Handle based on intent
        if intent == 'find_value':
            if 'cells' in response and response['cells']:
                cell = response['cells'][0]
                return f"The value in cell {cell['address']} is {cell['value']}."
            elif 'cell' in response:
                cell = response['cell']
                return f"The value in cell {cell['address']} is {cell['value']}."
            else:
                return "I couldn't find the value you're looking for."
                
        elif intent == 'list_sheets':
            if 'sheets' in response:
                sheets = response['sheets']
                if len(sheets) == 1:
                    return f"There is 1 sheet in this workbook: {sheets[0]['name']}."
                else:
                    sheet_names = [sheet['name'] for sheet in sheets]
                    return f"There are {len(sheets)} sheets in this workbook: {', '.join(sheet_names)}."
            else:
                return "I couldn't find any sheets in this workbook."
                
        elif intent == 'sheet_summary':
            if 'cells' in response:
                cells = response['cells']
                return f"Sheet contains {len(cells)} cells with data. " + \
                       f"There are {len([c for c in cells if c.get('data_type') == 'number'])} numeric cells and " + \
                       f"{len([c for c in cells if c.get('data_type') == 'string'])} text cells."
            else:
                return "I couldn't generate a summary for that sheet."
                
        elif intent == 'formula_dependencies':
            if 'sample_formulas' in response:
                for formula in response['sample_formulas']:
                    if formula.get('address') == response.get('extracted_params', {}).get('cell_ref'):
                        deps = formula.get('dependencies', [])
                        if deps:
                            return f"The formula in cell {formula['address']} depends on {len(deps)} cells: {', '.join(deps)}."
                        else:
                            return f"The formula in cell {formula['address']} doesn't depend on any other cells."
            
            return "I couldn't find the formula dependencies you're looking for."
                
        elif intent == 'count_tables':
            if 'tables' in response:
                tables = response['tables']
                if len(tables) == 1:
                    return f"There is 1 table in this workbook: {tables[0]['name']}."
                else:
                    return f"There are {len(tables)} tables in this workbook."
            else:
                return "I couldn't find any tables in this workbook."
        
        # Generate answer based on response content
        if 'workbook' in response:
            wb = response['workbook']
            return f"This Excel workbook contains {wb.get('sheets_count', 0)} sheets, " + \
                   f"{wb.get('tables_count', 0)} tables, and {wb.get('named_ranges_count', 0)} named ranges. " + \
                   f"It appears to be related to {wb.get('domain', 'general data')}."
        
        elif 'formula_counts_by_sheet' in response:
            counts = response['formula_counts_by_sheet']
            total = sum(counts.values())
            types = response.get('formula_types', {})
            
            if total == 0:
                return "There are no formulas in this workbook."
            else:
                type_str = ", ".join([f"{count} {type_}" for type_, count in types.items()])
                return f"There are {total} formulas in this workbook. Types include: {type_str}."
        
        elif 'tables' in response:
            tables = response['tables']
            if not tables:
                return "There are no tables in this workbook."
            elif len(tables) == 1:
                table = tables[0]
                return f"There is 1 table named '{table['name']}' in sheet '{table['sheet']}'."
            else:
                return f"There are {len(tables)} tables in this workbook."
        
        elif 'domain' in response:
            return f"This workbook appears to be about {response['domain']}. " + \
                   f"It contains {sum(response.get('entity_types', {}).values())} entities " + \
                   f"and {response.get('relationship_count', 0)} relationships."
        
        # Default response
        return "I analyzed the Excel workbook based on your query. Check the detailed results for more information."
    
    def suggest_follow_up_questions(self, query_text: str, response: Dict[str, Any]) -> List[str]:
        """
        Suggest follow-up questions based on the query and response.
        
        Args:
            query_text: Original query text
            response: Query response
            
        Returns:
            List[str]: Suggested follow-up questions
        """
        suggestions = []
        
        # Based on query intent
        intent = response.get('intent', '')
        
        if intent == 'list_sheets' and 'sheets' in response:
            # Suggest exploring a specific sheet
            if len(response['sheets']) > 0:
                sheet = response['sheets'][0]['name']
                suggestions.append(f"What data is in sheet {sheet}?")
                suggestions.append(f"What formulas are used in sheet {sheet}?")
        
        elif intent == 'find_value' and ('cell' in response or 'cells' in response):
            # Suggest exploring formula or related cells
            cell_addr = (response.get('cell', {}).get('address') or 
                        response.get('cells', [{}])[0].get('address', ''))
            
            if cell_addr:
                suggestions.append(f"Is there a formula in cell {cell_addr}?")
                suggestions.append(f"What other cells reference {cell_addr}?")
        
        elif intent == 'formula_query' and 'sample_formulas' in response:
            # Suggest exploring specific formulas
            if response['sample_formulas']:
                formula = response['sample_formulas'][0]
                suggestions.append(f"What cells depend on the formula in {formula['address']}?")
                suggestions.append("What are the most complex formulas in this workbook?")
        
        # Based on response content
        if 'tables' in response and response['tables']:
            suggestions.append("What is the structure of the tables in this workbook?")
            
        if 'domain' in response:
            suggestions.append(f"What entities are found in this {response['domain']} data?")
            suggestions.append("What relationships exist between entities in this data?")
        
        # Default suggestions if nothing specific
        if not suggestions:
            suggestions = [
                "What sheets are in this workbook?",
                "What formulas are used in this workbook?",
                "What is the domain of this data?",
                "Are there any tables in this workbook?"
            ]
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def get_entities_from_query(self, query_text: str) -> Dict[str, List[str]]:
        """
        Extract entities mentioned in the query text.
        
        Args:
            query_text: Query text
            
        Returns:
            Dict[str, List[str]]: Dictionary mapping entity types to entity values
        """
        entities = {
            'sheet_names': [],
            'cell_refs': [],
            'column_names': [],
            'table_names': []
        }
        
        # Extract sheet names
        sheet_matches = re.findall(r'(sheet|tab)\s+(\w+)', query_text, re.IGNORECASE)
        if sheet_matches:
            entities['sheet_names'].extend([m[1] for m in sheet_matches])
        
        # Also try to match known sheet names
        sheet_nodes = [self.kg.graph.nodes[n].get('name') 
                      for n, attrs in self.kg.graph.nodes(data=True) 
                      if attrs.get('type') == 'sheet']
        
        for sheet_name in sheet_nodes:
            if sheet_name.lower() in query_text.lower():
                entities['sheet_names'].append(sheet_name)
        
        # Extract cell references
        cell_refs = re.findall(r'([A-Z]+[0-9]+)', query_text, re.IGNORECASE)
        if cell_refs:
            entities['cell_refs'].extend(cell_refs)
        
        # Extract column names
        column_matches = re.findall(r'column\s+["\']?([^"\']+)["\']?', query_text, re.IGNORECASE)
        if column_matches:
            entities['column_names'].extend(column_matches)
        
        # Extract table names
        table_nodes = [self.kg.graph.nodes[n].get('name') 
                      for n, attrs in self.kg.graph.nodes(data=True) 
                      if attrs.get('type') == 'table']
        
        for table_name in table_nodes:
            if table_name and table_name.lower() in query_text.lower():
                entities['table_names'].append(table_name)
        
        # Remove duplicates
        for entity_type in entities:
            entities[entity_type] = list(set(entities[entity_type]))
        
        return entities
    
    def search_similar_cells(self, search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for cells with values similar to the search term.
        
        Args:
            search_term: Term to search for
            limit: Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: Matching cells
        """
        results = []
        
        for node, attrs in self.kg.graph.nodes(data=True):
            if attrs.get('type') == 'cell' and 'value' in attrs:
                cell_value = str(attrs.get('value', ''))
                
                # Check for exact match
                if search_term.lower() in cell_value.lower():
                    results.append({
                        'address': node.replace('cell:', ''),
                        'value': attrs.get('value'),
                        'sheet': attrs.get('sheet'),
                        'match_type': 'exact',
                        'score': 100
                    })
                else:
                    # Try fuzzy matching
                    score = fuzz.partial_ratio(search_term.lower(), cell_value.lower())
                    
                    if score > 70:
                        results.append({
                            'address': node.replace('cell:', ''),
                            'value': attrs.get('value'),
                            'sheet': attrs.get('sheet'),
                            'match_type': 'fuzzy',
                            'score': score
                        })
        
        # Sort by score and limit results
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]
    
    def generate_explanation(self, node_id: str) -> str:
        """
        Generate a natural language explanation of a node in the knowledge graph.
        
        Args:
            node_id: ID of the node to explain
            
        Returns:
            str: Natural language explanation
        """
        if not self.kg.graph.has_node(node_id):
            return f"I couldn't find information about {node_id}."
        
        attrs = self.kg.graph.nodes[node_id]
        node_type = attrs.get('type')
        
        if node_type == 'workbook':
            return f"This is an Excel workbook located at {attrs.get('file_path')}. " + \
                   f"It appears to be related to {attrs.get('domain', 'general data')}."
        
        elif node_type == 'sheet':
            sheet_name = attrs.get('name')
            
            # Count cells in this sheet
            cell_count = sum(1 for _, cell_attrs in self.kg.graph.nodes(data=True)
                          if cell_attrs.get('type') == 'cell' and 
                             cell_attrs.get('sheet') == sheet_name)
            
            # Count formulas in this sheet
            formula_count = sum(1 for _, cell_attrs in self.kg.graph.nodes(data=True)
                             if cell_attrs.get('type') == 'cell' and 
                                cell_attrs.get('sheet') == sheet_name and
                                'formula' in cell_attrs)
            
            return f"This is a sheet named '{sheet_name}' with {attrs.get('max_row', 0)} rows and " + \
                   f"{attrs.get('max_column', 0)} columns. It contains {cell_count} cells with data, " + \
                   f"including {formula_count} formulas."
        
        elif node_type == 'cell':
            cell_addr = node_id.replace('cell:', '')
            value = attrs.get('value')
            formula = attrs.get('formula')
            
            if formula:
                # Get dependencies
                dependencies = []
                for pred in self.kg.graph.predecessors(node_id):
                    edge_data = self.kg.graph.get_edge_data(pred, node_id)
                    if edge_data and edge_data.get('relationship') == 'used_in':
                        dependencies.append(pred.replace('cell:', ''))
                
                dep_str = ', '.join(dependencies) if dependencies else 'no other cells'
                
                return f"This is cell {cell_addr} containing the formula: {formula}. " + \
                       f"It depends on {dep_str}. The calculated value is {value}."
            else:
                return f"This is cell {cell_addr} containing the value: {value}. " + \
                       f"It is a {attrs.get('data_type')} data type."
        
        elif node_type == 'table':
            sheet = attrs.get('sheet')
            ref = attrs.get('ref')
            
            return f"This is a table named '{attrs.get('name')}' located in sheet '{sheet}'. " + \
                   f"It covers the range {ref}."
        
        elif node_type == 'named_range':
            return f"This is a named range called '{attrs.get('name')}' with the value {attrs.get('value')}."
        
        else:
            # Generic explanation
            return f"This is a {node_type} with the following attributes: " + \
                   ', '.join(f"{k}: {v}" for k, v in attrs.items() if k != 'type')
    
    def generate_query_plan(self, query_text: str) -> Dict[str, Any]:
        """
        Generate a plan for answering a query.
        
        Args:
            query_text: Query text
            
        Returns:
            Dict[str, Any]: Query plan
        """
        # Extract entities
        entities = self.get_entities_from_query(query_text)
        
        # Determine query type and intent
        query_type = self._match_query_type(query_text)
        intent = self._determine_intent(query_text)
        
        # Define steps based on query type and intent
        steps = []
        
        if query_type == 'metadata':
            steps.append("Retrieve workbook metadata")
            
            if entities['sheet_names']:
                for sheet in entities['sheet_names']:
                    steps.append(f"Retrieve metadata for sheet '{sheet}'")
            else:
                steps.append("Retrieve metadata for all sheets")
                
            steps.append("Compose response with relevant metadata")
        
        elif query_type == 'content':
            if entities['cell_refs'] and entities['sheet_names']:
                for cell_ref in entities['cell_refs']:
                    for sheet in entities['sheet_names']:
                        steps.append(f"Retrieve value of cell {sheet}!{cell_ref}")
            
            elif entities['cell_refs']:
                for cell_ref in entities['cell_refs']:
                    steps.append(f"Search for cell {cell_ref} in all sheets")
            
            elif entities['sheet_names']:
                for sheet in entities['sheet_names']:
                    steps.append(f"Retrieve content from sheet '{sheet}'")
            
            else:
                steps.append("Retrieve summary of content across all sheets")
            
            steps.append("Compose response with relevant content")
        
        elif query_type == 'formula':
            if entities['cell_refs']:
                for cell_ref in entities['cell_refs']:
                    steps.append(f"Retrieve formula information for cell {cell_ref}")
                    steps.append(f"Identify dependencies for formula in cell {cell_ref}")
            else:
                steps.append("Retrieve summary of formulas across the workbook")
                steps.append("Identify common formula patterns")
            
            steps.append("Compose response with formula analysis")
        
        elif query_type == 'structure':
            steps.append("Retrieve structure information for the workbook")
            
            if entities['sheet_names']:
                for sheet in entities['sheet_names']:
                    steps.append(f"Analyze structure of sheet '{sheet}'")
            
            if entities['table_names']:
                for table in entities['table_names']:
                    steps.append(f"Retrieve structure of table '{table}'")
            
            steps.append("Identify tables, hierarchies, and other structural elements")
            steps.append("Compose response with structural analysis")
        
        elif query_type == 'semantic':
            steps.append("Retrieve semantic information for the workbook")
            steps.append("Identify domain, entities, and relationships")
            
            if entities['sheet_names']:
                for sheet in entities['sheet_names']:
                    steps.append(f"Analyze semantic meaning of data in sheet '{sheet}'")
            
            steps.append("Compose response with semantic analysis")
        
        return {
            'query': query_text,
            'query_type': query_type,
            'intent': intent,
            'entities': entities,
            'steps': steps
        }