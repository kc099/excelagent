import json
import logging
from typing import Dict, List, Any, Tuple, Set, Optional, Union
import networkx as nx
import numpy as np
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

class LLMAgentInterface:
    """
    Provides an interface for LLM agents to interact with the Excel knowledge graph.
    """
    def __init__(self, knowledge_graph, semantic_analyzer=None):
        """
        Initialize the LLM agent interface.
        
        Args:
            knowledge_graph: ExcelKnowledgeGraph instance
            semantic_analyzer: Optional SemanticAnalyzer instance
        """
        self.kg = knowledge_graph
        self.semantic_analyzer = semantic_analyzer
        self.context_window = {}
        self.query_history = []
        
    def query(self, query_text: str) -> Dict[str, Any]:
        """
        Process a natural language query from an LLM agent.
        
        Args:
            query_text: Natural language query
            
        Returns:
            Dict[str, Any]: Response with query results
        """
        # Log the query
        self.query_history.append(query_text)
        
        # Parse the query
        query_type, query_params = self._parse_query(query_text)
        
        # Process the query based on its type
        if query_type == 'metadata':
            response = self._get_metadata(query_params)
        elif query_type == 'content':
            response = self._get_content(query_params)
        elif query_type == 'structure':
            response = self._get_structure(query_params)
        elif query_type == 'formula':
            response = self._get_formula_info(query_params)
        elif query_type == 'semantic':
            response = self._get_semantic_info(query_params)
        elif query_type == 'help':
            response = self._get_help()
        else:
            response = {
                'status': 'error',
                'message': 'Unknown query type',
                'query': query_text
            }
        
        return response
    
    def _parse_query(self, query_text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Parse a natural language query to determine its type and parameters.
        
        Args:
            query_text: Natural language query
            
        Returns:
            Tuple[str, Dict[str, Any]]: Query type and parameters
        """
        query_text = query_text.lower()
        
        # Check for metadata queries
        if any(keyword in query_text for keyword in ['sheet', 'sheets', 'workbook', 'file', 'metadata']):
            return 'metadata', {'text': query_text}
        
        # Check for content queries
        elif any(keyword in query_text for keyword in ['data', 'value', 'values', 'cell', 'cells', 'content']):
            sheet = None
            cell_ref = None
            
            # Try to extract sheet name
            sheet_patterns = [
                r'sheet\s+(\w+)',
                r'in\s+(\w+)'
            ]
            
            for pattern in sheet_patterns:
                import re
                match = re.search(pattern, query_text)
                if match:
                    sheet = match.group(1)
                    break
            
            # Try to extract cell reference
            cell_patterns = [
                r'cell\s+([a-zA-Z]+[0-9]+)',
                r'([a-zA-Z]+[0-9]+)'
            ]
            
            for pattern in cell_patterns:
                match = re.search(pattern, query_text)
                if match:
                    cell_ref = match.group(1)
                    break
            
            return 'content', {
                'text': query_text,
                'sheet': sheet,
                'cell_ref': cell_ref
            }
        
        # Check for structure queries
        elif any(keyword in query_text for keyword in ['structure', 'layout', 'table', 'format']):
            return 'structure', {'text': query_text}
        
        # Check for formula queries
        elif any(keyword in query_text for keyword in ['formula', 'formulas', 'calculation', 'compute']):
            return 'formula', {'text': query_text}
        
        # Check for semantic queries
        elif any(keyword in query_text for keyword in ['meaning', 'entity', 'relationship', 'concept', 'semantic']):
            return 'semantic', {'text': query_text}
        
        # Check for help queries
        elif any(keyword in query_text for keyword in ['help', 'capabilities', 'can you', 'what can']):
            return 'help', {'text': query_text}
        
        # Default to content query
        else:
            return 'content', {'text': query_text}
    
    def _get_metadata(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get metadata about the Excel file.
        
        Args:
            params: Query parameters
            
        Returns:
            Dict[str, Any]: Metadata response
        """
        # Get workbook node
        workbook_nodes = [node for node, attrs in self.kg.graph.nodes(data=True) 
                        if attrs.get('type') == 'workbook']
        
        if not workbook_nodes:
            return {
                'status': 'error',
                'message': 'No workbook information found'
            }
        
        workbook_node = workbook_nodes[0]
        workbook_attrs = self.kg.graph.nodes[workbook_node]
        
        # Get sheet information
        sheets = []
        for node, attrs in self.kg.graph.nodes(data=True):
            if attrs.get('type') == 'sheet':
                sheet_info = {
                    'name': attrs.get('name', ''),
                    'rows': attrs.get('max_row', 0),
                    'columns': attrs.get('max_column', 0)
                }
                sheets.append(sheet_info)
        
        # Get table information
        tables = []
        for node, attrs in self.kg.graph.nodes(data=True):
            if attrs.get('type') == 'table':
                table_info = {
                    'name': attrs.get('name', ''),
                    'sheet': attrs.get('sheet', ''),
                    'reference': attrs.get('ref', ''),
                    'implicit': attrs.get('implicit', False)
                }
                tables.append(table_info)
        
        # Get named range information
        named_ranges = []
        for node, attrs in self.kg.graph.nodes(data=True):
            if attrs.get('type') == 'named_range':
                range_info = {
                    'name': attrs.get('name', ''),
                    'value': attrs.get('value', '')
                }
                named_ranges.append(range_info)
        
        # Build response
        response = {
            'status': 'success',
            'workbook': {
                'file_path': workbook_attrs.get('file_path', ''),
                'sheets_count': len(sheets),
                'tables_count': len(tables),
                'named_ranges_count': len(named_ranges),
                'domain': workbook_attrs.get('domain', 'general')
            },
            'sheets': sheets,
            'tables': tables,
            'named_ranges': named_ranges
        }
        
        return response
    
    def _get_content(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get content from the Excel file.
        
        Args:
            params: Query parameters
            
        Returns:
            Dict[str, Any]: Content response
        """
        sheet = params.get('sheet')
        cell_ref = params.get('cell_ref')
        
        # If specific cell is requested
        if cell_ref:
            # If sheet is specified
            if sheet:
                cell_addr = f"{sheet}!{cell_ref}"
                cell_node = f"cell:{cell_addr}"
                
                if self.kg.graph.has_node(cell_node):
                    cell_attrs = self.kg.graph.nodes[cell_node]
                    
                    return {
                        'status': 'success',
                        'cell': {
                            'address': cell_addr,
                            'value': cell_attrs.get('value'),
                            'formula': cell_attrs.get('formula'),
                            'data_type': cell_attrs.get('data_type'),
                            'semantic_type': cell_attrs.get('semantic_type')
                        }
                    }
                else:
                    # Try to find cell by just the reference in any sheet
                    cell_nodes = [node for node in self.kg.graph.nodes()
                                 if node.endswith(f"!{cell_ref}")]
                    
                    if cell_nodes:
                        results = []
                        for node in cell_nodes:
                            cell_attrs = self.kg.graph.nodes[node]
                            results.append({
                                'address': node.replace('cell:', ''),
                                'value': cell_attrs.get('value'),
                                'formula': cell_attrs.get('formula'),
                                'data_type': cell_attrs.get('data_type'),
                                'sheet': cell_attrs.get('sheet')
                            })
                        
                        return {
                            'status': 'success',
                            'message': f'Found {len(results)} cells matching reference {cell_ref}',
                            'cells': results
                        }
                    
                    return {
                        'status': 'error',
                        'message': f'Cell {cell_addr} not found'
                    }
            else:
                # Try to find cell by just the reference in any sheet
                cell_nodes = [node for node in self.kg.graph.nodes()
                             if f"!{cell_ref}" in node]
                
                if cell_nodes:
                    results = []
                    for node in cell_nodes:
                        cell_attrs = self.kg.graph.nodes[node]
                        results.append({
                            'address': node.replace('cell:', ''),
                            'value': cell_attrs.get('value'),
                            'formula': cell_attrs.get('formula'),
                            'data_type': cell_attrs.get('data_type'),
                            'sheet': cell_attrs.get('sheet')
                        })
                    
                    return {
                        'status': 'success',
                        'message': f'Found {len(results)} cells matching reference {cell_ref}',
                        'cells': results
                    }
                
                return {
                    'status': 'error',
                    'message': f'Cell with reference {cell_ref} not found'
                }
        
        # If sheet is specified but no cell
        elif sheet:
            # Get all cells in the sheet
            sheet_cells = []
            
            for node, attrs in self.kg.graph.nodes(data=True):
                if attrs.get('type') == 'cell' and attrs.get('sheet') == sheet:
                    cell_info = {
                        'address': node.replace('cell:', ''),
                        'row': attrs.get('row'),
                        'column': attrs.get('column'),
                        'value': attrs.get('value'),
                        'data_type': attrs.get('data_type')
                    }
                    sheet_cells.append(cell_info)
            
            if sheet_cells:
                # Limit to 100 cells to avoid overwhelming response
                if len(sheet_cells) > 100:
                    return {
                        'status': 'success',
                        'message': f'Found {len(sheet_cells)} cells in sheet {sheet} (showing first 100)',
                        'cells': sheet_cells[:100]
                    }
                else:
                    return {
                        'status': 'success',
                        'message': f'Found {len(sheet_cells)} cells in sheet {sheet}',
                        'cells': sheet_cells
                    }
            else:
                # Try fuzzy matching sheet name
                sheet_nodes = [node for node, attrs in self.kg.graph.nodes(data=True)
                             if attrs.get('type') == 'sheet']
                
                best_match = None
                best_score = 0
                
                for node in sheet_nodes:
                    sheet_name = self.kg.graph.nodes[node].get('name', '')
                    score = fuzz.ratio(sheet.lower(), sheet_name.lower())
                    
                    if score > best_score:
                        best_score = score
                        best_match = sheet_name
                
                if best_match and best_score > 70:
                    return {
                        'status': 'error',
                        'message': f'Sheet {sheet} not found. Did you mean {best_match}?'
                    }
                
                return {
                    'status': 'error',
                    'message': f'Sheet {sheet} not found or contains no data'
                }
        
        # No specific params - return summary of content
        else:
            # Get counts by sheet
            sheet_counts = {}
            
            for node, attrs in self.kg.graph.nodes(data=True):
                if attrs.get('type') == 'cell':
                    sheet = attrs.get('sheet')
                    if sheet:
                        if sheet not in sheet_counts:
                            sheet_counts[sheet] = 0
                        sheet_counts[sheet] += 1
            
            # Get data type counts
            data_type_counts = {}
            
            for node, attrs in self.kg.graph.nodes(data=True):
                if attrs.get('type') == 'cell':
                    data_type = attrs.get('data_type')
                    if data_type:
                        if data_type not in data_type_counts:
                            data_type_counts[data_type] = 0
                        data_type_counts[data_type] += 1
            
            return {
                'status': 'success',
                'message': 'Content summary',
                'cell_counts_by_sheet': sheet_counts,
                'data_type_counts': data_type_counts
            }
    
    def _get_structure(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get structural information about the Excel file.
        
        Args:
            params: Query parameters
            
        Returns:
            Dict[str, Any]: Structure response
        """
        # Get table information
        tables = []
        for node, attrs in self.kg.graph.nodes(data=True):
            if attrs.get('type') == 'table':
                sheet = attrs.get('sheet')
                ref = attrs.get('ref')
                
                # Get header information if available
                headers = []
                
                if sheet and ref:
                    # Parse reference to get range
                    import re
                    match = re.match(r'([A-Z]+)(\d+):([A-Z]+)(\d+)', ref)
                    
                    if match:
                        start_col, start_row, end_col, end_row = match.groups()
                        
                        # Look for header cells in the first row
                        for node2, attrs2 in self.kg.graph.nodes(data=True):
                            if (attrs2.get('type') == 'cell' and 
                                attrs2.get('sheet') == sheet and 
                                attrs2.get('row') == int(start_row)):
                                
                                col = attrs2.get('column_letter')
                                if col:
                                    headers.append({
                                        'column': col,
                                        'value': attrs2.get('value')
                                    })
                
                table_info = {
                    'name': attrs.get('name', ''),
                    'sheet': sheet,
                    'reference': ref,
                    'implicit': attrs.get('implicit', False),
                    'headers': headers
                }
                tables.append(table_info)
        
        # Get implicit hierarchies
        hierarchies = []
        if self.semantic_analyzer and 'hierarchies' in self.semantic_analyzer.concepts:
            for hierarchy in self.semantic_analyzer.concepts['hierarchies']:
                hierarchies.append({
                    'sheet': hierarchy['sheet'],
                    'levels': hierarchy['levels'],
                    'level_columns': hierarchy['level_cols']
                })
        
        # Get time series information
        time_series = []
        if self.semantic_analyzer and 'time_series' in self.semantic_analyzer.concepts:
            for ts in self.semantic_analyzer.concepts['time_series']:
                time_series.append({
                    'sheet': ts['sheet'],
                    'date_column': ts['date_column'],
                    'value_columns': ts['value_columns']
                })
        
        # Build response
        response = {
            'status': 'success',
            'tables': tables,
            'hierarchies': hierarchies,
            'time_series': time_series
        }
        
        return response
    
    def _get_formula_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get information about formulas in the Excel file.
        
        Args:
            params: Query parameters
            
        Returns:
            Dict[str, Any]: Formula response
        """
        # Get formula counts by sheet
        formula_counts = {}
        formula_types = {}
        
        for node, attrs in self.kg.graph.nodes(data=True):
            if attrs.get('type') == 'cell' and 'formula' in attrs:
                # Count by sheet
                sheet = attrs.get('sheet')
                if sheet:
                    if sheet not in formula_counts:
                        formula_counts[sheet] = 0
                    formula_counts[sheet] += 1
                
                # Categorize formula types
                formula = attrs.get('formula', '')
                formula_type = 'other'
                
                if 'SUM' in formula:
                    formula_type = 'sum'
                elif 'AVERAGE' in formula:
                    formula_type = 'average'
                elif 'COUNT' in formula:
                    formula_type = 'count'
                elif 'IF' in formula:
                    formula_type = 'conditional'
                elif 'VLOOKUP' in formula or 'HLOOKUP' in formula:
                    formula_type = 'lookup'
                elif 'DATE' in formula or 'TIME' in formula:
                    formula_type = 'date/time'
                
                if formula_type not in formula_types:
                    formula_types[formula_type] = 0
                formula_types[formula_type] += 1
        
        # Get sample formulas
        sample_formulas = []
        formulas_seen = set()
        
        for node, attrs in self.kg.graph.nodes(data=True):
            if attrs.get('type') == 'cell' and 'formula' in attrs:
                formula = attrs.get('formula', '')
                
                # Skip if we've seen this formula type
                formula_type = 'other'
                
                if 'SUM' in formula:
                    formula_type = 'sum'
                elif 'AVERAGE' in formula:
                    formula_type = 'average'
                elif 'COUNT' in formula:
                    formula_type = 'count'
                elif 'IF' in formula:
                    formula_type = 'conditional'
                elif 'VLOOKUP' in formula or 'HLOOKUP' in formula:
                    formula_type = 'lookup'
                elif 'DATE' in formula or 'TIME' in formula:
                    formula_type = 'date/time'
                
                if formula_type not in formulas_seen:
                    formulas_seen.add(formula_type)
                    
                    cell_addr = node.replace('cell:', '')
                    
                    # Get dependencies
                    dependencies = []
                    dep_cells = self.kg.get_formula_dependencies(cell_addr)
                    
                    for dep in dep_cells:
                        dependencies.append(dep)
                    
                    sample_formulas.append({
                        'address': cell_addr,
                        'formula': formula,
                        'type': formula_type,
                        'dependencies': dependencies
                    })
                
                # Stop once we have samples of each type
                if len(formulas_seen) >= len(formula_types):
                    break
        
        # Calculate cells with most dependencies
        cells_by_deps = []
        
        for node, attrs in self.kg.graph.nodes(data=True):
            if attrs.get('type') == 'cell' and 'formula' in attrs:
                cell_addr = node.replace('cell:', '')
                dependencies = self.kg.get_formula_dependencies(cell_addr)
                
                cells_by_deps.append({
                    'address': cell_addr,
                    'formula': attrs.get('formula', ''),
                    'dependency_count': len(dependencies)
                })
        
        # Sort by dependency count and take top 5
        cells_by_deps.sort(key=lambda x: x['dependency_count'], reverse=True)
        cells_by_deps = cells_by_deps[:5]
        
        # Build response
        response = {
            'status': 'success',
            'formula_counts_by_sheet': formula_counts,
            'formula_types': formula_types,
            'sample_formulas': sample_formulas,
            'complex_formulas': cells_by_deps
        }
        
        return response
    
    def _get_semantic_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get semantic information about the Excel data.
        
        Args:
            params: Query parameters
            
        Returns:
            Dict[str, Any]: Semantic response
        """
        if not self.semantic_analyzer:
            return {
                'status': 'error',
                'message': 'Semantic analyzer not available'
            }
        
        # Get domain information
        domain = self.semantic_analyzer.domain_type
        
        # Get entity information
        entities = self.semantic_analyzer.entities
        
        # Get relationship information
        relationships = self.semantic_analyzer.relationships
        
        # Get concept information
        concepts = self.semantic_analyzer.concepts
        
        # Get summary
        summary = self.semantic_analyzer.get_semantic_summary()
        
        # Build response
        response = {
            'status': 'success',
            'domain': domain,
            'entity_types': {key: len(val) for key, val in entities.items()},
            'relationship_count': len(relationships),
            'concept_types': list(concepts.keys()),
            'summary': summary
        }
        
        return response
    
    def _get_help(self) -> Dict[str, Any]:
        """
        Get help information about the interface capabilities.
        
        Returns:
            Dict[str, Any]: Help response
        """
        return {
            'status': 'success',
            'capabilities': [
                {
                    'name': 'Metadata Queries',
                    'description': 'Get information about the workbook, sheets, tables, and named ranges',
                    'examples': [
                        'What sheets does this workbook have?',
                        'How many tables are in this file?',
                        'Show me the metadata for this Excel file'
                    ]
                },
                {
                    'name': 'Content Queries',
                    'description': 'Get data from specific cells or sheets',
                    'examples': [
                        'What is the value in cell A1?',
                        'Show me the data in Sheet1',
                        'What does cell B5 contain?'
                    ]
                },
                {
                    'name': 'Structure Queries',
                    'description': 'Get information about the structure of the data',
                    'examples': [
                        'What tables are in this workbook?',
                        'Are there any hierarchies in the data?',
                        'Show me the structure of Sheet1'
                    ]
                },
                {
                    'name': 'Formula Queries',
                    'description': 'Get information about formulas in the workbook',
                    'examples': [
                        'What formulas are used in this workbook?',
                        'Show me the complex formulas',
                        'What dependencies does the formula in cell C10 have?'
                    ]
                },
                {
                    'name': 'Semantic Queries',
                    'description': 'Get semantic information about the data',
                    'examples': [
                        'What entities are in this data?',
                        'What is the domain of this workbook?',
                        'What relationships exist between the data elements?'
                    ]
                }
            ]
        }
        
    def set_context(self, context_key: str, context_value: Any) -> None:
        """
        Set a value in the context window.
        
        Args:
            context_key: Key for the context value
            context_value: Value to store in context
        """
        self.context_window[context_key] = context_value
    
    def get_context(self, context_key: str) -> Any:
        """
        Get a value from the context window.
        
        Args:
            context_key: Key for the context value
            
        Returns:
            Any: Value from the context window or None if not found
        """
        return self.context_window.get(context_key)
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get a schema describing the structure of the Excel data.
        
        Returns:
            Dict[str, Any]: Schema description
        """
        schema = {
            'workbook': {},
            'sheets': [],
            'tables': [],
            'named_ranges': []
        }
        
        # Get workbook information
        workbook_nodes = [node for node, attrs in self.kg.graph.nodes(data=True) 
                        if attrs.get('type') == 'workbook']
        
        if workbook_nodes:
            workbook_node = workbook_nodes[0]
            workbook_attrs = self.kg.graph.nodes[workbook_node]
            
            schema['workbook'] = {
                'file_path': workbook_attrs.get('file_path', ''),
                'domain': workbook_attrs.get('domain', 'general')
            }
        
        # Get sheet information
        for node, attrs in self.kg.graph.nodes(data=True):
            if attrs.get('type') == 'sheet':
                sheet_name = attrs.get('name', '')
                
                # Get column information
                columns = {}
                
                for cell_node, cell_attrs in self.kg.graph.nodes(data=True):
                    if (cell_attrs.get('type') == 'cell' and 
                        cell_attrs.get('sheet') == sheet_name and
                        cell_attrs.get('data_type') == 'header'):
                        
                        col = cell_attrs.get('column')
                        if col:
                            col_letter = cell_attrs.get('column_letter', '')
                            
                            columns[col] = {
                                'letter': col_letter,
                                'name': str(cell_attrs.get('value', '')),
                                'semantic_type': cell_attrs.get('semantic_type'),
                                'key_column': cell_attrs.get('key_column', False),
                                'calculated_column': cell_attrs.get('calculated_column', False)
                            }
                
                sheet_info = {
                    'name': sheet_name,
                    'rows': attrs.get('max_row', 0),
                    'columns': attrs.get('max_column', 0),
                    'columns_info': columns
                }
                
                schema['sheets'].append(sheet_info)
        
        # Get table information
        for node, attrs in self.kg.graph.nodes(data=True):
            if attrs.get('type') == 'table':
                table_info = {
                    'name': attrs.get('name', ''),
                    'sheet': attrs.get('sheet', ''),
                    'reference': attrs.get('ref', ''),
                    'implicit': attrs.get('implicit', False)
                }
                
                schema['tables'].append(table_info)
        
        # Get named range information
        for node, attrs in self.kg.graph.nodes(data=True):
            if attrs.get('type') == 'named_range':
                range_info = {
                    'name': attrs.get('name', ''),
                    'value': attrs.get('value', '')
                }
                
                schema['named_ranges'].append(range_info)
        
        return schema
    
    def to_json(self) -> str:
        """
        Convert the interface state to a JSON string.
        
        Returns:
            str: JSON representation of the interface state
        """
        state = {
            'context_window': self.context_window,
            'query_history': self.query_history
        }
        
        return json.dumps(state, indent=2)
    
    def from_json(self, json_str: str) -> None:
        """
        Restore the interface state from a JSON string.
        
        Args:
            json_str: JSON representation of the interface state
        """
        state = json.loads(json_str)
        
        self.context_window = state.get('context_window', {})
        self.query_history = state.get('query_history', [])