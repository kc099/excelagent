import networkx as nx
import numpy as np
import pandas as pd
import re
import logging
from typing import Dict, List, Any, Tuple, Set, Optional, Union
from collections import defaultdict
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import spacy
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

class SemanticAnalyzer:
    """
    Performs semantic analysis on Excel data represented as a knowledge graph.
    """
    def __init__(self, knowledge_graph):
        """
        Initialize the semantic analyzer with a knowledge graph.
        
        Args:
            knowledge_graph: ExcelKnowledgeGraph instance
        """
        self.kg = knowledge_graph
        self.entities = {}
        self.relationships = []
        self.concepts = {}
        self.domain_type = None
        
        # Load spaCy for NLP
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            logger.warning("Could not load spaCy model. Installing with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def analyze(self) -> bool:
        """
        Perform complete semantic analysis on the knowledge graph.
        
        Returns:
            bool: True if analysis was successful, False otherwise
        """
        try:
            # Detect domains
            self.detect_domain()
            
            # Refine semantic types
            self.refine_data_types()
            
            # Extract entities
            self.extract_entities()
            
            # Identify relationships
            self.identify_relationships()
            
            # Detect concepts
            self.detect_concepts()
            
            return True
            
        except Exception as e:
            logger.error(f"Error in semantic analysis: {e}")
            return False
    
    def detect_domain(self) -> str:
        """
        Detect the domain of the Excel data.
        
        Returns:
            str: Detected domain type
        """
        # Domain detection patterns
        domains = {
            'financial': ['revenue', 'expense', 'profit', 'loss', 'budget', 'forecast', 
                         'cash flow', 'balance', 'asset', 'liability', 'equity', 'tax',
                         'interest', 'depreciation', 'amortization', 'dividend', 'investment'],
            'sales': ['customer', 'product', 'sale', 'order', 'invoice', 'discount', 
                     'promotion', 'retail', 'wholesale', 'commission', 'territory', 'quota'],
            'inventory': ['product', 'stock', 'inventory', 'warehouse', 'shipment', 'supplier',
                        'reorder', 'backorder', 'sku', 'unit', 'quantity'],
            'project': ['task', 'milestone', 'deadline', 'resource', 'allocation', 'gantt',
                        'dependency', 'deliverable', 'schedule', 'timeline', 'progress'],
            'hr': ['employee', 'salary', 'benefit', 'payroll', 'performance', 'evaluation',
                  'attendance', 'leave', 'recruitment', 'termination', 'position']
        }
        
        # Count term occurrences
        domain_counts = {domain: 0 for domain in domains}
        
        # Check headers and column names
        for node, attrs in self.kg.graph.nodes(data=True):
            if attrs.get('type') == 'cell' and attrs.get('data_type') == 'header':
                value = str(attrs.get('value', '')).lower()
                for domain, terms in domains.items():
                    for term in terms:
                        if term in value:
                            domain_counts[domain] += 1
            
            # Check sheet names
            if attrs.get('type') == 'sheet':
                sheet_name = str(attrs.get('name', '')).lower()
                for domain, terms in domains.items():
                    for term in terms:
                        if term in sheet_name:
                            domain_counts[domain] += 2  # Sheet names are more significant
        
        # Determine the most likely domain
        max_count = 0
        detected_domain = 'general'
        
        for domain, count in domain_counts.items():
            if count > max_count:
                max_count = count
                detected_domain = domain
        
        # Set the domain type
        self.domain_type = detected_domain
        
        # Add domain information to the workbook node
        workbook_node = list(filter(lambda n: n[1].get('type') == 'workbook', 
                                 self.kg.graph.nodes(data=True)))
        
        if workbook_node:
            node_id = workbook_node[0][0]
            self.kg.graph.nodes[node_id]['domain'] = detected_domain
            self.kg.graph.nodes[node_id]['domain_confidence'] = max_count / sum(domain_counts.values()) if sum(domain_counts.values()) > 0 else 0
        
        return detected_domain
    
    def refine_data_types(self) -> None:
        """
        Refine data types of cells based on domain knowledge and patterns.
        """
        # Domain-specific patterns
        domain_patterns = {
            'financial': {
                'account_number': re.compile(r'^\d{4,}$'),
                'currency_code': re.compile(r'^[A-Z]{3}$'),
                'percentage': re.compile(r'^-?\d+(\.\d+)?%$')
            },
            'sales': {
                'order_number': re.compile(r'^[A-Z]{2,3}\d{4,}$'),
                'customer_id': re.compile(r'^C\d{4,}$'),
                'product_id': re.compile(r'^P\d{4,}$')
            },
            'inventory': {
                'sku': re.compile(r'^[A-Z0-9]{5,}$'),
                'upc': re.compile(r'^\d{12}$'),
                'quantity': re.compile(r'^\d+\s?(pcs|ea|units)$', re.IGNORECASE)
            },
            'project': {
                'task_id': re.compile(r'^T-\d+$'),
                'milestone': re.compile(r'^M\d+$'),
                'percentage_complete': re.compile(r'^(\d{1,3})%$')
            },
            'hr': {
                'employee_id': re.compile(r'^E\d{4,}$'),
                'social_security': re.compile(r'^\d{3}-\d{2}-\d{4}$'),
                'date_range': re.compile(r'^\d{2}/\d{2}/\d{4}\s*-\s*\d{2}/\d{2}/\d{4}$')
            }
        }
        
        # Apply domain-specific patterns if a domain was detected
        if self.domain_type in domain_patterns:
            patterns = domain_patterns[self.domain_type]
            
            for node, attrs in self.kg.graph.nodes(data=True):
                if attrs.get('type') == 'cell':
                    value = str(attrs.get('value', ''))
                    
                    for type_name, pattern in patterns.items():
                        if pattern.match(value):
                            self.kg.graph.nodes[node]['semantic_type'] = type_name
                            self.kg.semantic_types[node] = type_name
                            break
        
        # Check for calculated columns
        self._identify_calculated_columns()
        
        # Check for key columns
        self._identify_key_columns()
    
    def _identify_calculated_columns(self) -> None:
        """Identify columns that appear to be calculated from other columns."""
        # Group cells by sheet and column
        sheet_cols = defaultdict(lambda: defaultdict(list))
        
        for node, attrs in self.kg.graph.nodes(data=True):
            if attrs.get('type') == 'cell':
                sheet = attrs.get('sheet')
                col = attrs.get('column')
                
                if sheet and col:
                    sheet_cols[sheet][col].append((node, attrs))
        
        # For each sheet
        for sheet, columns in sheet_cols.items():
            # For each column
            for col, cells in columns.items():
                formula_count = 0
                
                # Count formulas in the column
                for node, attrs in cells:
                    if 'formula' in attrs:
                        formula_count += 1
                
                # If more than 50% of cells have formulas, mark as calculated
                if formula_count > 0 and formula_count / len(cells) > 0.5:
                    for node, _ in cells:
                        self.kg.graph.nodes[node]['calculated_column'] = True
                        
                        # Try to derive the calculation type
                        if 'formula' in self.kg.graph.nodes[node]:
                            formula = self.kg.graph.nodes[node]['formula']
                            
                            if 'SUM' in formula or '+' in formula:
                                self.kg.graph.nodes[node]['calculation_type'] = 'sum'
                            elif 'AVERAGE' in formula:
                                self.kg.graph.nodes[node]['calculation_type'] = 'average'
                            elif 'MAX' in formula:
                                self.kg.graph.nodes[node]['calculation_type'] = 'maximum'
                            elif 'MIN' in formula:
                                self.kg.graph.nodes[node]['calculation_type'] = 'minimum'
                            elif '*' in formula:
                                self.kg.graph.nodes[node]['calculation_type'] = 'product'
                            elif '/' in formula:
                                self.kg.graph.nodes[node]['calculation_type'] = 'division'
                            else:
                                self.kg.graph.nodes[node]['calculation_type'] = 'other'
    
    def _identify_key_columns(self) -> None:
        """Identify columns that appear to be keys (unique identifiers)."""
        # Group cells by sheet and column
        sheet_cols = defaultdict(lambda: defaultdict(list))
        
        for node, attrs in self.kg.graph.nodes(data=True):
            if attrs.get('type') == 'cell':
                sheet = attrs.get('sheet')
                col = attrs.get('column')
                
                if sheet and col:
                    sheet_cols[sheet][col].append((node, attrs))
        
        # For each sheet
        for sheet, columns in sheet_cols.items():
            # For each column
            for col, cells in columns.items():
                # Skip columns with less than 3 values
                if len(cells) < 3:
                    continue
                
                # Check if values are unique
                values = []
                for _, attrs in cells:
                    if 'value' in attrs and attrs['value'] is not None:
                        values.append(attrs['value'])
                
                unique_ratio = len(set(values)) / len(values) if values else 0
                
                # If over 90% unique and more than 3 values, it's likely a key
                if unique_ratio > 0.9 and len(values) > 3:
                    for node, _ in cells:
                        self.kg.graph.nodes[node]['key_column'] = True
                        
                        # Check if it looks like an ID column
                        if 'id' in str(self.kg.graph.nodes[node].get('value', '')).lower() or \
                           'code' in str(self.kg.graph.nodes[node].get('value', '')).lower() or \
                           'number' in str(self.kg.graph.nodes[node].get('value', '')).lower():
                            self.kg.graph.nodes[node]['id_column'] = True
    
    def extract_entities(self) -> Dict[str, List[str]]:
        """
        Extract entities from the knowledge graph.
        
        Returns:
            Dict[str, List[str]]: Dictionary mapping entity types to entity values
        """
        entities_by_type = defaultdict(list)
        
        # First, look for cells with known entity types
        for node, attrs in self.kg.graph.nodes(data=True):
            if attrs.get('type') == 'cell' and 'semantic_type' in attrs:
                semantic_type = attrs['semantic_type']
                value = attrs.get('value')
                
                if value is not None:
                    entities_by_type[semantic_type].append(str(value))
        
        # Now use NLP to extract additional entities from text cells
        if self.nlp:
            for node, attrs in self.kg.graph.nodes(data=True):
                if attrs.get('type') == 'cell' and attrs.get('data_type') == 'string':
                    value = attrs.get('value')
                    
                    if isinstance(value, str) and len(value) > 3:
                        doc = self.nlp(value)
                        
                        for ent in doc.ents:
                            entities_by_type[ent.label_].append(ent.text)
                            
                            # Update cell with entity type
                            self.kg.graph.nodes[node]['entity_type'] = ent.label_
        
        # Remove duplicates
        for entity_type in entities_by_type:
            entities_by_type[entity_type] = list(set(entities_by_type[entity_type]))
        
        # Store the extracted entities
        self.entities = dict(entities_by_type)
        
        return self.entities
    
    def identify_relationships(self) -> List[Dict[str, Any]]:
        """
        Identify relationships between entities in the knowledge graph.
        
        Returns:
            List[Dict[str, Any]]: List of relationship dictionaries
        """
        relationships = []
        
        # Find relationships between columns in the same table
        # First, find all implied tables
        tables = {}
        
        for node, attrs in self.kg.graph.nodes(data=True):
            if attrs.get('type') == 'table':
                tables[node] = {
                    'sheet': attrs.get('sheet'),
                    'ref': attrs.get('ref'),
                    'cells': []
                }
        
        # Associate cells with tables
        for table_node, table_data in tables.items():
            for successor in self.kg.graph.successors(table_node):
                if self.kg.graph.nodes[successor].get('type') == 'cell':
                    tables[table_node]['cells'].append(successor)
        
        # For each table, analyze relationships between columns
        for table_node, table_data in tables.items():
            cells = table_data['cells']
            
            # Group cells by column
            columns = defaultdict(list)
            for cell_node in cells:
                attrs = self.kg.graph.nodes[cell_node]
                col = attrs.get('column')
                if col:
                    columns[col].append(cell_node)
            
            # Find header cells
            headers = {}
            for col, col_cells in columns.items():
                for cell_node in col_cells:
                    if self.kg.graph.nodes[cell_node].get('data_type') == 'header':
                        headers[col] = cell_node
                        break
            
            # Analyze relationships between headers
            for col1, header1 in headers.items():
                for col2, header2 in headers.items():
                    if col1 != col2:
                        header1_text = str(self.kg.graph.nodes[header1].get('value', ''))
                        header2_text = str(self.kg.graph.nodes[header2].get('value', ''))
                        
                        relationship = self._infer_relationship(
                            header1_text, header2_text, 
                            columns[col1], columns[col2]
                        )
                        
                        if relationship:
                            relationships.append({
                                'source': header1,
                                'target': header2,
                                'relationship': relationship,
                                'confidence': 0.7
                            })
        
        # Find foreign key relationships between tables
        self._identify_foreign_keys(relationships)
        
        # Store the relationships
        self.relationships = relationships
        
        # Add relationships to the graph
        for rel in relationships:
            source = rel['source']
            target = rel['target']
            rel_type = rel['relationship']
            
            # Add edge if it doesn't exist
            if not self.kg.graph.has_edge(source, target):
                self.kg.graph.add_edge(source, target, 
                                      relationship=rel_type,
                                      semantic=True,
                                      confidence=rel['confidence'])
        
        return relationships
    
    def _infer_relationship(self, header1: str, header2: str, 
                          cells1: List[str], cells2: List[str]) -> Optional[str]:
        """
        Infer relationship type between two columns.
        
        Args:
            header1: Header text for first column
            header2: Header text for second column
            cells1: List of cell nodes in first column
            cells2: List of cell nodes in second column
            
        Returns:
            Optional[str]: Inferred relationship type or None
        """
        # Check for common relationship patterns
        common_patterns = [
            # One-to-many relationships
            ('id', 'names', 'has_name'),
            ('id', 'address', 'has_address'),
            ('customer', 'order', 'placed'),
            ('product', 'category', 'belongs_to'),
            ('employee', 'department', 'works_in'),
            ('parent', 'child', 'has_child'),
            
            # Time-based relationships
            ('date', 'value', 'recorded_on'),
            ('year', 'amount', 'amount_for'),
            ('quarter', 'sales', 'sales_for'),
            ('month', 'expenses', 'expenses_for'),
            
            # Ownership relationships
            ('owner', 'asset', 'owns'),
            ('company', 'subsidiary', 'has_subsidiary'),
            ('project', 'task', 'has_task'),
            
            # Action relationships
            ('sender', 'recipient', 'sent_to'),
            ('buyer', 'seller', 'bought_from'),
            ('author', 'document', 'authored')
        ]
        
        h1_lower = header1.lower()
        h2_lower = header2.lower()
        
        for pattern in common_patterns:
            src, tgt, rel = pattern
            
            if src in h1_lower and tgt in h2_lower:
                return rel
                
            if src in h2_lower and tgt in h1_lower:
                return f"inverse_{rel}"
        
        # Check for XYZ_ID and XYZ relationship
        id_pattern = re.compile(r'(.+)_id$', re.IGNORECASE)
        m1 = id_pattern.match(h1_lower)
        m2 = id_pattern.match(h2_lower)
        
        if m1 and m2.group(1) in h2_lower:
            return 'refers_to'
        
        if m2 and m1.group(1) in h1_lower:
            return 'referred_by'
        
        # Try to infer from similar values
        if len(cells1) > 0 and len(cells2) > 0:
            # Sample values from both columns
            values1 = [str(self.kg.graph.nodes[c].get('value', '')) for c in cells1[:10]]
            values2 = [str(self.kg.graph.nodes[c].get('value', '')) for c in cells2[:10]]
            
            # Check if values in one column appear in the other
            if any(v1 in v2 for v1 in values1 for v2 in values2):
                return 'contains'
            
            if any(v2 in v1 for v1 in values1 for v2 in values2):
                return 'contained_in'
        
        # No relationship detected
        return None
    
    def _identify_foreign_keys(self, relationships: List[Dict[str, Any]]) -> None:
        """
        Identify foreign key relationships between tables.
        
        Args:
            relationships: List to store identified relationships
        """
        # Find potential key columns
        key_columns = []
        
        for node, attrs in self.kg.graph.nodes(data=True):
            if attrs.get('type') == 'cell' and (
                attrs.get('key_column') or 
                attrs.get('id_column') or 
                'id' in str(attrs.get('value', '')).lower()
            ):
                key_columns.append(node)
        
        # For each key column, look for matching columns in other tables
        for key_node in key_columns:
            key_sheet = self.kg.graph.nodes[key_node].get('sheet')
            key_col = self.kg.graph.nodes[key_node].get('column')
            key_values = []
            
            # Get values from this key column
            for node, attrs in self.kg.graph.nodes(data=True):
                if (attrs.get('type') == 'cell' and 
                    attrs.get('sheet') == key_sheet and 
                    attrs.get('column') == key_col and
                    'value' in attrs):
                    key_values.append(str(attrs['value']))
            
            # Look for columns in other sheets with similar values
            for other_node, other_attrs in self.kg.graph.nodes(data=True):
                if (other_attrs.get('type') == 'cell' and 
                    other_attrs.get('sheet') != key_sheet and
                    'value' in other_attrs):
                    other_value = str(other_attrs['value'])
                    
                    # Check if this value matches any key value
                    if other_value in key_values:
                        # Find the header for this column
                        other_col = other_attrs.get('column')
                        other_sheet = other_attrs.get('sheet')
                        
                        for header_node, header_attrs in self.kg.graph.nodes(data=True):
                            if (header_attrs.get('type') == 'cell' and 
                                header_attrs.get('sheet') == other_sheet and
                                header_attrs.get('column') == other_col and
                                header_attrs.get('data_type') == 'header'):
                                
                                # Found a potential foreign key relationship
                                relationships.append({
                                    'source': key_node,
                                    'target': header_node,
                                    'relationship': 'foreign_key',
                                    'confidence': 0.8
                                })
                                break
    
    def detect_concepts(self) -> Dict[str, Dict[str, Any]]:
        """
        Detect higher-level concepts represented in the data.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of detected concepts
        """
        concepts = {}
        
        # Look for time series data
        time_series = self._detect_time_series()
        if time_series:
            concepts['time_series'] = time_series
        
        # Look for hierarchical data
        hierarchies = self._detect_hierarchies()
        if hierarchies:
            concepts['hierarchies'] = hierarchies
        
        # Look for aggregation patterns
        aggregations = self._detect_aggregations()
        if aggregations:
            concepts['aggregations'] = aggregations
        
        # Look for grouping patterns
        groupings = self._detect_groupings()
        if groupings:
            concepts['groupings'] = groupings
        
        # Store the concepts
        self.concepts = concepts
        
        return concepts
    
    def _detect_time_series(self) -> List[Dict[str, Any]]:
        """
        Detect time series data in the sheets.
        
        Returns:
            List[Dict[str, Any]]: List of time series information
        """
        time_series = []
        
        # Look for date columns
        date_columns = {}
        
        for node, attrs in self.kg.graph.nodes(data=True):
            if (attrs.get('type') == 'cell' and 
                (attrs.get('semantic_type') == 'date' or
                 'date' in str(attrs.get('value', '')).lower() or
                 'time' in str(attrs.get('value', '')).lower() or
                 'year' in str(attrs.get('value', '')).lower() or
                 'month' in str(attrs.get('value', '')).lower())):
                
                sheet = attrs.get('sheet')
                col = attrs.get('column')
                
                if sheet and col:
                    if sheet not in date_columns:
                        date_columns[sheet] = []
                    
                    date_columns[sheet].append((col, node))
        
        # For each date column, find associated numeric columns
        for sheet, columns in date_columns.items():
            for date_col, date_node in columns:
                # Look for numeric columns in the same sheet
                numeric_cols = []
                
                for node, attrs in self.kg.graph.nodes(data=True):
                    if (attrs.get('type') == 'cell' and 
                        attrs.get('sheet') == sheet and
                        attrs.get('data_type') == 'number'):
                        
                        numeric_col = attrs.get('column')
                        if numeric_col and numeric_col != date_col:
                            numeric_cols.append((numeric_col, node))
                
                # If we found numeric columns, this might be a time series
                if numeric_cols:
                    time_series.append({
                        'sheet': sheet,
                        'date_column': date_col,
                        'date_node': date_node,
                        'value_columns': [col for col, _ in numeric_cols],
                        'value_nodes': [node for _, node in numeric_cols]
                    })
        
        return time_series
    
    def _detect_hierarchies(self) -> List[Dict[str, Any]]:
        """
        Detect hierarchical data in the sheets.
        
        Returns:
            List[Dict[str, Any]]: List of hierarchy information
        """
        hierarchies = []
        
        # Look for indentation patterns in rows
        for sheet, sheet_data in self._group_by_sheet().items():
            # Get all cells in the sheet
            cells = [(node, attrs) for node, attrs in sheet_data if attrs.get('type') == 'cell']
            
            # Sort by row and column
            cells.sort(key=lambda x: (x[1].get('row', 0), x[1].get('column', 0)))
            
            # Group by row
            rows = defaultdict(list)
            for cell in cells:
                row = cell[1].get('row')
                if row:
                    rows[row].append(cell)
            
            # Look for indentation patterns
            indent_levels = {}
            
            for row_idx, row_cells in rows.items():
                # Skip empty rows
                if not row_cells:
                    continue
                
                # Find the leftmost non-empty cell
                leftmost = min(row_cells, key=lambda x: x[1].get('column', float('inf')))
                leftmost_col = leftmost[1].get('column')
                
                if leftmost_col:
                    indent_levels[row_idx] = leftmost_col
            
            # Check if there are different indentation levels
            if len(set(indent_levels.values())) > 1:
                # Group rows by indentation level
                rows_by_level = defaultdict(list)
                for row_idx, level in indent_levels.items():
                    rows_by_level[level].append(row_idx)
                
                # Sort levels
                sorted_levels = sorted(rows_by_level.keys())
                
                # Create hierarchy structure
                hierarchy = {
                    'sheet': sheet,
                    'levels': len(sorted_levels),
                    'level_cols': sorted_levels,
                    'rows_by_level': {level: rows for level, rows in rows_by_level.items()}
                }
                
                hierarchies.append(hierarchy)
        
        return hierarchies
    
    def _detect_aggregations(self) -> List[Dict[str, Any]]:
        """
        Detect aggregation patterns in the data.
        
        Returns:
            List[Dict[str, Any]]: List of aggregation information
        """
        aggregations = []
        
        # Look for formulas with SUM, AVERAGE, etc.
        agg_functions = ['SUM', 'AVERAGE', 'COUNT', 'MAX', 'MIN']
        
        for node, attrs in self.kg.graph.nodes(data=True):
            if attrs.get('type') == 'cell' and 'formula' in attrs:
                formula = attrs['formula']
                
                for func in agg_functions:
                    if func in formula:
                        # This is an aggregation cell
                        agg_info = {
                            'node': node,
                            'sheet': attrs.get('sheet'),
                            'function': func,
                            'formula': formula,
                            'dependencies': []
                        }
                        
                        # Get the dependencies
                        for dep in self.kg.get_formula_dependencies(node.replace('cell:', '')):
                            agg_info['dependencies'].append(dep)
                        
                        aggregations.append(agg_info)
        
        return aggregations
    
    def _detect_groupings(self) -> List[Dict[str, Any]]:
        """
        Detect grouping patterns in the data.
        
        Returns:
            List[Dict[str, Any]]: List of grouping information
        """
        groupings = []
        
        # Get sheet data
        sheet_data = self._group_by_sheet()
        
        # For each sheet, look for repeated patterns in headers or formatting
        for sheet, cells in sheet_data.items():
            header_cells = [cell for cell in cells if cell[1].get('data_type') == 'header']
            
            # Skip sheets with no headers
            if not header_cells:
                continue
            
            # Group headers by text similarity
            if header_cells:
                # Extract header texts
                header_texts = [(node, str(attrs.get('value', ''))) for node, attrs in header_cells]
                
                # Group similar headers
                groups = self._group_by_similarity(header_texts)
                
                # If we found groups, record them
                if len(groups) > 1:
                    groupings.append({
                        'sheet': sheet,
                        'type': 'header_similarity',
                        'groups': groups
                    })
            
            # Check for similar formatting
            style_groups = self._group_by_styling(cells)
            if len(style_groups) > 1:
                groupings.append({
                    'sheet': sheet,
                    'type': 'formatting',
                    'groups': style_groups
                })
        
        return groupings
    
    def _group_by_sheet(self) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
        """
        Group nodes by sheet.
        
        Returns:
            Dict[str, List[Tuple[str, Dict[str, Any]]]]: Dictionary mapping sheet names to lists of node tuples
        """
        sheet_data = defaultdict(list)
        
        for node, attrs in self.kg.graph.nodes(data=True):
            sheet = attrs.get('sheet')
            if sheet:
                sheet_data[sheet].append((node, attrs))
        
        return sheet_data
    
    def _group_by_similarity(self, items: List[Tuple[str, str]], 
                           threshold: float = 70) -> List[List[str]]:
        """
        Group items by text similarity.
        
        Args:
            items: List of (node_id, text) tuples
            threshold: Similarity threshold (0-100)
            
        Returns:
            List[List[str]]: List of groups, where each group is a list of node IDs
        """
        groups = []
        grouped = set()
        
        for i, (node1, text1) in enumerate(items):
            if node1 in grouped:
                continue
                
            group = [node1]
            grouped.add(node1)
            
            for j, (node2, text2) in enumerate(items):
                if i != j and node2 not in grouped:
                    # Compare text similarity
                    similarity = fuzz.ratio(text1, text2)
                    
                    if similarity >= threshold:
                        group.append(node2)
                        grouped.add(node2)
            
            if len(group) > 1:
                groups.append(group)
        
        # Add any ungrouped items as single-item groups
        for node, _ in items:
            if node not in grouped:
                groups.append([node])
                grouped.add(node)
        
        return groups
    
    def _group_by_styling(self, cells: List[Tuple[str, Dict[str, Any]]]) -> List[List[str]]:
        """
        Group cells by similar styling.
        
        Args:
            cells: List of (node_id, attrs) tuples
            
        Returns:
            List[List[str]]: List of groups, where each group is a list of node IDs
        """
        style_groups = defaultdict(list)
        
        for node, attrs in cells:
            # Extract style attributes
            style_attrs = {}
            for key, value in attrs.items():
                if key.startswith('style_'):
                    style_attrs[key] = value
            
            # Skip cells with no styling
            if not style_attrs:
                continue
            
            # Create a style signature
            style_sig = json.dumps(style_attrs, sort_keys=True)
            style_groups[style_sig].append(node)
        
        # Return groups with more than one cell
        return [group for group in style_groups.values() if len(group) > 1]
    
    def enrich_graph(self) -> None:
        """Enrich the knowledge graph with semantic information."""
        # Add entity types
        for entity_type, entities in self.entities.items():
            for entity in entities:
                # Find nodes with this entity value
                for node, attrs in self.kg.graph.nodes(data=True):
                    if attrs.get('type') == 'cell' and str(attrs.get('value', '')) == entity:
                        self.kg.graph.nodes[node]['entity_type'] = entity_type
        
        # Add concepts
        if 'time_series' in self.concepts:
            for ts in self.concepts['time_series']:
                # Create a time series node
                ts_node = f"concept:time_series:{ts['sheet']}_{ts['date_column']}"
                
                self.kg.graph.add_node(ts_node,
                                      type='concept',
                                      concept_type='time_series',
                                      sheet=ts['sheet'])
                
                # Link to date column
                self.kg.graph.add_edge(ts_node, ts['date_node'], 
                                      relationship='time_dimension')
                
                # Link to value columns
                for value_node in ts['value_nodes']:
                    self.kg.graph.add_edge(ts_node, value_node, 
                                          relationship='measure')
        
        # Add hierarchy concepts
        if 'hierarchies' in self.concepts:
            for hierarchy in self.concepts['hierarchies']:
                # Create a hierarchy node
                h_node = f"concept:hierarchy:{hierarchy['sheet']}"
                
                self.kg.graph.add_node(h_node,
                                      type='concept',
                                      concept_type='hierarchy',
                                      sheet=hierarchy['sheet'],
                                      levels=hierarchy['levels'])
                
                # Link to level columns
                for level, rows in hierarchy['rows_by_level'].items():
                    # Find cells in these rows
                    for row in rows:
                        for node, attrs in self.kg.graph.nodes(data=True):
                            if (attrs.get('type') == 'cell' and 
                                attrs.get('sheet') == hierarchy['sheet'] and
                                attrs.get('row') == row):
                                
                                self.kg.graph.add_edge(h_node, node, 
                                                      relationship=f'level_{level}')
        
        # Add domain info
        if self.domain_type:
            for node, attrs in self.kg.graph.nodes(data=True):
                if attrs.get('type') in ['sheet', 'workbook']:
                    self.kg.graph.nodes[node]['domain'] = self.domain_type
    
    def get_semantic_summary(self) -> Dict[str, Any]:
        """
        Generate a semantic summary of the Excel data.
        
        Returns:
            Dict[str, Any]: Semantic summary dictionary
        """
        summary = {
            'domain': self.domain_type,
            'entities': self.entities,
            'relationships': [{
                'source': rel['source'].replace('cell:', ''),
                'target': rel['target'].replace('cell:', ''),
                'relationship': rel['relationship']
            } for rel in self.relationships],
            'concepts': self.concepts,
            'sheets': []
        }
        
        # Get sheet information
        for node, attrs in self.kg.graph.nodes(data=True):
            if attrs.get('type') == 'sheet':
                sheet_name = attrs.get('name')
                
                sheet_info = {
                    'name': sheet_name,
                    'num_rows': attrs.get('max_row', 0),
                    'num_columns': attrs.get('max_column', 0),
                    'tables': [],
                    'key_columns': [],
                    'calculated_columns': []
                }
                
                # Get tables in this sheet
                for table_node, table_attrs in self.kg.graph.nodes(data=True):
                    if table_attrs.get('type') == 'table' and table_attrs.get('sheet') == sheet_name:
                        sheet_info['tables'].append({
                            'name': table_attrs.get('name'),
                            'reference': table_attrs.get('ref'),
                            'implicit': table_attrs.get('implicit', False)
                        })
                
                # Get key columns
                for cell_node, cell_attrs in self.kg.graph.nodes(data=True):
                    if (cell_attrs.get('type') == 'cell' and 
                        cell_attrs.get('sheet') == sheet_name):
                        
                        if cell_attrs.get('key_column'):
                            col = cell_attrs.get('column')
                            if col not in sheet_info['key_columns']:
                                sheet_info['key_columns'].append(col)
                                
                        if cell_attrs.get('calculated_column'):
                            col = cell_attrs.get('column')
                            if col not in sheet_info['calculated_columns']:
                                sheet_info['calculated_columns'].append({
                                    'column': col,
                                    'calculation_type': cell_attrs.get('calculation_type', 'unknown')
                                })
                
                summary['sheets'].append(sheet_info)
        
        return summary