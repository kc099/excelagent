import networkx as nx
import pandas as pd
import json
import logging
from typing import Dict, List, Any, Tuple, Set, Optional, Union
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

class ExcelKnowledgeGraph:
    """
    Constructs and manages a knowledge graph representation of Excel data.
    """
    def __init__(self):
        """Initialize the knowledge graph with a directed graph structure."""
        self.graph = nx.DiGraph()
        self.semantic_types = {}
        self.entity_map = {}
        self.relationship_types = set()
        
    def build_from_parser(self, excel_parser) -> bool:
        """
        Build the knowledge graph from an ExcelParser instance.
        
        Args:
            excel_parser: Initialized and parsed ExcelParser instance
            
        Returns:
            bool: True if graph was successfully built, False otherwise
        """
        try:
            # Add workbook node
            self.graph.add_node('workbook', 
                                type='workbook',
                                file_path=excel_parser.file_path)
            
            # Add sheet nodes
            sheet_metadata = excel_parser.get_sheet_metadata()
            for sheet_name, metadata in sheet_metadata.items():
                self._add_sheet(sheet_name, metadata)
                
                # Link sheet to workbook
                self.graph.add_edge('workbook', f"sheet:{sheet_name}", 
                                   relationship='contains')
            
            # Add cell nodes with their data
            for sheet_name, df in excel_parser.sheet_data.items():
                self._add_cells_from_dataframe(sheet_name, df)
            
            # Add named ranges
            for range_name, range_value in excel_parser.named_ranges.items():
                self._add_named_range(range_name, range_value)
            
            # Add tables
            for table_name, table_data in excel_parser.tables.items():
                self._add_table(table_name, table_data)
            
            # Add formulas and their dependencies
            for sheet_name, formulas in excel_parser.formulas.items():
                for cell_addr, formula_data in formulas.items():
                    if isinstance(formula_data, dict):
                        formula = formula_data.get('formula', '')
                        dependencies = formula_data.get('dependencies', [])
                        tokens = formula_data.get('tokens', [])
                    else:
                        # For backward compatibility
                        formula = formula_data
                        dependencies = []
                        tokens = []
                    
                    self._add_formula(cell_addr, formula, dependencies, tokens)
            
            # Add cell styles
            cell_styles = excel_parser.get_cell_styles()
            for cell_addr, style_data in cell_styles.items():
                if self.graph.has_node(f"cell:{cell_addr}"):
                    for style_key, style_value in style_data.items():
                        if isinstance(style_value, dict):
                            for sub_key, sub_value in style_value.items():
                                if sub_value is not None:
                                    self.graph.nodes[f"cell:{cell_addr}"][f"style_{style_key}_{sub_key}"] = sub_value
                        elif style_value is not None:
                            self.graph.nodes[f"cell:{cell_addr}"][f"style_{style_key}"] = style_value
            
            # Apply semantic enrichment
            self._analyze_data_types()
            self._identify_table_structures()
            
            return True
            
        except Exception as e:
            logger.error(f"Error building knowledge graph: {e}")
            return False
    
    def _add_sheet(self, sheet_name: str, metadata: Dict[str, Any]) -> None:
        """
        Add a sheet node to the knowledge graph.
        
        Args:
            sheet_name: Name of the sheet
            metadata: Sheet metadata dictionary
        """
        sheet_node = f"sheet:{sheet_name}"
        
        # Add the sheet node with its metadata
        node_attrs = {'type': 'sheet', 'name': sheet_name}
        node_attrs.update(metadata)
        
        self.graph.add_node(sheet_node, **node_attrs)
    
    def _add_cells_from_dataframe(self, sheet_name: str, df: pd.DataFrame) -> None:
        """
        Add cells from a pandas DataFrame to the knowledge graph.
        
        Args:
            sheet_name: Name of the sheet
            df: DataFrame containing the sheet data
        """
        sheet_node = f"sheet:{sheet_name}"
        
        # Get column headers if they exist
        headers = df.columns.tolist()
        
        # Add column header nodes
        for col_idx, header in enumerate(headers):
            header_str = str(header)
            col_letter = self._get_column_letter(col_idx + 1)
            cell_addr = f"{sheet_name}!{col_letter}1"
            cell_node = f"cell:{cell_addr}"
            
            # Add the header cell node
            self.graph.add_node(cell_node, 
                               type='cell',
                               sheet=sheet_name,
                               row=1,
                               column=col_idx + 1,
                               value=header_str,
                               data_type='header',
                               column_letter=col_letter)
            
            # Link the header cell to the sheet
            self.graph.add_edge(sheet_node, cell_node, relationship='contains')
        
        # Add data cells
        for row_idx, row in df.iterrows():
            # Skip header row if it was used for column names
            actual_row = row_idx + 2 if len(headers) > 0 and all(isinstance(h, str) for h in headers) else row_idx + 1
            
            for col_idx, value in enumerate(row):
                if pd.notna(value):  # Skip empty cells
                    col_letter = self._get_column_letter(col_idx + 1)
                    cell_addr = f"{sheet_name}!{col_letter}{actual_row}"
                    cell_node = f"cell:{cell_addr}"
                    
                    # Determine data type
                    if isinstance(value, str):
                        data_type = 'string'
                    elif isinstance(value, (int, float)):
                        data_type = 'number'
                    elif isinstance(value, bool):
                        data_type = 'boolean'
                    elif pd.isna(value):
                        data_type = 'null'
                    else:
                        data_type = 'unknown'
                    
                    # Add the cell node
                    self.graph.add_node(cell_node,
                                       type='cell',
                                       sheet=sheet_name,
                                       row=actual_row,
                                       column=col_idx + 1,
                                       value=value,
                                       data_type=data_type,
                                       column_letter=col_letter)
                    
                    # Link the cell to the sheet
                    self.graph.add_edge(sheet_node, cell_node, relationship='contains')
                    
                    # Link to column header if it exists
                    if col_idx < len(headers):
                        header_cell = f"cell:{sheet_name}!{col_letter}1"
                        if self.graph.has_node(header_cell):
                            self.graph.add_edge(header_cell, cell_node, relationship='header_for')
    
    def _add_named_range(self, range_name: str, range_value: str) -> None:
        """
        Add a named range node to the knowledge graph.
        
        Args:
            range_name: Name of the range
            range_value: Range formula/reference
        """
        range_node = f"named_range:{range_name}"
        
        # Add the named range node
        self.graph.add_node(range_node,
                           type='named_range',
                           name=range_name,
                           value=range_value)
        
        # Link to workbook
        self.graph.add_edge('workbook', range_node, relationship='defines')
        
        # Try to resolve the cells in the range
        cells = self._resolve_range_cells(range_value)
        for cell_addr in cells:
            cell_node = f"cell:{cell_addr}"
            if self.graph.has_node(cell_node):
                self.graph.add_edge(range_node, cell_node, relationship='contains')
    
    def _add_table(self, table_name: str, table_data: Dict[str, Any]) -> None:
        """
        Add a table node to the knowledge graph.
        
        Args:
            table_name: Name of the table
            table_data: Table metadata and reference
        """
        table_node = f"table:{table_name}"
        
        # Add the table node
        self.graph.add_node(table_node,
                           type='table',
                           name=table_name,
                           **table_data)
        
        # Link to sheet
        sheet_name = table_data.get('sheet')
        if sheet_name:
            sheet_node = f"sheet:{sheet_name}"
            if self.graph.has_node(sheet_node):
                self.graph.add_edge(sheet_node, table_node, relationship='contains')
        
        # Try to resolve the cells in the table
        table_ref = table_data.get('ref')
        if table_ref:
            cells = self._resolve_range_cells(f"{sheet_name}!{table_ref}")
            for cell_addr in cells:
                cell_node = f"cell:{cell_addr}"
                if self.graph.has_node(cell_node):
                    self.graph.add_edge(table_node, cell_node, relationship='contains')
    
    def _add_formula(self, cell_addr: str, formula: str, dependencies: List[str], tokens: List[Dict[str, str]]) -> None:
        """
        Add a formula node to the knowledge graph.
        
        Args:
            cell_addr: Cell address containing the formula
            formula: Formula string
            dependencies: List of cell references the formula depends on
            tokens: Tokens from formula parsing
        """
        cell_node = f"cell:{cell_addr}"
        
        # Check if cell node exists
        if not self.graph.has_node(cell_node):
            # Extract sheet name and cell reference
            parts = cell_addr.split('!')
            if len(parts) == 2:
                sheet_name, cell_ref = parts
                
                # Create cell node if it doesn't exist
                self.graph.add_node(cell_node,
                                   type='cell',
                                   sheet=sheet_name,
                                   value=formula,
                                   data_type='formula')
                
                # Link to sheet
                sheet_node = f"sheet:{sheet_name}"
                if self.graph.has_node(sheet_node):
                    self.graph.add_edge(sheet_node, cell_node, relationship='contains')
        
        # Update cell with formula information
        self.graph.nodes[cell_node]['formula'] = formula
        self.graph.nodes[cell_node]['formula_tokens'] = tokens
        
        # Add dependencies
        for dep_addr in dependencies:
            dep_node = f"cell:{dep_addr}"
            
            # Handle range references
            if ':' in dep_addr:
                cells = self._resolve_range_cells(dep_addr)
                for cell in cells:
                    single_dep_node = f"cell:{cell}"
                    if self.graph.has_node(single_dep_node):
                        self.graph.add_edge(single_dep_node, cell_node, relationship='used_in')
            elif self.graph.has_node(dep_node):
                self.graph.add_edge(dep_node, cell_node, relationship='used_in')
    
    def _resolve_range_cells(self, range_ref: str) -> List[str]:
        """
        Resolve a range reference to individual cell addresses.
        
        Args:
            range_ref: Range reference (e.g., 'Sheet1!A1:B3')
            
        Returns:
            List[str]: List of individual cell addresses
        """
        cells = []
        
        try:
            # Handle sheet specification
            if '!' in range_ref:
                sheet, cell_range = range_ref.split('!')
            else:
                # Assume it's the first sheet if not specified
                sheet = self.graph.nodes.get('workbook', {}).get('sheet_names', ['Sheet1'])[0]
                cell_range = range_ref
            
            # Handle single cell reference
            if ':' not in cell_range:
                return [range_ref]
            
            # Handle range reference
            start_ref, end_ref = cell_range.split(':')
            
            # Extract column letters and row numbers
            start_col_match = re.match(r'([A-Z]+)(\d+)', start_ref)
            end_col_match = re.match(r'([A-Z]+)(\d+)', end_ref)
            
            if start_col_match and end_col_match:
                start_col, start_row = start_col_match.groups()
                end_col, end_row = end_col_match.groups()
                
                # Convert column letters to indices
                start_col_idx = self._column_letter_to_index(start_col)
                end_col_idx = self._column_letter_to_index(end_col)
                
                # Convert row strings to integers
                start_row = int(start_row)
                end_row = int(end_row)
                
                # Generate all cell references in the range
                for row in range(start_row, end_row + 1):
                    for col_idx in range(start_col_idx, end_col_idx + 1):
                        col_letter = self._get_column_letter(col_idx)
                        cells.append(f"{sheet}!{col_letter}{row}")
        
        except Exception as e:
            logger.warning(f"Error resolving range {range_ref}: {e}")
        
        return cells
    
    def _get_column_letter(self, col_idx: int) -> str:
        """
        Convert a column index to a column letter.
        
        Args:
            col_idx: Column index (1-based)
            
        Returns:
            str: Column letter (e.g., 'A', 'B', 'AA')
        """
        result = ""
        while col_idx > 0:
            col_idx, remainder = divmod(col_idx - 1, 26)
            result = chr(65 + remainder) + result
        return result
    
    def _column_letter_to_index(self, column_letter: str) -> int:
        """
        Convert a column letter to a column index.
        
        Args:
            column_letter: Column letter (e.g., 'A', 'B', 'AA')
            
        Returns:
            int: Column index (1-based)
        """
        result = 0
        for char in column_letter:
            result = result * 26 + (ord(char) - 64)
        return result
    
    def _analyze_data_types(self) -> None:
        """
        Analyze and infer semantic data types for cells in the graph.
        """
        # Patterns for different data types
        patterns = {
            'date': re.compile(r'^(19|20)\d{2}[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12][0-9]|3[01])$'),
            'percentage': re.compile(r'^-?\d+(\.\d+)?%$'),
            'currency': re.compile(r'^[$€£¥](-?\d+(\.\d+)?)$|^(-?\d+(\.\d+)?)\s*[$€£¥]$'),
            'email': re.compile(r'^[\w\.-]+@[\w\.-]+\.\w+$'),
            'phone': re.compile(r'^\+?[\d\s\-\(\)]{7,}$'),
            'url': re.compile(r'^(https?://|www\.)\S+\.\S+$'),
            'ip_address': re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')
        }
        
        # Check for numeric formats
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get('type') == 'cell':
                value = attrs.get('value')
                if value is not None:
                    value_str = str(value)
                    
                    # Check for each pattern
                    for type_name, pattern in patterns.items():
                        if pattern.match(value_str):
                            self.graph.nodes[node]['semantic_type'] = type_name
                            self.semantic_types[node] = type_name
                            break
                    
                    # Check for number format clues
                    number_format = attrs.get('style_number_format')
                    if number_format:
                        if '%' in number_format:
                            self.graph.nodes[node]['semantic_type'] = 'percentage'
                            self.semantic_types[node] = 'percentage'
                        elif any(c in number_format for c in ['$', '€', '£', '¥']):
                            self.graph.nodes[node]['semantic_type'] = 'currency'
                            self.semantic_types[node] = 'currency'
                        elif any(pattern in number_format.lower() for pattern in ['yy', 'mm', 'dd']):
                            self.graph.nodes[node]['semantic_type'] = 'date'
                            self.semantic_types[node] = 'date'
    
    def _identify_table_structures(self) -> None:
        """
        Identify implicit table structures in the sheets.
        """
        # Group cells by sheet
        cells_by_sheet = defaultdict(list)
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get('type') == 'cell':
                sheet = attrs.get('sheet')
                if sheet:
                    cells_by_sheet[sheet].append((node, attrs))
        
        # Analyze each sheet for table structures
        for sheet, cells in cells_by_sheet.items():
            # Sort cells by row and column
            cells.sort(key=lambda x: (x[1].get('row', 0), x[1].get('column', 0)))
            
            # Group cells by row
            rows = defaultdict(list)
            for node, attrs in cells:
                row = attrs.get('row')
                if row:
                    rows[row].append((node, attrs))
            
            # Check for consecutive rows with similar structure
            # (This is a simplified approach - real implementation would be more sophisticated)
            potential_tables = []
            current_table = []
            prev_row_len = 0
            
            for row_idx in sorted(rows.keys()):
                row_cells = rows[row_idx]
                
                # Skip empty rows
                if not row_cells:
                    if current_table:
                        potential_tables.append(current_table)
                        current_table = []
                    prev_row_len = 0
                    continue
                
                # Check if this row has similar structure to previous row
                if prev_row_len == 0 or len(row_cells) == prev_row_len:
                    current_table.append((row_idx, row_cells))
                    prev_row_len = len(row_cells)
                else:
                    if current_table:
                        potential_tables.append(current_table)
                        current_table = []
                    prev_row_len = 0
            
            # Add the last table if it exists
            if current_table:
                potential_tables.append(current_table)
            
            # Create implicit table nodes for tables with at least 2 rows
            for table_idx, table_rows in enumerate(potential_tables):
                if len(table_rows) > 1:
                    # Create table node
                    table_name = f"Implicit_Table_{sheet}_{table_idx}"
                    table_node = f"table:{table_name}"
                    
                    # Get the table bounding box
                    min_row = min(row[0] for row in table_rows)
                    max_row = max(row[0] for row in table_rows)
                    
                    # Get min and max columns
                    min_col = float('inf')
                    max_col = 0
                    for _, row_cells in table_rows:
                        for _, attrs in row_cells:
                            col = attrs.get('column', 0)
                            min_col = min(min_col, col)
                            max_col = max(max_col, col)
                    
                    # Convert to column letters
                    min_col_letter = self._get_column_letter(min_col)
                    max_col_letter = self._get_column_letter(max_col)
                    
                    # Create table reference
                    table_ref = f"{min_col_letter}{min_row}:{max_col_letter}{max_row}"
                    
                    # Add table node
                    self.graph.add_node(table_node,
                                       type='table',
                                       name=table_name,
                                       sheet=sheet,
                                       ref=table_ref,
                                       implicit=True)
                    
                    # Link to sheet
                    sheet_node = f"sheet:{sheet}"
                    if self.graph.has_node(sheet_node):
                        self.graph.add_edge(sheet_node, table_node, relationship='contains')
                    
                    # Link to cells
                    for row_idx, row_cells in table_rows:
                        for cell_node, _ in row_cells:
                            self.graph.add_edge(table_node, cell_node, relationship='contains')
                    
                    # Check if first row might be headers
                    first_row_idx, first_row_cells = table_rows[0]
                    second_row_idx, second_row_cells = table_rows[1]
                    
                    # Heuristic: If first row has different data types than second row,
                    # it might be headers
                    first_row_types = [attrs.get('data_type') for _, attrs in first_row_cells]
                    second_row_types = [attrs.get('data_type') for _, attrs in second_row_cells]
                    
                    if first_row_types and all(t == 'string' for t in first_row_types) and \
                       second_row_types and any(t != 'string' for t in second_row_types):
                        # Likely headers
                        for cell_node, _ in first_row_cells:
                            self.graph.nodes[cell_node]['data_type'] = 'header'
                            
                            # Find corresponding column cells
                            col = self.graph.nodes[cell_node].get('column')
                            if col:
                                for row_idx, row_cells in table_rows[1:]:
                                    for other_node, attrs in row_cells:
                                        if attrs.get('column') == col:
                                            self.graph.add_edge(cell_node, other_node, relationship='header_for')
    
    def get_entity_subgraph(self, entity_type: str = None) -> nx.DiGraph:
        """
        Get a subgraph containing only entities of the specified type.
        
        Args:
            entity_type: Type of entities to include ('cell', 'sheet', 'table', etc.)
            
        Returns:
            nx.DiGraph: Subgraph with the specified entities
        """
        if entity_type:
            nodes = [n for n, attrs in self.graph.nodes(data=True) 
                    if attrs.get('type') == entity_type]
            return self.graph.subgraph(nodes)
        return self.graph
    
    def get_cell_subgraph(self, cell_addr: str, depth: int = 1) -> nx.DiGraph:
        """
        Get a subgraph centered on a specific cell with specified depth.
        
        Args:
            cell_addr: Cell address (e.g., 'Sheet1!A1')
            depth: How many steps to include in the neighborhood
            
        Returns:
            nx.DiGraph: Subgraph centered on the specified cell
        """
        cell_node = f"cell:{cell_addr}"
        if not self.graph.has_node(cell_node):
            return nx.DiGraph()
        
        # Get nodes within the specified depth
        nodes = set([cell_node])
        current_nodes = {cell_node}
        
        for _ in range(depth):
            next_nodes = set()
            for node in current_nodes:
                # Add predecessors
                for pred in self.graph.predecessors(node):
                    next_nodes.add(pred)
                
                # Add successors
                for succ in self.graph.successors(node):
                    next_nodes.add(succ)
            
            nodes.update(next_nodes)
            current_nodes = next_nodes
        
        return self.graph.subgraph(nodes)
    
    def get_formula_dependencies(self, cell_addr: str) -> List[str]:
        """
        Get all cells that a formula depends on.
        
        Args:
            cell_addr: Cell address with the formula
            
        Returns:
            List[str]: List of cell addresses the formula depends on
        """
        cell_node = f"cell:{cell_addr}"
        if not self.graph.has_node(cell_node):
            return []
        
        # Get all nodes that point to this cell with 'used_in' relationship
        dependencies = []
        for pred in self.graph.predecessors(cell_node):
            edge_data = self.graph.get_edge_data(pred, cell_node)
            if edge_data and edge_data.get('relationship') == 'used_in':
                if pred.startswith('cell:'):
                    dependencies.append(pred[5:])  # Remove 'cell:' prefix
        
        return dependencies
    
    def get_dependent_cells(self, cell_addr: str) -> List[str]:
        """
        Get all cells that depend on this cell.
        
        Args:
            cell_addr: Cell address
            
        Returns:
            List[str]: List of cell addresses that depend on this cell
        """
        cell_node = f"cell:{cell_addr}"
        if not self.graph.has_node(cell_node):
            return []
        
        # Get all nodes that this cell points to with 'used_in' relationship
        dependents = []
        for succ in self.graph.successors(cell_node):
            edge_data = self.graph.get_edge_data(cell_node, succ)
            if edge_data and edge_data.get('relationship') == 'used_in':
                if succ.startswith('cell:'):
                    dependents.append(succ[5:])  # Remove 'cell:' prefix
        
        return dependents
    
    def serialize(self, output_path: str) -> bool:
        """
        Serialize the knowledge graph to a JSON file.
        
        Args:
            output_path: Path to save the serialized graph
            
        Returns:
            bool: True if serialization was successful, False otherwise
        """
        try:
            # Convert to networkx JSON format
            data = nx.node_link_data(self.graph)
            
            # Add additional metadata
            data['semantic_types'] = self.semantic_types
            data['entity_map'] = self.entity_map
            data['relationship_types'] = list(self.relationship_types)
            
            # Write to file
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            logger.error(f"Error serializing knowledge graph: {e}")
            return False
    
    def load(self, input_path: str) -> bool:
        """
        Load a knowledge graph from a JSON file.
        
        Args:
            input_path: Path to load the serialized graph from
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        try:
            # Read from file
            with open(input_path, 'r') as f:
                data = json.load(f)
            
            # Convert from networkx JSON format
            self.graph = nx.node_link_graph(data)
            
            # Load additional metadata
            self.semantic_types = data.get('semantic_types', {})
            self.entity_map = data.get('entity_map', {})
            self.relationship_types = set(data.get('relationship_types', []))
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading knowledge graph: {e}")
            return False
            
    def to_cypher_statements(self) -> List[str]:
        """
        Convert the knowledge graph to Cypher statements for Neo4j import.
        
        Returns:
            List[str]: List of Cypher statements
        """
        statements = []
        
        # Create nodes
        for node, attrs in self.graph.nodes(data=True):
            # Get node type and remove it from attributes
            node_type = attrs.get('type', 'Unknown')
            attrs_copy = attrs.copy()
            if 'type' in attrs_copy:
                del attrs_copy['type']
            
            # Format attributes for Cypher
            attrs_str = ', '.join([f'n.{key} = {json.dumps(value)}' for key, value in attrs_copy.items()])
            
            # Create the node
            statements.append(f"CREATE (n:{node_type} {{id: {json.dumps(node)}}}) SET {attrs_str}")
        
        # Create edges
        for source, target, attrs in self.graph.edges(data=True):
            source_escaped = json.dumps(source)
            target_escaped = json.dumps(target)
            
            # Get relationship type
            rel_type = attrs.get('relationship', 'RELATED_TO').upper()
            
            # Format attributes for Cypher
            attrs_copy = attrs.copy()
            if 'relationship' in attrs_copy:
                del attrs_copy['relationship']
            
            if attrs_copy:
                attrs_str = ' {' + ', '.join([f'{key}: {json.dumps(value)}' for key, value in attrs_copy.items()]) + '}'
            else:
                attrs_str = ''
            
            # Create the relationship
            statements.append(f"MATCH (a), (b) WHERE a.id = {source_escaped} AND b.id = {target_escaped} CREATE (a)-[:{rel_type}{attrs_str}]->(b)")
        
        return statements