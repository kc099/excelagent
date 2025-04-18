import argparse
import logging
import os
import sys
import json
from typing import Dict, List, Any, Optional
import time

# Import custom modules
from excel_parser import ExcelParser
from knowledge_graph import ExcelKnowledgeGraph
from semantic_analyzer import SemanticAnalyzer
from llm_agent_interface import LLMAgentInterface
from query_processor import QueryProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("excel_kg.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class ExcelKnowledgeGraphApp:
    """
    Main application for Excel Knowledge Graph Parser.
    """
    def __init__(self):
        """Initialize the application."""
        self.parser = None
        self.kg = None
        self.semantic_analyzer = None
        self.agent_interface = None
        self.query_processor = None
        self.file_path = None
        self.output_dir = None
        
    def initialize(self, file_path: str, output_dir: str = None) -> bool:
        """
        Initialize the application with an Excel file.
        
        Args:
            file_path: Path to the Excel file
            output_dir: Directory to store output files
            
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        self.file_path = file_path
        self.output_dir = output_dir or os.path.dirname(file_path)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize Excel parser
        logger.info(f"Initializing Excel parser for {file_path}")
        self.parser = ExcelParser(file_path)
        
        # Parse the Excel file
        if not self.parser.load_workbook():
            logger.error(f"Failed to load workbook: {file_path}")
            return False
        
        # Parse all sheets
        logger.info("Parsing sheets")
        self.parser.parse_all_sheets()
        
        # Initialize knowledge graph
        logger.info("Initializing knowledge graph")
        self.kg = ExcelKnowledgeGraph()
        
        # Build knowledge graph from parser
        logger.info("Building knowledge graph")
        if not self.kg.build_from_parser(self.parser):
            logger.error("Failed to build knowledge graph")
            return False
        
        # Initialize semantic analyzer
        logger.info("Initializing semantic analyzer")
        self.semantic_analyzer = SemanticAnalyzer(self.kg)
        
        # Perform semantic analysis
        logger.info("Performing semantic analysis")
        if not self.semantic_analyzer.analyze():
            logger.warning("Semantic analysis encountered issues")
        
        # Enrich the graph with semantic information
        logger.info("Enriching knowledge graph with semantic information")
        self.semantic_analyzer.enrich_graph()
        
        # Initialize agent interface
        logger.info("Initializing LLM agent interface")
        self.agent_interface = LLMAgentInterface(self.kg, self.semantic_analyzer)
        
        # Initialize query processor
        logger.info("Initializing query processor")
        self.query_processor = QueryProcessor(self.agent_interface)
        
        logger.info("Initialization complete")
        return True
    
    def save_knowledge_graph(self, output_path: Optional[str] = None) -> str:
        """
        Save the knowledge graph to a file.
        
        Args:
            output_path: Optional path to save the knowledge graph
            
        Returns:
            str: Path where the knowledge graph was saved
        """
        if output_path is None:
            # Generate a default output path based on the input file
            base_name = os.path.splitext(os.path.basename(self.file_path))[0]
            output_path = os.path.join(self.output_dir, f"{base_name}_kg.json")
        
        logger.info(f"Saving knowledge graph to {output_path}")
        self.kg.serialize(output_path)
        
        return output_path
    
    def load_knowledge_graph(self, input_path: str) -> bool:
        """
        Load a knowledge graph from a file.
        
        Args:
            input_path: Path to the knowledge graph file
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        logger.info(f"Loading knowledge graph from {input_path}")
        
        if self.kg is None:
            self.kg = ExcelKnowledgeGraph()
        
        if not self.kg.load(input_path):
            logger.error(f"Failed to load knowledge graph from {input_path}")
            return False
        
        # Initialize semantic analyzer
        self.semantic_analyzer = SemanticAnalyzer(self.kg)
        
        # Initialize agent interface
        self.agent_interface = LLMAgentInterface(self.kg, self.semantic_analyzer)
        
        # Initialize query processor
        self.query_processor = QueryProcessor(self.agent_interface)
        
        logger.info("Knowledge graph loaded successfully")
        return True
    
    def process_query(self, query_text: str) -> Dict[str, Any]:
        """
        Process a natural language query.
        
        Args:
            query_text: Natural language query
            
        Returns:
            Dict[str, Any]: Query response
        """
        if self.query_processor is None:
            logger.error("Query processor not initialized")
            return {
                'status': 'error',
                'message': 'System not initialized'
            }
        
        logger.info(f"Processing query: {query_text}")
        start_time = time.time()
        
        response = self.query_processor.process_query(query_text)
        
        # Add suggestions for follow-up questions
        response['suggestions'] = self.query_processor.suggest_follow_up_questions(
            query_text, response
        )
        
        end_time = time.time()
        response['processing_time'] = end_time - start_time
        
        logger.info(f"Query processed in {response['processing_time']:.2f} seconds")
        return response
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get a schema of the Excel data.
        
        Returns:
            Dict[str, Any]: Schema information
        """
        if self.agent_interface is None:
            logger.error("Agent interface not initialized")
            return {
                'status': 'error',
                'message': 'System not initialized'
            }
        
        logger.info("Retrieving schema")
        return self.agent_interface.get_schema()
    
    def get_semantic_summary(self) -> Dict[str, Any]:
        """
        Get a semantic summary of the Excel data.
        
        Returns:
            Dict[str, Any]: Semantic summary
        """
        if self.semantic_analyzer is None:
            logger.error("Semantic analyzer not initialized")
            return {
                'status': 'error',
                'message': 'Semantic analyzer not initialized'
            }
        
        logger.info("Retrieving semantic summary")
        return self.semantic_analyzer.get_semantic_summary()
    
    def generate_cypher(self, output_path: Optional[str] = None) -> str:
        """
        Generate Cypher statements for Neo4j import.
        
        Args:
            output_path: Optional path to save the Cypher statements
            
        Returns:
            str: Path where the Cypher statements were saved
        """
        if self.kg is None:
            logger.error("Knowledge graph not initialized")
            return ""
        
        if output_path is None:
            # Generate a default output path
            base_name = os.path.splitext(os.path.basename(self.file_path))[0]
            output_path = os.path.join(self.output_dir, f"{base_name}_cypher.cql")
        
        logger.info(f"Generating Cypher statements to {output_path}")
        statements = self.kg.to_cypher_statements()
        
        with open(output_path, 'w') as f:
            for statement in statements:
                f.write(f"{statement};\n")
        
        return output_path
    
    def run_cli(self):
        """Run the interactive command-line interface."""
        print("\nExcel Knowledge Graph Parser CLI")
        print("================================\n")
        
        # Load or parse Excel file
        while True:
            action = input("Do you want to (1) parse a new Excel file or (2) load an existing knowledge graph? ").strip()
            
            if action == '1':
                file_path = input("Enter the path to the Excel file: ").strip()
                
                if not os.path.exists(file_path):
                    print(f"Error: File not found: {file_path}")
                    continue
                
                output_dir = input("Enter output directory (press Enter for default): ").strip()
                if not output_dir:
                    output_dir = None
                
                print("\nInitializing system...")
                if not self.initialize(file_path, output_dir):
                    print("Initialization failed. Check the logs for details.")
                    continue
                
                break
                
            elif action == '2':
                kg_path = input("Enter the path to the knowledge graph file: ").strip()
                
                if not os.path.exists(kg_path):
                    print(f"Error: File not found: {kg_path}")
                    continue
                
                print("\nLoading knowledge graph...")
                if not self.load_knowledge_graph(kg_path):
                    print("Loading failed. Check the logs for details.")
                    continue
                
                break
                
            else:
                print("Invalid option. Please enter 1 or 2.")
        
        # Save the knowledge graph
        save_kg = input("\nDo you want to save the knowledge graph? (y/n): ").strip().lower()
        if save_kg == 'y':
            kg_path = input("Enter output path (press Enter for default): ").strip()
            kg_path = self.save_knowledge_graph(kg_path if kg_path else None)
            print(f"Knowledge graph saved to: {kg_path}")
        
        # Generate Cypher statements
        gen_cypher = input("\nDo you want to generate Cypher statements for Neo4j? (y/n): ").strip().lower()
        if gen_cypher == 'y':
            cypher_path = input("Enter output path (press Enter for default): ").strip()
            cypher_path = self.generate_cypher(cypher_path if cypher_path else None)
            print(f"Cypher statements saved to: {cypher_path}")
        
        # Show schema
        show_schema = input("\nDo you want to see the schema? (y/n): ").strip().lower()
        if show_schema == 'y':
            schema = self.get_schema()
            print("\nSchema:")
            print(json.dumps(schema, indent=2))
        
        # Show semantic summary
        show_summary = input("\nDo you want to see the semantic summary? (y/n): ").strip().lower()
        if show_summary == 'y':
            summary = self.get_semantic_summary()
            print("\nSemantic Summary:")
            print(json.dumps(summary, indent=2))
        
        # Interactive query loop
        print("\nEnter queries below or type 'exit' to quit.")
        print("Example queries:")
        print("  - What sheets are in this workbook?")
        print("  - What formulas are used in this workbook?")
        print("  - What is the value in cell A1?")
        print("  - What is the structure of the data?")
        print("  - What is the domain of this data?\n")
        
        while True:
            query = input("\nQuery: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                break
            
            if not query:
                continue
            
            response = self.process_query(query)
            
            # Print natural language answer
            if 'nl_answer' in response:
                print(f"\nAnswer: {response['nl_answer']}")
            
            # Print suggestions
            if 'suggestions' in response:
                print("\nSuggested follow-up questions:")
                for i, suggestion in enumerate(response['suggestions'], 1):
                    print(f"  {i}. {suggestion}")
            
            # Ask if user wants to see full details
            show_details = input("\nDo you want to see full details? (y/n): ").strip().lower()
            if show_details == 'y':
                # Remove large data structures to make output more readable
                if 'cells' in response and len(response['cells']) > 5:
                    response['cells'] = response['cells'][:5]
                    response['cells'].append("... more cells (truncated)")
                
                print("\nFull Response:")
                print(json.dumps(response, indent=2, default=str))
        
        print("\nThank you for using Excel Knowledge Graph Parser!")

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description='Excel Knowledge Graph Parser')
    parser.add_argument('--file', '-f', help='Path to the Excel file')
    parser.add_argument('--output', '-o', help='Output directory')
    parser.add_argument('--load', '-l', help='Load an existing knowledge graph file')
    parser.add_argument('--save', '-s', help='Save the knowledge graph to a file')
    parser.add_argument('--cypher', '-c', help='Generate Cypher statements for Neo4j')
    parser.add_argument('--query', '-q', help='Process a natural language query')
    parser.add_argument('--schema', action='store_true', help='Print the schema')
    parser.add_argument('--summary', action='store_true', help='Print the semantic summary')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    app = ExcelKnowledgeGraphApp()
    
    if args.interactive:
        app.run_cli()
        return
    
    # Load or parse Excel file
    if args.load:
        if not app.load_knowledge_graph(args.load):
            logger.error("Failed to load knowledge graph")
            return
    elif args.file:
        if not app.initialize(args.file, args.output):
            logger.error("Failed to initialize application")
            return
    else:
        logger.error("Either --file or --load must be specified")
        parser.print_help()
        return
    
    # Save knowledge graph
    if args.save:
        saved_path = app.save_knowledge_graph(args.save)
        logger.info(f"Knowledge graph saved to {saved_path}")
    
    # Generate Cypher statements
    if args.cypher:
        cypher_path = app.generate_cypher(args.cypher)
        logger.info(f"Cypher statements saved to {cypher_path}")
    
    # Print schema
    if args.schema:
        schema = app.get_schema()
        print(json.dumps(schema, indent=2))
    
    # Print semantic summary
    if args.summary:
        summary = app.get_semantic_summary()
        print(json.dumps(summary, indent=2))
    
    # Process query
    if args.query:
        response = app.process_query(args.query)
        print(json.dumps(response, indent=2, default=str))

if __name__ == '__main__':
    main()