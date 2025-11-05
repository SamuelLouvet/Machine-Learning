#!/usr/bin/env python3
"""
SQL Query Viewer - Interactive database exploration
"""

import sqlite3
import sys
from pathlib import Path


def execute_query(db_path, query, description=""):
    """Execute a SQL query and display results in a formatted table"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        
        if description:
            print(f"\n{'='*100}")
            print(f"{description}")
            print('='*100)
        
        if not results:
            print("No results found.")
            return
        
        # Calculate column widths
        col_widths = [len(col) for col in columns]
        for row in results:
            for i, val in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(val)))
        
        # Print header
        header = " | ".join(col.ljust(col_widths[i]) for i, col in enumerate(columns))
        print("\n" + header)
        print("-" * len(header))
        
        # Print rows
        for row in results:
            print(" | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row)))
        
        print(f"\nTotal rows: {len(results)}\n")
        
    except sqlite3.Error as e:
        print(f"SQL Error: {e}")
    finally:
        conn.close()


def show_schema(db_path):
    """Show database schema"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("\n" + "="*100)
    print("DATABASE SCHEMA")
    print("="*100)
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = cursor.fetchall()
    
    for (table_name,) in tables:
        print(f"\n📊 Table: {table_name}")
        print("-" * 100)
        
        # Get table schema
        cursor.execute(f"PRAGMA table_info({table_name})")
        schema = cursor.fetchall()
        
        print(f"{'Column':<30} {'Type':<15} {'Not Null':<10} {'Default':<15} {'Primary Key'}")
        print("-" * 100)
        for col in schema:
            cid, name, type_, notnull, default, pk = col
            print(f"{name:<30} {type_:<15} {str(bool(notnull)):<10} {str(default):<15} {bool(pk)}")
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"\nRows in table: {count}")
    
    conn.close()


def main():
    db_path = './deep_learning_project/artifacts/results.db'
    
    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        print("Run training first to create the database")
        return
    
    print(f"\n{'='*100}")
    print(f"SQL DATABASE VIEWER - {db_path}")
    print('='*100)
    
    # Show schema
    show_schema(db_path)
    
    # Example queries
    queries = [
        ("SELECT * FROM training_sessions", 
         "ALL TRAINING SESSIONS"),
        
        ("SELECT session_id, epoch, split, loss, accuracy, precision, recall, f1_score FROM epoch_metrics ORDER BY epoch, split LIMIT 15", 
         "EPOCH METRICS (First 15 rows)"),
        
        ("SELECT session_id, epoch, split, accuracy, f1_score FROM epoch_metrics WHERE split='test'", 
         "TEST SET METRICS"),
        
        ("SELECT * FROM model_checkpoints", 
         "MODEL CHECKPOINTS"),
        
        ("SELECT * FROM confusion_matrices", 
         "CONFUSION MATRICES"),
        
        ("SELECT session_id, viz_type, file_path FROM visualizations", 
         "VISUALIZATIONS"),
        
        ("SELECT session_id, COUNT(*) as prediction_count FROM test_predictions GROUP BY session_id", 
         "TEST PREDICTIONS COUNT BY SESSION"),
        
        ("SELECT true_label, predicted_label, COUNT(*) as count FROM test_predictions GROUP BY true_label, predicted_label", 
         "PREDICTION DISTRIBUTION"),
        
        ("SELECT split, AVG(accuracy) as avg_accuracy, AVG(f1_score) as avg_f1 FROM epoch_metrics GROUP BY split", 
         "AVERAGE METRICS BY SPLIT"),
        
        ("SELECT epoch, MAX(CASE WHEN split='train' THEN accuracy END) as train_acc, MAX(CASE WHEN split='valid' THEN accuracy END) as valid_acc FROM epoch_metrics WHERE split IN ('train', 'valid') GROUP BY epoch ORDER BY epoch", 
         "TRAINING PROGRESS (Train vs Valid Accuracy)"),
    ]
    
    for query, description in queries:
        execute_query(db_path, query, description)
    
    print("\n" + "="*100)
    print("CUSTOM SQL QUERIES")
    print("="*100)
    print("\nYou can run custom queries using Python:")
    print("""
import sqlite3
conn = sqlite3.connect('./deep_learning_project/artifacts/demo_results.db')
cursor = conn.cursor()
cursor.execute("YOUR SQL QUERY HERE")
results = cursor.fetchall()
conn.close()
""")
    print("="*100 + "\n")


if __name__ == '__main__':
    main()
