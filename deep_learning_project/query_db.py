#!/usr/bin/env python3
"""
Database Query Utility for Face Detection Results
Allows viewing and exporting training/test results from the database
"""

import argparse
import os
from database import ResultsDatabase, print_session_report


def parse_args():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_db_path = os.path.join(script_dir, 'artifacts', 'results.db')
    default_export_path = os.path.join(script_dir, 'artifacts', 'db_export.csv')
    
    parser = argparse.ArgumentParser(description='Query and view results from the database')
    parser.add_argument('--db', default=default_db_path, help='Database path')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # List all sessions
    list_parser = subparsers.add_parser('list', help='List all training sessions')
    
    # Show session details
    show_parser = subparsers.add_parser('show', help='Show detailed session report')
    show_parser.add_argument('session_id', type=int, help='Session ID to display')
    
    # Show best models
    best_parser = subparsers.add_parser('best', help='Show best models')
    best_parser.add_argument('--top', type=int, default=5, help='Number of top models to show')
    
    # Export to CSV
    export_parser = subparsers.add_parser('export', help='Export metrics to CSV')
    export_parser.add_argument('--output', default=default_export_path, help='Output CSV file')
    export_parser.add_argument('--session-id', type=int, help='Session ID to export (all if not specified)')
    
    # Show latest session
    latest_parser = subparsers.add_parser('latest', help='Show latest training session')
    
    return parser.parse_args()


def list_sessions(db_path):
    """List all training sessions"""
    with ResultsDatabase(db_path) as db:
        sessions = db.get_all_sessions()
        
        if not sessions:
            print("No training sessions found in database")
            return
        
        print("\n" + "="*100)
        print("ALL TRAINING SESSIONS")
        print("="*100)
        print(f"{'ID':<6} {'Timestamp':<20} {'Epochs':<8} {'Batch':<8} {'LR':<10} {'Valid Acc':<12} {'Test Acc':<12} {'Threshold':<10}")
        print("-"*100)
        
        for session in sessions:
            print(f"{session['session_id']:<6} "
                  f"{session['timestamp'][:19]:<20} "
                  f"{session['epochs'] or 'N/A':<8} "
                  f"{session['batch_size'] or 'N/A':<8} "
                  f"{session['learning_rate'] or 'N/A':<10.4f} "
                  f"{session['best_valid_acc'] or 'N/A':<12} "
                  f"{session['final_test_acc'] or 'N/A':<12} "
                  f"{session['threshold'] or 'N/A':<10}")
        
        print("="*100 + "\n")


def show_session(db_path, session_id):
    """Show detailed session report"""
    print_session_report(db_path, session_id)


def show_best_models(db_path, top_n=5):
    """Show top N best models"""
    with ResultsDatabase(db_path) as db:
        best_models = db.get_best_models(top_n)
        
        if not best_models:
            print("No models found in database")
            return
        
        print("\n" + "="*100)
        print(f"TOP {top_n} BEST MODELS (by validation accuracy)")
        print("="*100)
        print(f"{'Session ID':<12} {'Timestamp':<20} {'Valid Acc':<15} {'Test Acc':<15} {'Checkpoint Path':<50}")
        print("-"*100)
        
        for model in best_models:
            print(f"{model['session_id']:<12} "
                  f"{model['timestamp'][:19]:<20} "
                  f"{model['best_valid_acc']:<15.4f} "
                  f"{model['final_test_acc'] or 'N/A':<15} "
                  f"{model['checkpoint_path']:<50}")
        
        print("="*100 + "\n")


def export_to_csv(db_path, output_path, session_id=None):
    """Export metrics to CSV"""
    with ResultsDatabase(db_path) as db:
        db.export_to_csv(output_path, session_id)
        print(f"Metrics exported to: {output_path}")


def show_latest(db_path):
    """Show the latest training session"""
    with ResultsDatabase(db_path) as db:
        sessions = db.get_all_sessions()
        if not sessions:
            print("No training sessions found in database")
            return
        
        latest_session = sessions[0]  # Already sorted by timestamp DESC
        print_session_report(db_path, latest_session['session_id'])


def main():
    args = parse_args()
    
    if not args.command:
        print("No command specified. Use --help for usage information.")
        return
    
    if args.command == 'list':
        list_sessions(args.db)
    elif args.command == 'show':
        show_session(args.db, args.session_id)
    elif args.command == 'best':
        show_best_models(args.db, args.top)
    elif args.command == 'export':
        export_to_csv(args.db, args.output, args.session_id)
    elif args.command == 'latest':
        show_latest(args.db)


if __name__ == '__main__':
    main()
