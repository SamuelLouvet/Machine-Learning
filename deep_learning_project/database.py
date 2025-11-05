import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any


class ResultsDatabase:
    """Database manager for storing training results, metrics, and metadata"""
    
    def __init__(self, db_path: str = './artifacts/results.db'):
        """Initialize database connection and create tables if they don't exist"""
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
    
    def create_tables(self):
        """Create all necessary tables for storing results"""
        cursor = self.conn.cursor()
        
        # Training sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                epochs INTEGER,
                batch_size INTEGER,
                learning_rate REAL,
                optimizer TEXT,
                architecture TEXT,
                total_train_samples INTEGER,
                total_valid_samples INTEGER,
                total_test_samples INTEGER,
                config_json TEXT,
                best_epoch INTEGER,
                best_valid_acc REAL,
                final_test_acc REAL,
                threshold REAL
            )
        ''')
        
        # Training metrics per epoch
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS epoch_metrics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                epoch INTEGER,
                split TEXT,
                loss REAL,
                accuracy REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
            )
        ''')
        
        # Model checkpoints metadata
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_checkpoints (
                checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                epoch INTEGER,
                checkpoint_path TEXT,
                file_size_bytes INTEGER,
                is_best BOOLEAN,
                valid_acc REAL,
                timestamp TEXT,
                FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
            )
        ''')
        
        # Test predictions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                sample_index INTEGER,
                true_label INTEGER,
                predicted_label INTEGER,
                confidence_class0 REAL,
                confidence_class1 REAL,
                is_correct BOOLEAN,
                FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
            )
        ''')
        
        # Confusion matrix results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS confusion_matrices (
                cm_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                split TEXT,
                threshold REAL,
                true_negative INTEGER,
                false_positive INTEGER,
                false_negative INTEGER,
                true_positive INTEGER,
                timestamp TEXT,
                FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
            )
        ''')
        
        # Visualization outputs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS visualizations (
                viz_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                viz_type TEXT,
                file_path TEXT,
                timestamp TEXT,
                FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
            )
        ''')
        
        self.conn.commit()
    
    def create_training_session(self, config: Dict[str, Any]) -> int:
        """Create a new training session and return its ID"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO training_sessions 
            (timestamp, epochs, batch_size, learning_rate, optimizer, architecture, 
             config_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            config.get('epochs'),
            config.get('batch_size'),
            config.get('learning_rate'),
            config.get('optimizer', 'Adam'),
            config.get('architecture', 'CNN'),
            json.dumps(config)
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def update_training_session(self, session_id: int, updates: Dict[str, Any]):
        """Update training session with final results"""
        cursor = self.conn.cursor()
        set_clause = ', '.join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [session_id]
        cursor.execute(f'''
            UPDATE training_sessions 
            SET {set_clause}
            WHERE session_id = ?
        ''', values)
        self.conn.commit()
    
    def add_epoch_metrics(self, session_id: int, epoch: int, split: str, 
                         metrics: Dict[str, float]):
        """Store metrics for a specific epoch and split"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO epoch_metrics 
            (session_id, epoch, split, loss, accuracy, precision, recall, f1_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id, epoch, split,
            metrics.get('loss', 0.0),
            metrics.get('accuracy', 0.0),
            metrics.get('precision', 0.0),
            metrics.get('recall', 0.0),
            metrics.get('f1_score', 0.0)
        ))
        self.conn.commit()
    
    def add_model_checkpoint(self, session_id: int, epoch: int, 
                            checkpoint_path: str, is_best: bool = False,
                            valid_acc: float = 0.0):
        """Store model checkpoint metadata"""
        cursor = self.conn.cursor()
        file_size = Path(checkpoint_path).stat().st_size if Path(checkpoint_path).exists() else 0
        cursor.execute('''
            INSERT INTO model_checkpoints 
            (session_id, epoch, checkpoint_path, file_size_bytes, is_best, 
             valid_acc, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id, epoch, checkpoint_path, file_size, is_best,
            valid_acc, datetime.now().isoformat()
        ))
        self.conn.commit()
    
    def add_test_predictions(self, session_id: int, predictions: List[Dict[str, Any]]):
        """Store test predictions in bulk"""
        cursor = self.conn.cursor()
        data = [
            (
                session_id,
                pred['sample_index'],
                pred['true_label'],
                pred['predicted_label'],
                pred['confidence_class0'],
                pred['confidence_class1'],
                pred['true_label'] == pred['predicted_label']
            )
            for pred in predictions
        ]
        cursor.executemany('''
            INSERT INTO test_predictions 
            (session_id, sample_index, true_label, predicted_label, 
             confidence_class0, confidence_class1, is_correct)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', data)
        self.conn.commit()
    
    def add_confusion_matrix(self, session_id: int, split: str, 
                            cm: List[List[int]], threshold: float = 0.5):
        """Store confusion matrix results"""
        cursor = self.conn.cursor()
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        cursor.execute('''
            INSERT INTO confusion_matrices 
            (session_id, split, threshold, true_negative, false_positive, 
             false_negative, true_positive, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id, split, threshold, tn, fp, fn, tp,
            datetime.now().isoformat()
        ))
        self.conn.commit()
    
    def add_visualization(self, session_id: int, viz_type: str, file_path: str):
        """Store visualization file metadata"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO visualizations 
            (session_id, viz_type, file_path, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (session_id, viz_type, file_path, datetime.now().isoformat()))
        self.conn.commit()
    
    def get_training_history(self, session_id: Optional[int] = None) -> List[Dict]:
        """Get training history for a session or all sessions"""
        cursor = self.conn.cursor()
        if session_id:
            cursor.execute('''
                SELECT * FROM epoch_metrics 
                WHERE session_id = ? 
                ORDER BY epoch, split
            ''', (session_id,))
        else:
            cursor.execute('SELECT * FROM epoch_metrics ORDER BY session_id, epoch, split')
        
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_session_summary(self, session_id: int) -> Dict:
        """Get comprehensive summary of a training session"""
        cursor = self.conn.cursor()
        
        # Get session info
        cursor.execute('SELECT * FROM training_sessions WHERE session_id = ?', (session_id,))
        session_row = cursor.fetchone()
        if not session_row:
            return {}
        
        columns = [desc[0] for desc in cursor.description]
        session_info = dict(zip(columns, session_row))
        
        # Get epoch metrics
        cursor.execute('''
            SELECT epoch, split, loss, accuracy, f1_score 
            FROM epoch_metrics 
            WHERE session_id = ? 
            ORDER BY epoch, split
        ''', (session_id,))
        session_info['metrics'] = cursor.fetchall()
        
        # Get checkpoints
        cursor.execute('''
            SELECT epoch, checkpoint_path, is_best, valid_acc 
            FROM model_checkpoints 
            WHERE session_id = ?
        ''', (session_id,))
        session_info['checkpoints'] = cursor.fetchall()
        
        # Get confusion matrices
        cursor.execute('''
            SELECT split, threshold, true_negative, false_positive, 
                   false_negative, true_positive 
            FROM confusion_matrices 
            WHERE session_id = ?
        ''', (session_id,))
        session_info['confusion_matrices'] = cursor.fetchall()
        
        return session_info
    
    def get_all_sessions(self) -> List[Dict]:
        """Get summary of all training sessions"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT session_id, timestamp, epochs, batch_size, learning_rate,
                   best_valid_acc, final_test_acc, threshold
            FROM training_sessions 
            ORDER BY timestamp DESC
        ''')
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_best_models(self, top_n: int = 5) -> List[Dict]:
        """Get top N best models by validation accuracy"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT ts.session_id, ts.timestamp, ts.best_valid_acc, 
                   ts.final_test_acc, mc.checkpoint_path
            FROM training_sessions ts
            JOIN model_checkpoints mc ON ts.session_id = mc.session_id
            WHERE mc.is_best = 1
            ORDER BY ts.best_valid_acc DESC
            LIMIT ?
        ''', (top_n,))
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def export_to_csv(self, output_path: str, session_id: Optional[int] = None):
        """Export metrics to CSV file"""
        import csv
        
        history = self.get_training_history(session_id)
        if not history:
            return
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
    
    def close(self):
        """Close database connection"""
        self.conn.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


def print_session_report(db_path: str, session_id: int):
    """Print a formatted report for a specific session"""
    with ResultsDatabase(db_path) as db:
        summary = db.get_session_summary(session_id)
        
        if not summary:
            print(f"Session {session_id} not found")
            return
        
        print("\n" + "="*70)
        print(f"TRAINING SESSION REPORT - ID: {session_id}")
        print("="*70)
        print(f"\nTimestamp: {summary['timestamp']}")
        print(f"Epochs: {summary['epochs']}")
        print(f"Batch Size: {summary['batch_size']}")
        print(f"Learning Rate: {summary['learning_rate']}")
        print(f"Best Validation Accuracy: {summary.get('best_valid_acc', 'N/A')}")
        print(f"Final Test Accuracy: {summary.get('final_test_acc', 'N/A')}")
        print(f"Optimal Threshold: {summary.get('threshold', 'N/A')}")
        
        if summary.get('metrics'):
            print("\n" + "-"*70)
            print("EPOCH METRICS:")
            print("-"*70)
            print(f"{'Epoch':<8} {'Split':<8} {'Loss':<10} {'Accuracy':<10} {'F1-Score':<10}")
            print("-"*70)
            for epoch, split, loss, acc, f1 in summary['metrics']:
                print(f"{epoch:<8} {split:<8} {loss:<10.4f} {acc:<10.4f} {f1:<10.4f}")
        
        if summary.get('confusion_matrices'):
            print("\n" + "-"*70)
            print("CONFUSION MATRICES:")
            print("-"*70)
            for split, thresh, tn, fp, fn, tp in summary['confusion_matrices']:
                print(f"\n{split.upper()} (threshold={thresh:.3f}):")
                print(f"  TN: {tn:>6}  FP: {fp:>6}")
                print(f"  FN: {fn:>6}  TP: {tp:>6}")
        
        print("\n" + "="*70 + "\n")
