#!/usr/bin/env python3
"""
Database module for Demokratibasen data.
Loads CSV and JSON files into a SQLite database with proper indexing and relationships.
"""

import csv
import glob
import json
import logging
import os
from pathlib import Path
import sys
from typing import List, Dict, Any, Optional

import sqlite3
import pandas as pd


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global parameter for testing
TEST_LINES = 3

SOURCE_FILES = [
    'sources/36812-demokratibasen-urls-20250528.csv',
    'sources/36812-demokratibasen-texts-20250528.jsonl',
    'sources/17569-demokratibasen-inferences-20250624.csv',
    'sources/42254-demokratibasen-urls-20250811.csv',
    'sources/111721-demokratibasen-test-urls-20250920.csv',
    'sources/12243-demokratibasen-uib-ide-texts-20250920.csv',
    'sources/12243-demokratibasen-uib-ide-urls-20250920.csv',
    'sources/29602-demokratibasen-test-inferences-20250920.csv',
    'sources/6100-demokratibasen-uib-ide-inferences-20250920.csv',
    'sources/111188-demokratibasen-prod-urls-20250921.csv',
    'sources/29281-demokratibasen-prod-inferences-20250921.csv',
    'sources/103908-dokumenter-texts-20250921.jsonl',
]

class TrainingDataDatabase:
    """Simple SQLite database for training data."""
    
    def __init__(self, db_path: str = "training_data.db"):
        """Initialize database connection and create tables if they don't exist."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # Enable dict-like access
        
        # Ensure tables exist and are on the current schema (table name 'dokumenter')
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='dokumenter'")
        has_dokumenter = cursor.fetchone() is not None
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents'")
        has_documents = cursor.fetchone() is not None
        if not has_dokumenter and not has_documents:
            self._create_tables()
            logger.info("Database tables created successfully")
        else:
            logger.info("Using existing database")
            # Run migration/rename if needed
            self._migrate_add_summary_title()
            # Ensure required columns exist (not only during migration)
            try:
                cursor.execute("PRAGMA table_info(dokumenter)")
                cols = [col[1] for col in cursor.fetchall()]
                # If a temporary 'tittel' exists (from earlier change), fold it back into dok_tittel
                if 'tittel' in cols and 'dok_tittel' in cols:
                    logger.info("Normalizing title columns: copying non-null 'tittel' into 'dok_tittel' where missing, then dropping 'tittel'.")
                    cursor.execute("UPDATE dokumenter SET dok_tittel = COALESCE(dok_tittel, tittel)")
                    try:
                        cursor.execute("ALTER TABLE dokumenter DROP COLUMN tittel")
                    except Exception:
                        # SQLite before 3.35 doesn't support DROP COLUMN; leave it if unsupported
                        logger.info("SQLite does not support DROP COLUMN; keeping 'tittel' as a no-op column.")
                    self.conn.commit()
                # Ensure dok_tekst exists; backfill from legacy doc_tekst
                cursor.execute("PRAGMA table_info(dokumenter)")
                cols = [col[1] for col in cursor.fetchall()]
                if 'dok_tekst' not in cols and 'doc_tekst' in cols:
                    logger.info("Adding 'dok_tekst' to 'dokumenter' and backfilling from 'doc_tekst'.")
                    cursor.execute("ALTER TABLE dokumenter ADD COLUMN dok_tekst TEXT")
                    cursor.execute("UPDATE dokumenter SET dok_tekst = doc_tekst WHERE dok_tekst IS NULL")
                    self.conn.commit()
            except Exception as e:
                logger.warning(f"Column normalization skipped: {e}")
    
    def _create_tables(self):
        """Create database tables with proper schema."""
        cursor = self.conn.cursor()
        
        # Main table - combines all data sources
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dokumenter (
                dok_id TEXT PRIMARY KEY,
                dok_type TEXT,
                kommune INTEGER,
                dok_tittel TEXT,
                url TEXT,
                dok_tekst TEXT,
                oppsum_tittel TEXT,
                oppsummering TEXT,
                personer TEXT,
                nokkelord TEXT,
                nyhetsverdi INTEGER,
                url_fil TEXT,
                tekst_fil TEXT,
                oppsum_fil TEXT,
                modell TEXT,
                batch_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_dokumenter_kommune ON dokumenter(kommune)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_dokumenter_doc_type ON dokumenter(dok_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_dokumenter_batch_id ON dokumenter(batch_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_dokumenter_url ON dokumenter(url)")
        
        # Metadata table for tracking loaded files
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_metadata (
                file_path TEXT PRIMARY KEY,
                file_type TEXT,
                record_count INTEGER,
                last_loaded TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.commit()
        logger.info("Database tables created successfully")
    
    def _migrate_add_summary_title(self):
        """Legacy no-op: kept for compatibility with older runs."""
        cursor = self.conn.cursor()
        # If legacy 'documents' table exists, migrate/rename to 'dokumenter'
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents'")
        has_documents = cursor.fetchone() is not None
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='dokumenter'")
        has_dokumenter = cursor.fetchone() is not None
        if not has_documents:
            return
        logger.info("Migrating legacy table 'documents' to 'dokumenter' with dok_* columns...")
        cursor.execute("PRAGMA table_info(documents)")
        legacy_columns = [col[1] for col in cursor.fetchall()]
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dokumenter_new (
                dok_id TEXT PRIMARY KEY,
                dok_type TEXT,
                kommune INTEGER,
                dok_tittel TEXT,
                url TEXT,
                dok_tekst TEXT,
                oppsum_tittel TEXT,
                oppsummering TEXT,
                personer TEXT,
                nokkelord TEXT,
                nyhetsverdi INTEGER,
                url_fil TEXT,
                tekst_fil TEXT,
                oppsum_fil TEXT,
                modell TEXT,
                batch_id TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        """)
        if 'dok_id' not in legacy_columns:
            cursor.execute("""
                INSERT INTO dokumenter_new (
                    dok_id, dok_type, kommune, dok_tittel, url, dok_tekst, oppsum_tittel, oppsummering,
                    personer, nokkelord, nyhetsverdi, url_fil, tekst_fil, oppsum_fil, modell, batch_id, created_at, updated_at
                )
                SELECT 
                    dokument_id as dok_id,
                    doc_type as dok_type,
                    kommune,
                    tittel as dok_tittel,
                    url,
                    doc_tekst as dok_tekst,
                    summary_title as oppsum_tittel,
                    oppsummering,
                    personer,
                    nokkelord,
                    nyhetsverdi,
                    CASE WHEN source_file LIKE '%urls%' THEN source_file END as url_fil,
                    CASE WHEN source_file LIKE '%texts%' OR source_file LIKE '%.jsonl' THEN source_file END as tekst_fil,
                    CASE WHEN source_file LIKE '%inferences%' THEN source_file END as oppsum_fil,
                    NULL as modell,
                    batch_id,
                    created_at,
                    updated_at
                FROM documents
            """)
        else:
            cursor.execute("""
                INSERT INTO dokumenter_new (
                    dok_id, dok_type, kommune, dok_tittel, url, dok_tekst, oppsum_tittel, oppsummering,
                    personer, nokkelord, nyhetsverdi, url_fil, tekst_fil, oppsum_fil, modell, batch_id, created_at, updated_at
                )
                SELECT 
                    dok_id,
                    dok_type,
                    kommune,
                    COALESCE(dok_tittel, tittel) as dok_tittel,
                    url,
                    COALESCE(dok_tekst, doc_tekst) as dok_tekst,
                    oppsum_tittel,
                    oppsummering,
                    personer,
                    nokkelord,
                    nyhetsverdi,
                    url_fil,
                    tekst_fil,
                    oppsum_fil,
                    NULL as modell,
                    batch_id,
                    created_at,
                    updated_at
                FROM documents
            """)

        cursor.execute("DROP TABLE documents")
        if has_dokumenter:
            cursor.execute("DROP TABLE dokumenter")
        cursor.execute("ALTER TABLE dokumenter_new RENAME TO dokumenter")
        # Recreate indexes on dokumenter
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_dokumenter_kommune ON dokumenter(kommune)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_dokumenter_doc_type ON dokumenter(dok_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_dokumenter_batch_id ON dokumenter(batch_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_dokumenter_url ON dokumenter(url)")
        self.conn.commit()
        logger.info("Migration completed.")
        # Ensure 'tittel' column exists on 'dokumenter' and backfill from legacy 'dok_tittel'
        try:
            cursor.execute("PRAGMA table_info(dokumenter)")
            cols = [col[1] for col in cursor.fetchall()]
            if 'tittel' not in cols and 'dok_tittel' in cols:
                logger.info("Adding 'tittel' column to 'dokumenter' and backfilling from 'dok_tittel'...")
                cursor.execute("ALTER TABLE dokumenter ADD COLUMN tittel TEXT")
                cursor.execute("UPDATE dokumenter SET tittel = dok_tittel WHERE tittel IS NULL")
                self.conn.commit()
        except Exception as e:
            logger.warning(f"Post-migration title normalization skipped: {e}")
        # Ensure 'dok_tekst' exists; backfill from legacy 'doc_tekst'
        try:
            cursor.execute("PRAGMA table_info(dokumenter)")
            cols = [col[1] for col in cursor.fetchall()]
            if 'dok_tekst' not in cols and 'doc_tekst' in cols:
                logger.info("Adding 'dok_tekst' column to 'dokumenter' and backfilling from 'doc_tekst'...")
                cursor.execute("ALTER TABLE dokumenter ADD COLUMN dok_tekst TEXT")
                cursor.execute("UPDATE dokumenter SET dok_tekst = doc_tekst WHERE dok_tekst IS NULL")
                self.conn.commit()
        except Exception as e:
            logger.warning(f"Post-migration text normalization skipped: {e}")
    
    def load_csv_file(self, file_path: str, testing: bool = False) -> int:
        """Load data from a CSV file into the database."""
        try:
            logger.info(f"Loading CSV file: {file_path} (testing={testing})")
            
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # In testing mode, only process first TEST_LINES
            if testing:
                df = df.head(TEST_LINES)
                logger.info(f"Testing mode: processing only first {TEST_LINES} records")
            else:
                logger.info(f"CSV file contains {len(df)} records")
            
            # Determine file type based on filename
            filename = os.path.basename(file_path)
            if 'urls' in filename:
                file_type = 'urls'
            elif 'texts' in filename:
                file_type = 'texts'
            elif 'inferences' in filename:
                file_type = 'inferences'
            else:
                file_type = 'unknown'
            
            # Prepare data for insertion
            records_inserted = 0
            
            # In testing mode, clear existing data from this file first
            if testing:
                cursor = self.conn.cursor()
                cursor.execute("DELETE FROM dokumenter WHERE url_fil = ? OR tekst_fil = ? OR oppsum_fil = ?", (file_path, file_path, file_path))
                self.conn.commit()
                logger.info(f"Cleared existing data for {file_path} in testing mode")
            
            for _, row in df.iterrows():
                # Handle different CSV structures
                if file_type == 'urls':
                    # URLs CSV: dokument_id, doc_type, kommune, tittel, url
                    data = {
                        'dokument_id': row.get('dokument_id'),
                        'doc_type': row.get('doc_type'),
                        'kommune': row.get('kommune'),
                        'tittel': row.get('tittel'),
                        'url': row.get('url'),
                        'source_file': file_path
                    }
                elif file_type == 'texts':
                    # Texts CSV: dokument_id, doc_type, kommune, tittel, url, doc_tekst
                    data = {
                        'dokument_id': row.get('dokument_id'),
                        'dok_type': row.get('doc_type'),
                        'kommune': row.get('kommune'),
                        'dok_tittel': row.get('tittel'),
                        'url': row.get('url'),
                        'dok_tekst': row.get('doc_tekst', ''),
                        'tekst_fil': file_path
                    }
                elif file_type == 'inferences':
                    # Inferences CSV: dokument_id, batch_id, tittel, oppsummering, personer, nokkelord, nyhetsverdi
                    data = {
                        'dokument_id': row.get('dokument_id'),
                        'batch_id': row.get('batch_id'),
                        'summary_title': row.get('tittel'),  # Use summary_title for inference files
                        'oppsummering': row.get('oppsummering'),
                        'personer': row.get('personer'),
                        'nokkelord': row.get('nokkelord'),
                        'nyhetsverdi': row.get('nyhetsverdi'),
                        'source_file': file_path
                    }
                else:
                    # Generic handling for unknown CSV structure
                    data = {
                        'dokument_id': row.get('dokument_id'),
                        'source_file': file_path
                    }
                    # Add any other columns that exist
                    for col in df.columns:
                        if col not in ['dokument_id']:
                            data[col.lower()] = row.get(col)
                
                # Insert or update record
                if self._upsert_document(data):
                    records_inserted += 1
            
            # Update file metadata
            self._update_file_metadata(file_path, file_type, records_inserted)
            
            logger.info(f"Successfully loaded {records_inserted} records from {file_path}")
            return records_inserted
            
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {e}")
            return 0
    
    def load_jsonl_file(self, file_path: str, testing: bool = False) -> int:
        """Load data from a JSONL file into the database.

        Mapping: dokument_id -> dok_id, doc_type -> dok_type, doc_tekst -> dok_tekst (fallback from 'tekst').
        Warn about inconsistencies against existing DB rows.
        """
        try:
            logger.info(f"Loading JSONL file: {file_path} (testing={testing})")
            
            records_inserted = 0
            
            # In testing mode, clear existing data from this file first
            if testing:
                cursor = self.conn.cursor()
                cursor.execute("DELETE FROM dokumenter WHERE tekst_fil = ?", (file_path,))
                self.conn.commit()
                logger.info(f"Cleared existing data for {file_path} in testing mode")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    # In testing mode, only process first TEST_LINES
                    if testing and line_num > TEST_LINES:
                        break
                        
                    try:
                        data = json.loads(line.strip())
                        
                        # Prepare data for insertion
                        doc_data = {
                            'dokument_id': data.get('dokument_id'),
                            'dok_type': data.get('doc_type'),
                            'kommune': data.get('kommune'),
                            'dok_tittel': data.get('tittel'),
                            'url': data.get('url'),
                            'dok_tekst': data.get('doc_tekst', data.get('tekst', '')),
                            'tekst_fil': file_path
                        }
                        # Warn about inconsistencies if record exists
                        cursor = self.conn.cursor()
                        if doc_data['dokument_id']:
                            cursor.execute("SELECT dok_type, kommune, dok_tittel, url, dok_tekst FROM dokumenter WHERE dok_id = ?", (doc_data['dokument_id'],))
                            rec = cursor.fetchone()
                            if rec:
                                db_dok_type, db_kommune, db_dok_tittel, db_url, db_dok_tekst = rec
                                checks = [
                                    ('dok_type', db_dok_type, doc_data.get('dok_type')),
                                    ('kommune', db_kommune, doc_data.get('kommune')),
                                    ('dok_tittel', db_dok_tittel, doc_data.get('dok_tittel')),
                                    ('url', db_url, doc_data.get('url')),
                                    ('dok_tekst', db_dok_tekst, doc_data.get('dok_tekst')),
                                ]
                                for field, db_val, file_val in checks:
                                    if db_val is not None and file_val is not None and db_val != file_val:
                                        logger.warning(
                                            f"Inconsistency for dok_id={doc_data['dokument_id']}: field '{field}' existing='{db_val}' vs new='{file_val}' (file {os.path.basename(file_path)})"
                                        )
                        
                        # Insert or update record
                        if self._upsert_document(doc_data):
                            records_inserted += 1
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error parsing JSON on line {line_num}: {e}")
                        continue
            
            # Update file metadata
            self._update_file_metadata(file_path, 'jsonl', records_inserted)
            
            logger.info(f"Successfully loaded {records_inserted} records from {file_path}")
            return records_inserted
            
        except Exception as e:
            logger.error(f"Error loading JSONL file {file_path}: {e}")
            return 0
    
    def _upsert_document(self, data: Dict[str, Any]) -> bool:
        """Insert or update a document record with conflict detection."""
        try:
            cursor = self.conn.cursor()
            # Normalize key name: accept 'dokument_id' from inputs, store as 'dok_id'
            if 'dok_id' not in data and 'dokument_id' in data:
                data = {**data}
                data['dok_id'] = data.pop('dokument_id')
            
            # Check if document exists
            cursor.execute("SELECT * FROM dokumenter WHERE dok_id = ?", (data['dok_id'],))
            existing_record = cursor.fetchone()
            
            if existing_record:
                # Convert existing record to dict for comparison
                existing_data = dict(existing_record)
                
                # Check for conflicts and warn
                conflicts = []
                update_fields = []
                update_values = []
                
                for key, new_value in data.items():
                    if key != 'dok_id' and new_value is not None:
                        existing_value = existing_data.get(key)
                        
                        # Check for conflicts (both values exist and are different)
                        # Exclude source_file, created_at, updated_at, batch_id from conflict checking
                        if (existing_value is not None and existing_value != new_value and 
                            key not in ['source_file', 'created_at', 'updated_at', 'batch_id', 'url_fil', 'tekst_fil', 'oppsum_fil']):
                            conflicts.append({
                                'field': key,
                                'existing': existing_value,
                                'new': new_value,
                                'source_file': data.get('source_file', 'unknown')
                            })
                        
                        # Always update with new value (even if there's a conflict)
                        update_fields.append(f"{key} = ?")
                        update_values.append(new_value)
                
                # Log conflicts if any
                if conflicts:
                    logger.warning(f"Data conflicts detected for document {data['dok_id']}:")
                    for conflict in conflicts:
                        logger.warning(f"  Field '{conflict['field']}': existing='{conflict['existing']}' vs new='{conflict['new']}' (from {conflict['source_file']})")
                
                # Update existing record
                if update_fields:
                    update_values.append(data['dok_id'])
                    query = f"UPDATE dokumenter SET {', '.join(update_fields)}, updated_at = CURRENT_TIMESTAMP WHERE dok_id = ?"
                    cursor.execute(query, update_values)
            else:
                # Insert new record
                fields = list(data.keys())
                placeholders = ['?' for _ in fields]
                values = list(data.values())
                
                query = f"INSERT INTO dokumenter ({', '.join(fields)}) VALUES ({', '.join(placeholders)})"
                cursor.execute(query, values)
            
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error upserting document {data.get('dok_id', data.get('dokument_id', 'unknown'))}: {e}")
            self.conn.rollback()
            return False
    
    def _update_file_metadata(self, file_path: str, file_type: str, record_count: int):
        """Update file metadata table."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO file_metadata (file_path, file_type, record_count, last_loaded)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (file_path, file_type, record_count))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error updating file metadata: {e}")
    
    def load_all_files(self, directory: str = "sources", testing: bool = False) -> Dict[str, int]:
        """Load all CSV and JSONL files from the specified directory."""
        results = {}
        
        # Find all CSV files matching the pattern
        csv_pattern = os.path.join(directory, "*demokratibasen*.csv")
        csv_files = glob.glob(csv_pattern)
        
        # Find all JSONL files matching the pattern
        jsonl_pattern = os.path.join(directory, "*demokratibasen*.jsonl")
        jsonl_files = glob.glob(jsonl_pattern)
        
        logger.info(f"Found {len(csv_files)} CSV files and {len(jsonl_files)} JSONL files (testing={testing})")
        
        # Load CSV files
        for csv_file in csv_files:
            count = self.load_csv_file(csv_file, testing=testing)
            results[csv_file] = count
        
        # Load JSONL files
        for jsonl_file in jsonl_files:
            count = self.load_jsonl_file(jsonl_file, testing=testing)
            results[jsonl_file] = count
        
        return results
    
    def get_document(self, dokument_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM dokumenter WHERE dok_id = ?", (dokument_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def search_documents(self, **kwargs) -> List[Dict[str, Any]]:
        """Search documents by various criteria."""
        cursor = self.conn.cursor()
        
        conditions = []
        values = []
        
        for key, value in kwargs.items():
            if value is not None:
                if key in ['kommune', 'nyhetsverdi']:
                    conditions.append(f"{key} = ?")
                    values.append(value)
                else:
                    conditions.append(f"{key} LIKE ?")
                    values.append(f"%{value}%")
        
        if conditions:
            query = f"SELECT * FROM documents WHERE {' AND '.join(conditions)}"
        else:
            query = "SELECT * FROM documents"
        
        cursor.execute(query, values)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        cursor = self.conn.cursor()
        
        stats = {}
        
        # Total documents
        cursor.execute("SELECT COUNT(*) FROM dokumenter")
        stats['total_documents'] = cursor.fetchone()[0]
        
        # Documents by type
        cursor.execute("SELECT dok_type, COUNT(*) FROM dokumenter GROUP BY dok_type")
        stats['by_doc_type'] = dict(cursor.fetchall())
        
        # Documents by kommune
        cursor.execute("SELECT kommune, COUNT(*) FROM dokumenter GROUP BY kommune")
        stats['by_kommune'] = dict(cursor.fetchall())
        
        # Documents with text
        cursor.execute("SELECT COUNT(*) FROM dokumenter WHERE dok_tekst IS NOT NULL AND dok_tekst != ''")
        stats['documents_with_text'] = cursor.fetchone()[0]
        
        # Documents with inferences
        cursor.execute("SELECT COUNT(*) FROM dokumenter WHERE oppsummering IS NOT NULL AND oppsummering != ''")
        stats['documents_with_inferences'] = cursor.fetchone()[0]
        
        # File metadata
        cursor.execute("SELECT file_type, COUNT(*), SUM(record_count) FROM file_metadata GROUP BY file_type")
        file_stats = cursor.fetchall()
        stats['loaded_files'] = {row[0]: {'count': row[1], 'total_records': row[2]} for row in file_stats}
        
        return stats
    
    def check_data_consistency(self, directory: str = "sources") -> Dict[str, List[Dict[str, Any]]]:
        """Check consistency between source files and database content."""
        logger.info("Starting comprehensive data consistency check...")
        
        all_conflicts = {}
        cursor = self.conn.cursor()
        
        # Find all CSV files matching the pattern
        csv_pattern = os.path.join(directory, "*demokratibasen*.csv")
        csv_files = glob.glob(csv_pattern)
        
        # Find all JSONL files matching the pattern
        jsonl_pattern = os.path.join(directory, "*demokratibasen*.jsonl")
        jsonl_files = glob.glob(jsonl_pattern)
        
        logger.info(f"Checking consistency for {len(csv_files)} CSV files and {len(jsonl_files)} JSONL files")
        
        # Check CSV files
        for csv_file in csv_files:
            conflicts = self._check_csv_consistency(csv_file, cursor)
            if conflicts:
                all_conflicts[csv_file] = conflicts
        
        # Check JSONL files
        for jsonl_file in jsonl_files:
            conflicts = self._check_jsonl_consistency(jsonl_file, cursor)
            if conflicts:
                all_conflicts[jsonl_file] = conflicts
        
        return all_conflicts

    def check_url_consistency(self, directory: str = "sources", testing: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """Check only URL CSV files ("*-urls-*") against database.

        For each row, compare dok_type, kommune, dok_tittel, url with the values in `dokumenter`
        for the same dok_id. Report per-file inconsistencies and return mapping file->list of issues.
        """
        logger.info("Starting URL-only consistency check...")
        pattern = os.path.join(directory, "*-urls-*.csv")
        url_files = glob.glob(pattern)
        logger.info(f"Checking consistency for {len(url_files)} URL CSV files")

        all_conflicts: Dict[str, List[Dict[str, Any]]] = {}
        cursor = self.conn.cursor()

        for csv_path in url_files:
            try:
                df = pd.read_csv(csv_path)
                if testing:
                    df = df.head(TEST_LINES)
                conflicts: List[Dict[str, Any]] = []
                for _, row in df.iterrows():
                    dok_id = row.get('dokument_id')
                    if not dok_id or pd.isna(dok_id):
                        continue
                    cursor.execute(
                        "SELECT dok_type, kommune, dok_tittel, url FROM dokumenter WHERE dok_id = ?",
                        (dok_id,)
                    )
                    rec = cursor.fetchone()
                    if not rec:
                        # Not present in DB; skip or record? We'll record as info for visibility
                        conflicts.append({
                            'dok_id': dok_id,
                            'field': 'missing_in_db',
                            'database_value': None,
                            'file_value': {
                                'dok_type': row.get('doc_type'),
                                'kommune': row.get('kommune'),
                                'dok_tittel': row.get('tittel'),
                                'url': row.get('url')
                            },
                            'file_path': csv_path
                        })
                        continue
                    db_dok_type, db_kommune, db_dok_tittel, db_url = rec
                    checks = [
                        ('dok_type', db_dok_type, row.get('doc_type')),
                        ('kommune', db_kommune, row.get('kommune')),
                        ('dok_tittel', db_dok_tittel, row.get('tittel')),
                        ('url', db_url, row.get('url')),
                    ]
                    for field, db_val, file_val in checks:
                        if db_val is not None and file_val is not None and db_val != file_val:
                            conflicts.append({
                                'dok_id': dok_id,
                                'field': field,
                                'database_value': db_val,
                                'file_value': file_val,
                                'file_path': csv_path
                            })
                if conflicts:
                    logger.warning(f"Found {len(conflicts)} URL inconsistencies in {os.path.basename(csv_path)}")
                    all_conflicts[csv_path] = conflicts
                else:
                    logger.info(f"No URL inconsistencies found in {os.path.basename(csv_path)}")
            except Exception as e:
                logger.error(f"Error checking URL consistency for {csv_path}: {e}")
        return all_conflicts
    
    def _check_csv_consistency(self, file_path: str, cursor) -> List[Dict[str, Any]]:
        """Check consistency for a single CSV file."""
        try:
            logger.info(f"Checking consistency for: {file_path}")
            
            df = pd.read_csv(file_path)
            filename = os.path.basename(file_path)
            
            # Determine file type
            if 'urls' in filename:
                file_type = 'urls'
            elif 'texts' in filename:
                file_type = 'texts'
            elif 'inferences' in filename:
                file_type = 'inferences'
            else:
                file_type = 'unknown'
            
            conflicts = []
            
            for _, row in df.iterrows():
                dokument_id = row.get('dokument_id')
                if not dokument_id:
                    continue
                
                # Get existing database record
                cursor.execute("SELECT * FROM dokumenter WHERE dok_id = ?", (dokument_id,))
                existing_record = cursor.fetchone()
                
                if not existing_record:
                    continue  # Skip if not in database
                
                existing_data = dict(existing_record)
                
                # Prepare expected data based on file type
                if file_type == 'urls':
                    expected_data = {
                        'dok_id': row.get('dokument_id'),
                        'dok_type': row.get('doc_type'),
                        'kommune': row.get('kommune'),
                        'tittel': row.get('tittel'),
                        'url': row.get('url')
                    }
                elif file_type == 'texts':
                    expected_data = {
                        'dok_id': row.get('dokument_id'),
                        'dok_type': row.get('doc_type'),
                        'kommune': row.get('kommune'),
                        'tittel': row.get('tittel'),
                        'url': row.get('url'),
                        'doc_tekst': row.get('doc_tekst', '')
                    }
                elif file_type == 'inferences':
                    expected_data = {
                        'dokument_id': row.get('dokument_id'),
                        'batch_id': row.get('batch_id'),
                        'summary_title': row.get('tittel'),  # Map tittel to summary_title for inference files
                        'oppsummering': row.get('oppsummering'),
                        'personer': row.get('personer'),
                        'nokkelord': row.get('nokkelord'),
                        'nyhetsverdi': row.get('nyhetsverdi')
                    }
                else:
                    # Generic handling
                    expected_data = {k: v for k, v in row.items() if pd.notna(v)}
                
                # Check for conflicts
                for key, expected_value in expected_data.items():
                    if expected_value is not None and key in existing_data:
                        existing_value = existing_data[key]
                        
                        # Skip metadata fields
                        if key in ['source_file', 'created_at', 'updated_at', 'batch_id']:
                            continue
                        
                        # Check for conflicts
                        if existing_value != expected_value:
                            conflicts.append({
                                'dokument_id': dokument_id,
                                'field': key,
                                'database_value': existing_value,
                                'file_value': expected_value,
                                'file_path': file_path
                            })
            
            if conflicts:
                logger.warning(f"Found {len(conflicts)} conflicts in {file_path}")
            else:
                logger.info(f"No conflicts found in {file_path}")
            
            return conflicts
            
        except Exception as e:
            logger.error(f"Error checking consistency for {file_path}: {e}")
            return []
    
    def _check_jsonl_consistency(self, file_path: str, cursor) -> List[Dict[str, Any]]:
        """Check consistency for a single JSONL file."""
        try:
            logger.info(f"Checking consistency for: {file_path}")
            
            conflicts = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        dokument_id = data.get('dokument_id')
                        
                        if not dokument_id:
                            continue
                        
                        # Get existing database record
                        cursor.execute("SELECT * FROM dokumenter WHERE dok_id = ?", (dokument_id,))
                        existing_record = cursor.fetchone()
                        
                        if not existing_record:
                            continue  # Skip if not in database
                        
                        existing_data = dict(existing_record)
                        
                        # Prepare expected data
                        expected_data = {
                            'dok_id': data.get('dokument_id'),
                            'dok_type': data.get('doc_type'),
                            'kommune': data.get('kommune'),
                            'tittel': data.get('tittel'),
                            'url': data.get('url'),
                            'doc_tekst': data.get('tekst', '')
                        }
                        
                        # Check for conflicts
                        for key, expected_value in expected_data.items():
                            if expected_value is not None and key in existing_data:
                                existing_value = existing_data[key]
                                
                                # Skip metadata fields
                                if key in ['source_file', 'created_at', 'updated_at', 'batch_id']:
                                    continue
                                
                                # Check for conflicts
                                if existing_value != expected_value:
                                    conflicts.append({
                                        'dokument_id': dokument_id,
                                        'field': key,
                                        'database_value': existing_value,
                                        'file_value': expected_value,
                                        'file_path': file_path,
                                        'line_number': line_num
                                    })
                    
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error parsing JSON on line {line_num}: {e}")
                        continue
            
            if conflicts:
                logger.warning(f"Found {len(conflicts)} conflicts in {file_path}")
            else:
                logger.info(f"No conflicts found in {file_path}")
            
            return conflicts
            
        except Exception as e:
            logger.error(f"Error checking consistency for {file_path}: {e}")
            return []
    
    def export_complete_triples(self, output_file: str = "training_data/id_doctext_summary.csv"):
        """Export dokument_id, doc_tekst, oppsummering triples where none are missing."""
        logger.info(f"Exporting complete triples to {output_file}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT dok_id, dok_tekst, oppsummering 
            FROM dokumenter 
            WHERE dok_id IS NOT NULL 
            AND dok_tekst IS NOT NULL 
            AND dok_tekst != '' 
            AND oppsummering IS NOT NULL 
            AND oppsummering != ''
        """)
        
        results = cursor.fetchall()
        
        # Write to CSV file
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['dokument_id', 'doc_tekst', 'oppsummering'])
            
            for row in results:
                writer.writerow([row[0], row[1], row[2]])
        
        logger.info(f"Exported {len(results)} complete triples to {output_file}")
        return len(results)
    
    def export_urls_without_text(self, output_file: str = "urls_with_missing_doctext.csv"):
        """Export dokument_id, url pairs where both are set but doc_tekst is missing, excluding case_attachment documents."""
        logger.info(f"Exporting URLs with missing text to {output_file} (excluding case_attachment documents)")
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT dok_id, url 
            FROM dokumenter 
            WHERE dok_id IS NOT NULL 
            AND url IS NOT NULL 
            AND url != '' 
            AND (dok_tekst IS NULL OR dok_tekst = '')
            AND dok_type != 'case_attachment'
        """)
        
        results = cursor.fetchall()
        
        # Write to CSV file
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['dokument_id', 'url'])
            
            for row in results:
                writer.writerow([row[0], row[1]])
        
        logger.info(f"Exported {len(results)} URLs with missing text to {output_file} (excluding case_attachment documents)")
        return len(results)
    
    def load_all_urls(self, testing: bool = False) -> Dict[str, int]:
        """Load all rows from SOURCE_FILES that are URL CSVs ("*-urls-*") in listed order.

        Reads only files whose basenames contain "-urls-" or end with "-urls.csv" and loads
        the columns: dokument_id, doc_type, kommune, tittel, url. For each dokument_id seen
        more than once, warn if any of doc_type, kommune, tittel, or url differ from the
        existing value in the database. Uses the standard upsert path so the last occurrence
        in SOURCE_FILES order wins.
        """
        results: Dict[str, int] = {}
        cursor = self.conn.cursor()
        for file_path in SOURCE_FILES:
            filename = os.path.basename(file_path)
            if not ("-urls-" in filename or filename.endswith("-urls.csv") or "urls" in filename) or not filename.endswith(".csv"):
                continue
            try:
                logger.info(f"Loading URL file (ordered): {file_path} (testing={testing})")
                df = pd.read_csv(file_path)
                if testing:
                    df = df.head(TEST_LINES)
                inserted = 0
                for _, row in df.iterrows():
                    # New row
                    dokument_id = row.get('dokument_id')
                    if pd.isna(dokument_id) or dokument_id is None:
                        continue
                    new_values = {
                        'dok_type': row.get('doc_type'),
                        'kommune': row.get('kommune'),
                        'dok_tittel': row.get('tittel'),
                        'url': row.get('url')
                    }
                    # Check existing for conflicts on selected fields
                    cursor.execute("SELECT dok_id, dok_type, kommune, dok_tittel, url FROM dokumenter WHERE dok_id = ?", (dokument_id,))
                    existing = cursor.fetchone()
                    if existing:
                        existing_dict = dict(existing)
                        for fld in ['dok_type', 'kommune', 'dok_tittel', 'url']:
                            ev = existing_dict.get(fld)
                            nv = new_values.get(fld)
                            if ev is not None and nv is not None and ev != nv:
                                logger.warning(
                                    f"Inconsistency for dokument_id={dokument_id}: field '{fld}' existing='{ev}' vs new='{nv}' (file {filename})"
                                )
                    # Upsert
                    data = {
                        'dokument_id': dokument_id,
                        'dok_type': new_values['dok_type'],
                        'kommune': new_values['kommune'],
                        'dok_tittel': new_values['dok_tittel'],
                        'url': new_values['url'],
                        'url_fil': file_path
                    }
                    if self._upsert_document(data):
                        inserted += 1
                # Record file metadata
                self._update_file_metadata(file_path, 'urls', inserted)
                results[file_path] = inserted
                logger.info(f"Loaded {inserted} URL rows from {file_path}")
            except Exception as e:
                logger.error(f"Error loading URL file {file_path}: {e}")
                results[file_path] = 0
        return results
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main(load_urls: bool = False, load_texts: bool = False, load_inferences: bool = False, 
         testing: bool = False, check_consistency: bool = False, 
         export_triples: bool = False, export_urls: bool = False):
    """Main function to load files into the database, check consistency, or export data."""
    # Change to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Initialize database
    with TrainingDataDatabase() as db:
        if load_urls:
            logger.info("Loading URLs...")
            results = db.load_all_urls(testing=testing)
            logger.info(f"✅ Loaded {results} URLs")
        elif load_texts:
            logger.info("Loading texts...")
            results = db.load_all_texts(testing=testing)
            logger.info(f"✅ Loaded {results} texts")
        elif load_inferences:
            logger.info("Loading inferences...")
            results = db.load_all_inferences(testing=testing)
            logger.info(f"✅ Loaded {results} inferences")
        if export_triples:
            logger.info("Exporting complete triples...")
            count = db.export_complete_triples()
            logger.info(f"✅ Exported {count} complete triples to training_data/id_doctext_summary.csv")
        elif export_urls:
            logger.info("Exporting URLs with missing text...")
            count = db.export_urls_without_text()
            logger.info(f"✅ Exported {count} URLs with missing text to urls_with_missing_doctext.csv")
        elif check_consistency:
            logger.info("Running comprehensive data consistency check...")
            conflicts = db.check_data_consistency()
            
            if conflicts:
                logger.warning(f"\n=== CONSISTENCY CHECK RESULTS ===")
                logger.warning(f"Found conflicts in {len(conflicts)} files:")
                
                total_conflicts = 0
                for file_path, file_conflicts in conflicts.items():
                    logger.warning(f"\nFile: {os.path.basename(file_path)}")
                    logger.warning(f"  Conflicts: {len(file_conflicts)}")
                    total_conflicts += len(file_conflicts)
                    
                    # Show first few conflicts as examples
                    for i, conflict in enumerate(file_conflicts[:5]):
                        logger.warning(f"  {i+1}. Document {conflict['dokument_id'][:8]}...")
                        logger.warning(f"     Field '{conflict['field']}': DB='{conflict['database_value']}' vs File='{conflict['file_value']}'")
                    
                    if len(file_conflicts) > 5:
                        logger.warning(f"     ... and {len(file_conflicts) - 5} more conflicts")
                
                logger.warning(f"\nTotal conflicts found: {total_conflicts}")
            else:
                logger.info("✅ No data consistency issues found! All source files match database content.")
        else:
            logger.info(f"Starting to load all training data files... (testing={testing})")
            
            # Load all files
            results = db.load_all_files(testing=testing)
            
            # Print results
            logger.info("\nLoading Results:")
            total_records = 0
            for file_path, count in results.items():
                logger.info(f"  {os.path.basename(file_path)}: {count} records")
                total_records += count
            
            logger.info(f"\nTotal records loaded: {total_records}")
            
            # Print statistics
            stats = db.get_statistics()
            logger.info("\nDatabase Statistics:")
            logger.info(f"  Total documents: {stats['total_documents']}")
            logger.info(f"  Documents with text: {stats['documents_with_text']}")
            logger.info(f"  Documents with inferences: {stats['documents_with_inferences']}")
            logger.info(f"  Documents by type: {stats['by_doc_type']}")
            logger.info(f"  Documents by kommune: {stats['by_kommune']}")


if __name__ == "__main__":

    load_urls = "--load-urls" in sys.argv or "-u" in sys.argv
    load_texts = "--load-texts" in sys.argv or "-t" in sys.argv
    load_inferences = "--load-inferences" in sys.argv or "-i" in sys.argv
    testing = "--testing" in sys.argv or "-T" in sys.argv
    check_consistency = "--check-consistency" in sys.argv or "-c" in sys.argv
    export_triples = "--export-triples" in sys.argv
    export_urls = "--export-urls" in sys.argv
    main(load_urls=load_urls, 
         load_texts=load_texts, 
         load_inferences=load_inferences,
         testing=testing, 
         check_consistency=check_consistency, 
         export_triples=export_triples, 
         export_urls=export_urls)
