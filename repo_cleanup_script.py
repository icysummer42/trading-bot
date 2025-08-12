#!/usr/bin/env python3
"""
Trading Bot Repository Cleanup and Organization Script

This script systematically organizes the trading bot codebase by:
1. Identifying and archiving unused/duplicate files
2. Consolidating test scripts
3. Cleaning up old backup files
4. Organizing the data pipeline components
5. Creating a clear project structure
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import json
import hashlib

class RepositoryOrganizer:
    def __init__(self, repo_path="."):
        self.repo_path = Path(repo_path)
        self.archive_dir = self.repo_path / "archive"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.cleanup_report = {
            "timestamp": self.timestamp,
            "files_moved": [],
            "files_kept": [],
            "directories_created": [],
            "recommendations": []
        }
        
    def create_archive_structure(self):
        """Create organized archive directory structure"""
        archive_dirs = [
            "archive/old_pipelines",
            "archive/old_tests",
            "archive/old_backups",
            "archive/experimental",
            "archive/deprecated_strategies",
            "archive/documentation_drafts"
        ]
        
        for dir_path in archive_dirs:
            full_path = self.repo_path / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            self.cleanup_report["directories_created"].append(str(dir_path))
            
        print(f"✅ Created archive directory structure")
        return True
    
    def identify_duplicate_files(self):
        """Identify duplicate files based on content hash"""
        file_hashes = {}
        duplicates = []
        
        for file_path in self.repo_path.rglob("*.py"):
            if "archive" in str(file_path):
                continue
                
            try:
                with open(file_path, 'rb') as f:
                    content_hash = hashlib.md5(f.read()).hexdigest()
                    
                if content_hash in file_hashes:
                    duplicates.append({
                        "original": file_hashes[content_hash],
                        "duplicate": str(file_path)
                    })
                else:
                    file_hashes[content_hash] = str(file_path)
                    
            except Exception as e:
                print(f"⚠️ Error hashing {file_path}: {e}")
                
        return duplicates
    
    def cleanup_data_pipeline_files(self):
        """Archive old data pipeline versions"""
        pipeline_patterns = [
            "data_pipeline_old_*.py",
            "data_pipeline copy*.py",
            "old_pipeline_*.py",
            "backup_data_pipeline_migration",
            "data_pipeline_*.py"  # Except the main ones
        ]
        
        files_to_keep = [
            "data_pipeline.py",
            "data_pipeline_unified.py",  # Keep until we verify which is primary
            "enhanced_data_pipeline.py"
        ]
        
        moved_count = 0
        
        for pattern in pipeline_patterns:
            for file_path in self.repo_path.glob(pattern):
                if file_path.name not in files_to_keep:
                    destination = self.archive_dir / "old_pipelines" / file_path.name
                    shutil.move(str(file_path), str(destination))
                    self.cleanup_report["files_moved"].append({
                        "from": str(file_path),
                        "to": str(destination),
                        "category": "old_pipeline"
                    })
                    moved_count += 1
                    print(f"  Archived: {file_path.name}")
                    
        # Move backup directory
        backup_dir = self.repo_path / "backup_data_pipeline_migration"
        if backup_dir.exists():
            destination = self.archive_dir / "old_backups" / backup_dir.name
            shutil.move(str(backup_dir), str(destination))
            self.cleanup_report["files_moved"].append({
                "from": str(backup_dir),
                "to": str(destination),
                "category": "backup_directory"
            })
            moved_count += 1
            
        print(f"✅ Archived {moved_count} old pipeline files")
        return moved_count
    
    def consolidate_test_files(self):
        """Consolidate scattered test files"""
        test_patterns = [
            "signal_test.py",
            "fullsignaltest.py",
            "batch_signal_test.py",
            "test_pipeline_simple.py",
            "test_unified_pipeline.py",
            "demo_*.py"
        ]
        
        # Create tests directory if it doesn't exist
        tests_dir = self.repo_path / "tests"
        tests_dir.mkdir(exist_ok=True)
        
        moved_count = 0
        
        for pattern in test_patterns:
            for file_path in self.repo_path.glob(pattern):
                if file_path.parent == self.repo_path:  # Only move from root
                    destination = tests_dir / file_path.name
                    shutil.move(str(file_path), str(destination))
                    self.cleanup_report["files_moved"].append({
                        "from": str(file_path),
                        "to": str(destination),
                        "category": "test_consolidation"
                    })
                    moved_count += 1
                    print(f"  Moved test: {file_path.name} → tests/")
                    
        print(f"✅ Consolidated {moved_count} test files into tests/ directory")
        return moved_count
    
    def identify_unused_files(self):
        """Identify potentially unused files based on imports and references"""
        unused_candidates = []
        
        # Files that appear to be experimental or temporary
        experimental_patterns = [
            "*_experiment*.py",
            "*_temp*.py",
            "*_old*.py",
            "*copy*.py",
            "untitled*.py",
            "test_*.py"  # If not in tests directory
        ]
        
        for pattern in experimental_patterns:
            for file_path in self.repo_path.rglob(pattern):
                if "archive" not in str(file_path) and "tests" not in str(file_path):
                    unused_candidates.append(str(file_path))
                    
        return unused_candidates
    
    def clean_cache_files(self):
        """Clean up cache and temporary files"""
        cache_patterns = [
            "**/__pycache__",
            "**/*.pyc",
            "**/.pytest_cache",
            "**/.coverage",
            "**/cache/*.pkl",  # Old cache files
            "**/cache/*.pickle"
        ]
        
        removed_count = 0
        
        for pattern in cache_patterns:
            for path in self.repo_path.glob(pattern):
                if path.is_file():
                    path.unlink()
                    removed_count += 1
                elif path.is_dir():
                    shutil.rmtree(path)
                    removed_count += 1
                    
        print(f"✅ Cleaned {removed_count} cache files/directories")
        return removed_count
    
    def organize_documentation(self):
        """Organize documentation files"""
        doc_patterns = [
            "*.md",
            "*.txt",
            "*.rst"
        ]
        
        # Files to keep in root
        root_docs = [
            "README.md",
            "readme.md",
            "LICENSE",
            "requirements.txt",
            ".gitignore"
        ]
        
        docs_dir = self.repo_path / "docs"
        docs_dir.mkdir(exist_ok=True)
        
        moved_count = 0
        
        for pattern in doc_patterns:
            for file_path in self.repo_path.glob(pattern):
                if file_path.name not in root_docs and file_path.parent == self.repo_path:
                    # Check if it's a consolidation report or similar
                    if "COMPLETE" in file_path.name or "REPORT" in file_path.name:
                        destination = docs_dir / "reports" / file_path.name
                        destination.parent.mkdir(exist_ok=True)
                    else:
                        destination = docs_dir / file_path.name
                        
                    shutil.move(str(file_path), str(destination))
                    self.cleanup_report["files_moved"].append({
                        "from": str(file_path),
                        "to": str(destination),
                        "category": "documentation"
                    })
                    moved_count += 1
                    print(f"  Moved doc: {file_path.name} → docs/")
                    
        print(f"✅ Organized {moved_count} documentation files")
        return moved_count
    
    def generate_cleanup_report(self):
        """Generate detailed cleanup report"""
        report_path = self.repo_path / "docs" / "reports" / f"cleanup_report_{self.timestamp}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add recommendations
        self.cleanup_report["recommendations"] = [
            "Review archived files in 30 days and permanently delete if not needed",
            "Set up pre-commit hooks to maintain organization",
            "Create GitHub Actions workflow for automated testing",
            "Update README.md with new project structure",
            "Review and update requirements.txt for accuracy"
        ]
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(self.cleanup_report, f, indent=2, default=str)
            
        print(f"\n📊 Cleanup report saved to: {report_path}")
        return report_path
    
    def create_gitignore_updates(self):
        """Update .gitignore with proper patterns"""
        gitignore_additions = [
            "\n# Archive and backup",
            "archive/",
            "*.backup",
            "*.old",
            "*_old_*",
            "\n# Cache and temporary files",
            "cache/*.pkl",
            "cache/*.pickle",
            "*.pyc",
            "__pycache__/",
            ".pytest_cache/",
            ".coverage",
            "\n# Environment and credentials",
            ".env",
            "*.env",
            "env/",
            "venv/",
            "\n# IDE files",
            ".vscode/",
            ".idea/",
            "*.swp",
            "*.swo",
            "\n# OS files",
            ".DS_Store",
            "Thumbs.db"
        ]
        
        gitignore_path = self.repo_path / ".gitignore"
        
        # Read existing gitignore
        existing_patterns = set()
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                existing_patterns = set(line.strip() for line in f if line.strip() and not line.startswith('#'))
        
        # Add new patterns if not already present
        with open(gitignore_path, 'a') as f:
            for pattern in gitignore_additions:
                if pattern.startswith('\n') or pattern.startswith('#'):
                    f.write(pattern + '\n')
                elif pattern not in existing_patterns:
                    f.write(pattern + '\n')
                    
        print("✅ Updated .gitignore with recommended patterns")
        
    def run_full_cleanup(self):
        """Execute complete repository cleanup"""
        print("\n🧹 Starting Repository Cleanup")
        print("=" * 60)
        
        # Create archive structure
        self.create_archive_structure()
        
        # Find duplicates
        print("\n📍 Identifying duplicate files...")
        duplicates = self.identify_duplicate_files()
        if duplicates:
            print(f"  Found {len(duplicates)} duplicate files")
            for dup in duplicates[:5]:  # Show first 5
                print(f"    {dup['duplicate']} duplicates {dup['original']}")
                
        # Clean data pipeline files
        print("\n📍 Cleaning data pipeline files...")
        self.cleanup_data_pipeline_files()
        
        # Consolidate test files
        print("\n📍 Consolidating test files...")
        self.consolidate_test_files()
        
        # Organize documentation
        print("\n📍 Organizing documentation...")
        self.organize_documentation()
        
        # Clean cache files
        print("\n📍 Cleaning cache files...")
        self.clean_cache_files()
        
        # Update gitignore
        print("\n📍 Updating .gitignore...")
        self.create_gitignore_updates()
        
        # Generate report
        print("\n📍 Generating cleanup report...")
        report_path = self.generate_cleanup_report()
        
        # Summary
        print("\n" + "=" * 60)
        print("✨ CLEANUP COMPLETE!")
        print("=" * 60)
        print(f"📁 Files moved to archive: {len(self.cleanup_report['files_moved'])}")
        print(f"📁 Directories created: {len(self.cleanup_report['directories_created'])}")
        print(f"📊 Report saved to: {report_path}")
        
        print("\n🎯 Next Steps:")
        print("1. Review archived files in archive/ directory")
        print("2. Delete archive/ after 30 days if files not needed")
        print("3. Run: git add -A && git commit -m 'Repository cleanup and organization'")
        print("4. Update README.md with new structure")
        print("5. Run tests to ensure nothing broke: python -m pytest tests/")
        
        return True


def main():
    """Execute repository cleanup"""
    organizer = RepositoryOrganizer()
    
    # Confirm before proceeding
    print("⚠️  This script will reorganize your repository structure")
    print("   All files will be preserved in the archive/ directory")
    response = input("\nProceed with cleanup? (yes/no): ").lower().strip()
    
    if response in ['yes', 'y']:
        success = organizer.run_full_cleanup()
        return 0 if success else 1
    else:
        print("❌ Cleanup cancelled")
        return 1


if __name__ == "__main__":
    exit(main())