###############################################################################################################################################################################
# FILE ORGANIZATION - Move all output files to organized directory
###############################################################################################################################################################################

import os
import shutil
import glob
from datetime import datetime

def organize_backtest_outputs(results_prefix, create_subdirs=True):
    """
    Organize all backtest output files into a structured directory
    
    Parameters:
    results_prefix: The prefix used for saving files (e.g., 'monthly_backtest_results_20241215_143022')
    create_subdirs: Whether to create subdirectories by file type
    """
    
    # Create main results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    main_dir = f'backtest_results_{timestamp}'
    
    # Create subdirectories if requested
    if create_subdirs:
        subdirs = {
            'csv': os.path.join(main_dir, 'csv_files'),
            'pkl': os.path.join(main_dir, 'pickle_files'), 
            'xlsx': os.path.join(main_dir, 'excel_files'),
            'png': os.path.join(main_dir, 'charts'),
            'other': os.path.join(main_dir, 'other_files')
        }
        
        # Create all directories
        for subdir in subdirs.values():
            os.makedirs(subdir, exist_ok=True)
    else:
        os.makedirs(main_dir, exist_ok=True)
        subdirs = {key: main_dir for key in ['csv', 'pkl', 'xlsx', 'png', 'other']}
    
    print(f"📁 Created results directory: {main_dir}")
    
    # Define file patterns to look for
    file_patterns = {
        'csv': ['*.csv'],
        'pkl': ['*.pkl', '*.pickle'],
        'xlsx': ['*.xlsx', '*.xls'],
        'png': ['*.png', '*.jpg', '*.jpeg', '*.pdf'],
        'other': ['*.txt', '*.json', '*.log']
    }
    
    moved_files = {key: [] for key in file_patterns.keys()}
    
    # Move files by type
    for file_type, patterns in file_patterns.items():
        for pattern in patterns:
            files = glob.glob(pattern)
            
            for file in files:
                # Check if file is related to this backtest run
                if results_prefix in file or any(keyword in file.lower() for keyword in 
                    ['backtest', 'portfolio', 'monthly', 'checkpoint', 'weights', 'performance']):
                    
                    try:
                        dest_path = os.path.join(subdirs[file_type], file)
                        shutil.move(file, dest_path)
                        moved_files[file_type].append(file)
                        print(f"   📄 Moved {file} → {subdirs[file_type]}")
                    except Exception as e:
                        print(f"   ⚠️  Error moving {file}: {e}")
    
    # Create summary report
    summary_file = os.path.join(main_dir, 'file_organization_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Backtest Results Organization Summary\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Results Prefix: {results_prefix}\n\n")
        
        for file_type, files in moved_files.items():
            f.write(f"{file_type.upper()} Files ({len(files)}):\n")
            for file in files:
                f.write(f"  - {file}\n")
            f.write(f"\n")
    
    print(f"\n📋 Summary written to: {summary_file}")
    
    # Print organization summary
    total_moved = sum(len(files) for files in moved_files.values())
    print(f"\n✅ File organization complete!")
    print(f"   Total files moved: {total_moved}")
    print(f"   Main directory: {main_dir}")
    
    if create_subdirs:
        for file_type, files in moved_files.items():
            if files:
                print(f"   {file_type.upper()}: {len(files)} files → {subdirs[file_type]}")
    
    return main_dir, moved_files

# Alternative: Simple version that moves everything to one folder
def organize_backtest_outputs_simple(results_prefix):
    """Simple version - moves all backtest files to one directory"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'backtest_results_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    # File extensions to move
    extensions = ['*.csv', '*.pkl', '*.xlsx', '*.png', '*.txt', '*.json']
    moved_count = 0
    
    for ext in extensions:
        files = glob.glob(ext)
        for file in files:
            if results_prefix in file or any(keyword in file.lower() for keyword in 
                ['backtest', 'portfolio', 'monthly', 'checkpoint', 'weights']):
                try:
                    shutil.move(file, os.path.join(results_dir, file))
                    moved_count += 1
                    print(f"📄 Moved: {file}")
                except Exception as e:
                    print(f"⚠️  Error moving {file}: {e}")
    
    print(f"\n✅ Moved {moved_count} files to: {results_dir}")
    return results_dir