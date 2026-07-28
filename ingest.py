import os
import shutil
import tempfile
import git
from pathlib import Path

# Configurable extensions
SUPPORTED_EXTENSIONS = {
    '.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.go', '.rs', 
    '.rb', '.php', '.cpp', '.c', '.h', '.cs', '.md', '.sql'
}

IGNORED_DIRS = {
    'node_modules', '.git', '__pycache__', 'dist', 'build', 'venv', 'env'
}

def clone_repository(repo_url, branch=None):
    """Shallow clones a repository to a temporary directory."""
    temp_dir = tempfile.mkdtemp()
    try:
        clone_args = {"depth": 1}
        if branch:
            clone_args["branch"] = branch
        
        git.Repo.clone_from(repo_url, temp_dir, **clone_args)
        return temp_dir
    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise e

def get_repo_files(repo_path, max_files=1500):
    """Walks the repo and returns a list of file paths and contents."""
    docs = []
    file_count = 0
    
    for root, dirs, files in os.walk(repo_path):
        # Filter ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith('.')]
        
        for file in files:
            if file_count >= max_files:
                break
                
            file_path = Path(root) / file
            rel_path = os.path.relpath(file_path, repo_path)
            
            # Skip if hidden or not supported extension or too large (>500KB)
            if (file.startswith('.') or 
                file_path.suffix not in SUPPORTED_EXTENSIONS or 
                os.path.getsize(file_path) > 500 * 1024):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if content.strip():
                        docs.append({
                            "content": content,
                            "metadata": {
                                "file_path": rel_path,
                                "file_name": file,
                                "file_ext": file_path.suffix
                            }
                        })
                        file_count += 1
            except Exception:
                continue
                
    return docs