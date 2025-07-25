#!/usr/bin/env python3
"""
Clean Git History - Remove exposed tokens
"""

import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a git command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed")
            return result.stdout.strip()
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return None
    except Exception as e:
        print(f"❌ {description} error: {e}")
        return None

def clean_git_history():
    """Clean git history to remove exposed tokens"""
    print("🧹 Cleaning Git History")
    print("=" * 50)
    
    # Check if we're in a git repository
    if not Path(".git").exists():
        print("❌ Not in a git repository")
        return False
    
    # Get current branch
    current_branch = run_command("git branch --show-current", "Getting current branch")
    if not current_branch:
        return False
    
    print(f"📋 Current branch: {current_branch}")
    
    # Show recent commits
    print("\n📝 Recent commits:")
    run_command("git log --oneline -5", "Showing recent commits")
    
    # Interactive rebase to remove the token
    print(f"\n🔧 To remove the token from history, you need to:")
    print("1. Run: git rebase -i HEAD~2")
    print("2. Change 'pick' to 'edit' for the commit with the token")
    print("3. Run: git reset HEAD~1")
    print("4. Run: git add .")
    print("5. Run: git commit -m 'Remove hardcoded token'")
    print("6. Run: git rebase --continue")
    print("7. Run: git push --force-with-lease")
    
    # Alternative: Create a new commit that removes the token
    print(f"\n🔄 Alternative: Creating a new commit to remove the token...")
    
    # Check if the file still contains the token
    token_file = Path("scripts/setup_token.py")
    if token_file.exists():
        try:
            content = token_file.read_text(encoding='utf-8')
            if "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" in content:
                print("⚠️  Token still found in file. Please run the updated setup_token.py script first.")
                return False
        except UnicodeDecodeError:
            print("⚠️  Could not read file due to encoding issues. Proceeding with cleanup...")
    
    # Stage the changes
    run_command("git add scripts/setup_token.py", "Staging updated file")
    
    # Create a new commit
    commit_msg = "Remove hardcoded Hugging Face token for security"
    run_command(f'git commit -m "{commit_msg}"', "Creating security commit")
    
    print("\n✅ Token removed from current state")
    print("📝 Next steps:")
    print("1. Run: git push --force-with-lease")
    print("2. If that fails, you may need to clean the full history")
    
    return True

if __name__ == "__main__":
    clean_git_history() 