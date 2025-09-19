#!/usr/bin/env python3
"""
Setup script for Golf Video Anonymizer
Creates virtual environment and installs dependencies
"""

import subprocess
import sys
import os

def create_virtual_environment():
    """Create Python virtual environment"""
    print("Creating virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
    print("Virtual environment created successfully!")

def install_dependencies():
    """Install dependencies from requirements.txt"""
    print("Installing dependencies...")
    
    # Determine the correct pip path based on OS
    if os.name == 'nt':  # Windows
        pip_path = os.path.join("venv", "Scripts", "pip")
    else:  # Unix/Linux/macOS
        pip_path = os.path.join("venv", "bin", "pip")
    
    subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
    print("Dependencies installed successfully!")

def main():
    """Main setup function"""
    try:
        create_virtual_environment()
        install_dependencies()
        print("\nSetup complete!")
        print("To activate the virtual environment:")
        if os.name == 'nt':
            print("  venv\\Scripts\\activate")
        else:
            print("  source venv/bin/activate")
    except subprocess.CalledProcessError as e:
        print(f"Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()