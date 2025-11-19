import sys
import os
import importlib.util
import subprocess

REQUIRED_PACKAGES = [
    "networkx",
    "pydantic",
    "torch",
    "transformers",
    "spacy",
    "pyyaml",
    "pytest",
    "mypy",
    "llama-cpp-python"
]

def check_dependencies():
    print("[*] Checking dependencies...")
    missing = []
    for pkg in REQUIRED_PACKAGES:
        if importlib.util.find_spec(pkg) is None:
            missing.append(pkg)
    
    if missing:
        print(f"[!] Missing packages: {', '.join(missing)}")
        print("[*] Attempting to install missing packages...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
            print("[*] Installation successful.")
        except subprocess.CalledProcessError:
            print("[!] Failed to install packages automatically.")
            return False
    else:
        print("[*] All dependencies installed.")
    return True

def check_spacy_model():
    print("[*] Checking spaCy model 'en_core_web_sm'...")
    import spacy
    try:
        spacy.load("en_core_web_sm")
        print("[*] spaCy model found.")
    except OSError:
        print("[!] spaCy model not found. Downloading...")
        try:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            print("[*] Download successful.")
        except Exception as e:
            print(f"[!] Failed to download spaCy model: {e}")

def main():
    print("=== Neurosymbolic AI Preflight Check ===")
    if not check_dependencies():
        sys.exit(1)
    check_spacy_model()
    print("\n[SUCCESS] Preflight complete. System is ready.")

if __name__ == "__main__":
    main()
