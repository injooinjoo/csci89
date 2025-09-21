import subprocess
import sys
import os

def find_pdflatex():
    """Find pdflatex executable"""
    # Common MiKTeX paths
    common_paths = [
        r"C:\Users\injoo\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe",
        r"C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe",
        r"C:\texlive\2025\bin\windows\pdflatex.exe",
        "pdflatex"  # Try system PATH
    ]

    for path in common_paths:
        try:
            result = subprocess.run([path, "--version"],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"Found pdflatex: {path}")
                return path
        except:
            continue

    return None

def convert_tex_to_pdf(tex_file):
    """Convert TeX file to PDF"""
    if not os.path.exists(tex_file):
        print(f"Error: File {tex_file} not found")
        return False

    pdflatex_path = find_pdflatex()
    if not pdflatex_path:
        print("Error: pdflatex not found!")
        return False

    print(f"Converting {tex_file} to PDF...")

    # Get directory for compilation
    tex_dir = os.path.dirname(os.path.abspath(tex_file))
    tex_name = os.path.basename(tex_file)

    # Change to tex directory
    original_dir = os.getcwd()
    os.chdir(tex_dir)

    try:
        # Run pdflatex
        result = subprocess.run([
            pdflatex_path,
            "-interaction=nonstopmode",
            tex_name
        ], capture_output=True, text=True)

        if result.returncode == 0:
            pdf_name = tex_name.replace('.tex', '.pdf')
            print(f"Success! Created {pdf_name}")
            return True
        else:
            print("Compilation failed. Checking for missing packages...")
            # Print first few lines of error for debugging
            if result.stdout:
                lines = result.stdout.split('\n')[:10]
                for line in lines:
                    if 'error' in line.lower() or 'missing' in line.lower():
                        print(line)
            return False

    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python latex_converter.py <file.tex>")
        sys.exit(1)

    tex_file = sys.argv[1]
    convert_tex_to_pdf(tex_file)