import subprocess
import sys
import os
import shutil

def check_latex_installation():
    """Check if LaTeX is installed on the system"""
    try:
        # Check for pdflatex
        result = subprocess.run(['pdflatex', '--version'],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return 'pdflatex'
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # Try alternative way using where command
        try:
            result = subprocess.run(['where', 'pdflatex'],
                                  capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                return 'pdflatex'
        except:
            pass

    try:
        # Check for xelatex
        result = subprocess.run(['xelatex', '--version'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return 'xelatex'
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    try:
        # Check for lualatex
        result = subprocess.run(['lualatex', '--version'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return 'lualatex'
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None

def update_miktex():
    """Try to update MiKTeX if available"""
    try:
        print("Attempting to update MiKTeX...")
        result = subprocess.run(['miktex-update'], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("MiKTeX updated successfully")
            return True
        else:
            print("MiKTeX update failed or not needed")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("MiKTeX update command not found or timed out")
        return False

def convert_tex_to_pdf(tex_path, output_path=None, engine=None):
    """
    Convert LaTeX file to PDF

    Args:
        tex_path (str): Path to the .tex file
        output_path (str): Output PDF path (optional)
        engine (str): LaTeX engine to use (pdflatex, xelatex, lualatex)
    """
    if not os.path.exists(tex_path):
        print(f"Error: TeX file '{tex_path}' not found")
        return False

    # Check LaTeX installation
    if engine is None:
        engine = check_latex_installation()

    if engine is None:
        print("Error: No LaTeX engine found!")
        print("Please install one of the following:")
        print("- MiKTeX: winget install MiKTeX.MiKTeX")
        print("- TeX Live: https://tug.org/texlive/")
        return False

    print(f"Using LaTeX engine: {engine}")

    # Try to update MiKTeX if it's available
    if 'miktex' in subprocess.run(['where', engine], capture_output=True, text=True).stdout.lower():
        update_miktex()

    # Get directory and filename
    tex_dir = os.path.dirname(os.path.abspath(tex_path))
    tex_filename = os.path.basename(tex_path)

    # Change to tex directory for compilation
    original_dir = os.getcwd()

    try:
        os.chdir(tex_dir)

        # Compile LaTeX (run twice for references)
        for i in range(2):
            print(f"Running {engine} (pass {i+1}/2)...")
            # Add MiKTeX-specific options to auto-install packages
            cmd = [engine]
            if 'miktex' in engine.lower() or 'pdflatex' in engine or 'xelatex' in engine or 'lualatex' in engine:
                cmd.extend(['-synctex=1', '-interaction=nonstopmode'])
            else:
                cmd.extend(['-interaction=nonstopmode'])

            cmd.extend(['-output-format=pdf', tex_filename])

            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')

            if result.returncode != 0:
                print(f"Error during compilation:")
                print(result.stdout)
                print(result.stderr)
                return False

        # Find generated PDF
        pdf_filename = tex_filename.replace('.tex', '.pdf')
        generated_pdf = os.path.join(tex_dir, pdf_filename)

        if not os.path.exists(generated_pdf):
            print(f"Error: PDF file was not generated")
            return False

        # Move to output path if specified
        if output_path:
            shutil.move(generated_pdf, output_path)
            final_pdf = output_path
        else:
            final_pdf = generated_pdf

        print(f"Successfully converted '{tex_path}' to '{final_pdf}'")

        # Clean up auxiliary files
        cleanup_extensions = ['.aux', '.log', '.out', '.toc', '.nav', '.snm', '.fls', '.fdb_latexmk']
        base_name = tex_filename.replace('.tex', '')

        for ext in cleanup_extensions:
            aux_file = os.path.join(tex_dir, base_name + ext)
            if os.path.exists(aux_file):
                os.remove(aux_file)

        return True

    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return False

    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tex_to_pdf.py <tex_path> [output_path] [engine]")
        print("Engines: pdflatex, xelatex, lualatex")
        sys.exit(1)

    tex_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    engine = sys.argv[3] if len(sys.argv) > 3 else None

    convert_tex_to_pdf(tex_path, output_path, engine)