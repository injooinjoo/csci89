import subprocess
import sys
import os
import shutil

def convert_tex_to_pdf(tex_path, output_path=None):
    """
    Convert LaTeX file to PDF using pdflatex

    Args:
        tex_path (str): Path to the .tex file
        output_path (str): Output PDF path (optional)
    """
    if not os.path.exists(tex_path):
        print(f"Error: TeX file '{tex_path}' not found")
        return False

    # Get directory and filename
    tex_dir = os.path.dirname(os.path.abspath(tex_path))
    tex_filename = os.path.basename(tex_path)

    # Change to tex directory for compilation
    original_dir = os.getcwd()

    try:
        os.chdir(tex_dir)

        print(f"Compiling {tex_filename}...")

        # Simple pdflatex command
        result = subprocess.run([
            'pdflatex',
            '-interaction=nonstopmode',
            tex_filename
        ], capture_output=True)

        if result.returncode != 0:
            print("First compilation failed, trying again...")
            # Try once more
            result = subprocess.run([
                'pdflatex',
                '-interaction=nonstopmode',
                tex_filename
            ], capture_output=True)

            if result.returncode != 0:
                print("Error: LaTeX compilation failed")
                print("Make sure MiKTeX is properly installed and updated")
                print("You can manually run: miktex-console")
                return False

        # Find generated PDF
        pdf_filename = tex_filename.replace('.tex', '.pdf')
        generated_pdf = os.path.join(tex_dir, pdf_filename)

        if not os.path.exists(generated_pdf):
            print("Error: PDF file was not generated")
            return False

        # Move to output path if specified
        if output_path:
            shutil.move(generated_pdf, output_path)
            final_pdf = output_path
        else:
            final_pdf = generated_pdf

        print(f"Successfully converted '{tex_path}' to '{final_pdf}'")

        # Clean up auxiliary files
        cleanup_extensions = ['.aux', '.log', '.out', '.toc']
        base_name = tex_filename.replace('.tex', '')

        for ext in cleanup_extensions:
            aux_file = os.path.join(tex_dir, base_name + ext)
            if os.path.exists(aux_file):
                os.remove(aux_file)

        return True

    except FileNotFoundError:
        print("Error: pdflatex not found!")
        print("Checking if pdflatex is installed...")
        try:
            check_result = subprocess.run(['where', 'pdflatex'], capture_output=True, text=True)
            if check_result.returncode == 0:
                print(f"Found pdflatex at: {check_result.stdout.strip()}")
                print("There might be a PATH issue. Try restarting your terminal.")
            else:
                print("Please install MiKTeX: winget install MiKTeX.MiKTeX")
        except:
            print("Please install MiKTeX: winget install MiKTeX.MiKTeX")
        return False
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return False
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tex_to_pdf_simple.py <tex_path> [output_path]")
        sys.exit(1)

    tex_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    convert_tex_to_pdf(tex_path, output_path)