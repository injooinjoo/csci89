import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import TagRemovePreprocessor
import sys
import os
import webbrowser

def convert_notebook_to_html(notebook_path, output_path=None):
    """
    Convert Jupyter notebook to HTML with light mode styling
    Opens in browser for manual PDF save

    Args:
        notebook_path (str): Path to the .ipynb file
        output_path (str): Output HTML path (optional)
    """
    if not os.path.exists(notebook_path):
        print(f"Error: Notebook file '{notebook_path}' not found")
        return False

    if output_path is None:
        output_path = notebook_path.replace('.ipynb', '.html')

    try:
        # Read the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)

        # Create HTML exporter with light mode configuration
        html_exporter = HTMLExporter()
        html_exporter.template_name = 'classic'

        # Remove any dark mode tags if present
        tag_remover = TagRemovePreprocessor()
        tag_remover.remove_cell_tags = {'dark-mode'}
        html_exporter.register_preprocessor(tag_remover, enabled=True)

        # Convert to HTML
        (body, resources) = html_exporter.from_notebook_node(notebook)

        # Write HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(body)

        print(f"Successfully converted '{notebook_path}' to '{output_path}'")
        print("Opening in browser - use Ctrl+P to print/save as PDF")

        # Open in default browser
        webbrowser.open(f'file://{os.path.abspath(output_path)}')
        return True

    except Exception as e:
        print(f"Error converting notebook: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python notebook_to_pdf.py <notebook_path> [output_path]")
        sys.exit(1)

    notebook_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    convert_notebook_to_html(notebook_path, output_path)