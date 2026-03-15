"""
Script to convert Markdown to PDF
Requires: pip install markdown pdfkit
Note: Also need to install wkhtmltopdf from https://wkhtmltopdf.org/downloads.html
"""

import markdown
import pdfkit
import os

def markdown_to_pdf(md_file, pdf_file):
    """Convert Markdown file to PDF"""
    
    # Read markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
    
    # Add CSS styling
    html_with_style = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                max-width: 900px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                border-bottom: 2px solid #95a5a6;
                padding-bottom: 8px;
                margin-top: 30px;
            }}
            h3 {{
                color: #7f8c8d;
                margin-top: 20px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #3498db;
                color: white;
                font-weight: bold;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            code {{
                background-color: #f4f4f4;
                padding: 2px 5px;
                border-radius: 3px;
                font-family: monospace;
            }}
            pre {{
                background-color: #f4f4f4;
                padding: 10px;
                border-radius: 5px;
                overflow-x: auto;
            }}
            ul, ol {{
                margin-left: 20px;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Convert HTML to PDF
    try:
        pdfkit.from_string(html_with_style, pdf_file, options={
            'encoding': 'UTF-8',
            'page-size': 'A4',
            'margin-top': '20mm',
            'margin-right': '20mm',
            'margin-bottom': '20mm',
            'margin-left': '20mm',
        })
        print(f"✅ Successfully converted {md_file} to {pdf_file}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n⚠️  Make sure you have installed wkhtmltopdf:")
        print("   Download from: https://wkhtmltopdf.org/downloads.html")
        return False

if __name__ == "__main__":
    # Convert TIMELINE.md to PDF
    md_file = "TIMELINE.md"
    pdf_file = "TIMELINE.pdf"
    
    if os.path.exists(md_file):
        markdown_to_pdf(md_file, pdf_file)
    else:
        print(f"❌ File {md_file} not found!")
        
    # Also convert other documentation files
    docs = ["README.md", "METHODOLOGY.md", "EXPERIMENTS.md", "RESULTS.md", "REFERENCES.md"]
    
    print("\n📄 Convert other documentation? (y/n): ", end='')
    if input().lower() == 'y':
        for doc in docs:
            if os.path.exists(doc):
                pdf_name = doc.replace('.md', '.pdf')
                print(f"\nConverting {doc}...")
                markdown_to_pdf(doc, pdf_name)
