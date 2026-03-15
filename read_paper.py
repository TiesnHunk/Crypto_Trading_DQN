import PyPDF2
import sys

# Set UTF-8 encoding for output
sys.stdout.reconfigure(encoding='utf-8')

try:
    with open('docs/paper.pdf', 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        print(f"Total pages: {len(reader.pages)}\n")
        print("="*80)
        
        # Read all pages
        for i in range(len(reader.pages)):
            print(f"\n{'='*80}")
            print(f"PAGE {i+1}")
            print(f"{'='*80}\n")
            text = reader.pages[i].extract_text()
            # Replace problematic characters
            text = text.encode('utf-8', errors='ignore').decode('utf-8')
            print(text)
            
except Exception as e:
    print(f"Error reading PDF: {e}")
    sys.exit(1)
