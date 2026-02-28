import sys

# Try to read the PDF and extract text
try:
    import pypdf
    pdf_path = r"c:\Users\Abubakar\Downloads\Documents\End-to-End ML System_ Flight Delay Prediction.pdf"
    
    with open(pdf_path, 'rb') as file:
        reader = pypdf.PdfReader(file)
        print(f"PDF has {len(reader.pages)} pages\n")
        
        for i, page in enumerate(reader.pages[:5]):  # First 5 pages
            text = page.extract_text()
            print(f"{'='*80}")
            print(f"PAGE {i+1}")
            print(f"{'='*80}")
            print(text)
            print("\n")
except ImportError:
    print("pypdf not installed, trying alternative method...")
except Exception as e:
    print(f"Error: {e}")
    print("\nPlease check the PDF file path")
