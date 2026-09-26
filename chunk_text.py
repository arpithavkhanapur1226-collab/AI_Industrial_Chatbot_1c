from pypdf import PdfReader

# PDF location
pdf_path = "documents/industrial_robotics.pdf"

# Read PDF
reader = PdfReader(pdf_path)

# Extract all text
full_text = ""

for page in reader.pages:
    text = page.extract_text()
    full_text += text + "\n"

# Split text into chunks
chunk_size = 500

chunks = []

for i in range(0, len(full_text), chunk_size):
    chunk = full_text[i:i + chunk_size]
    chunks.append(chunk)

# Display chunks
print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print("\n" + "=" * 60)
    print("CHUNK", i + 1)
    print("=" * 60)
    print(chunk)