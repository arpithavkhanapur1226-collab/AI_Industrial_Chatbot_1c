from pypdf import PdfReader

# Location of our PDF
pdf_path = "documents/industrial_robotics.pdf"

# Open the PDF
reader = PdfReader(pdf_path)

print("Number of pages:", len(reader.pages))

# Extract text from every page
for page_number, page in enumerate(reader.pages, start=1):

    text = page.extract_text()

    print("\n" + "=" * 60)
    print("PAGE", page_number)
    print("=" * 60)

    print(text)