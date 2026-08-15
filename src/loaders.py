from pathlib import Path
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_documents(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    files = data_dir.glob("*")

    results = []
    for f in files:
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        if ext == ".txt" or ext == ".md":
            text = f.read_text(encoding="utf-8")
            results.append({"filename": f.name, "text": text})
        elif ext == ".pdf":
            pdf = PdfReader(f)
            total_pages = len(pdf.pages)
            pages_text = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text)

            text = "\n".join(pages_text)
            if not text.strip():
                print(
                    f"[WARNING] {f.name}: no extractable text "
                    f"({len(pages_text)} of {total_pages} pages). "
                    f"The PDF is likely scanned or uses outline/vector fonts; "
                    f"text extraction returned nothing."
                )
            results.append({"filename": f.name, "text": text})
        else:
            print(f"Skipping unsupported file: {f.name}")

    return results


if __name__ == "__main__":
    documents = load_documents()
    for doc in documents:
        print(f"{doc['filename']} -> {len(doc['text'])} characters")
        preview = doc["text"][:120].replace("\n", " ")
        print(f"   {preview}")
        print()
