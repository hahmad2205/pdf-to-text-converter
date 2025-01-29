import fitz


def pdf_to_text(pdf_path, txt_path):
    doc = fitz.open(pdf_path)

    with open(txt_path, "w", encoding="utf-8") as txt_file:
        for page in doc:
            text = page.get_text()
            txt_file.write(text + "\n")

    print(f"Text extracted and saved to {txt_path}")


# Example usage
pdf_to_text("documents/Harry-Potter-and-the-Chamber-of-Secrets.pdf",
            "ragtest/input/Harry-Potter-and-the-Chamber-of-Secrets.txt")
