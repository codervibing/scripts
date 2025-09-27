# pip install pywin32 docx2python
import win32com.client as win32
from docx2python import docx2python
from pathlib import Path

pdf_path = Path(r"C:\path\in.pdf")
docx_path = pdf_path.with_suffix(".docx")

# Convert PDF -> DOCX using Microsoft Word
word = win32.Dispatch("Word.Application")
word.Visible = False
doc = word.Documents.Open(str(pdf_path))
doc.SaveAs(str(docx_path), FileFormat=16)  # 16 = wdFormatDocumentDefault (.docx)
doc.Close(False)
word.Quit()

# Extract text (docx2python gets paragraphs, tables, headers/footers, some textboxes)
with docx2python(str(docx_path)) as docx_content:
    text = "\n".join(docx_content.text)  # flat text
print(text[:1000])
