import sys
from pathlib import Path

def main():
    pdf_path = Path('deep_presentation.pdf')
    out_path = Path('deep_presentation.txt')
    if not pdf_path.exists():
        print('PDF_NOT_FOUND')
        sys.exit(1)
    try:
        from pdfminer.high_level import extract_text
    except Exception as exc:
        print('PDFMINER_IMPORT_ERROR')
        raise
    text = extract_text(str(pdf_path))
    out_path.write_text(text, encoding='utf-8')
    print(f'WROTE:{out_path}')

if __name__ == '__main__':
    main()


