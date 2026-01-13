import argparse
import io
import sys
import zipfile
from pathlib import Path
from typing import List, Optional
from xml.etree import ElementTree as ET


WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W = "{" + WORD_NAMESPACE + "}"


def read_xml_from_docx(docx_path: Path, internal_path: str) -> Optional[ET.Element]:
    try:
        with zipfile.ZipFile(docx_path) as zf:
            with zf.open(internal_path) as fp:
                data = fp.read()
        return ET.fromstring(data)
    except KeyError:
        return None


def extract_paragraph_texts(root: ET.Element) -> List[str]:
    paragraphs: List[str] = []
    for p in root.iter(f"{W}p"):
        parts: List[str] = []
        for node in p.iter():
            tag = node.tag
            if tag == f"{W}t":
                parts.append(node.text or "")
            elif tag == f"{W}tab":
                parts.append("\t")
            elif tag == f"{W}br":
                parts.append("\n")
        text = "".join(parts).strip()
        paragraphs.append(text)
    return paragraphs


def extract_docx_to_text(docx_path: Path) -> str:
    document = read_xml_from_docx(docx_path, "word/document.xml")
    if document is None:
        raise FileNotFoundError("word/document.xml not found inside the .docx")

    doc_paragraphs = extract_paragraph_texts(document)

    footnote_root = read_xml_from_docx(docx_path, "word/footnotes.xml")
    endnote_root = read_xml_from_docx(docx_path, "word/endnotes.xml")

    output = io.StringIO()
    for para in doc_paragraphs:
        output.write(para)
        output.write("\n")
    
    if footnote_root is not None:
        output.write("\n" + ("-" * 20) + "\nFOOTNOTES\n" + ("-" * 20) + "\n")
        for para in extract_paragraph_texts(footnote_root):
            if para:
                output.write(para + "\n")

    if endnote_root is not None:
        output.write("\n" + ("-" * 20) + "\nENDNOTES\n" + ("-" * 20) + "\n")
        for para in extract_paragraph_texts(endnote_root):
            if para:
                output.write(para + "\n")

    return output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract plain text from a .docx file without external dependencies.")
    parser.add_argument("input", type=str, help="Path to the .docx file")
    parser.add_argument("-o", "--output", type=str, default=None, help="Path to the output .txt file")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        text = extract_docx_to_text(input_path)
    except Exception as exc:
        print(f"Failed to extract text: {exc}", file=sys.stderr)
        sys.exit(2)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = input_path.with_suffix(".txt")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    except Exception as exc:
        print(f"Failed to write output: {exc}", file=sys.stderr)
        sys.exit(3)

    print(f"Wrote text to: {output_path}")


if __name__ == "__main__":
    main()


