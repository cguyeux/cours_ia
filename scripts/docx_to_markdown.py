import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}


def load_numbering(zf: zipfile.ZipFile) -> dict[str, dict[int, str]]:
    numbering_map: dict[str, dict[int, str]] = {}
    if 'word/numbering.xml' not in zf.namelist():
        return numbering_map
    root = ET.fromstring(zf.read('word/numbering.xml'))
    abstract: dict[str, dict[int, str]] = {}
    for abstr in root.findall('w:abstractNum', NS):
        aid = abstr.attrib.get(f'{{{NS["w"]}}}abstractNumId')
        if not aid:
            continue
        levels: dict[int, str] = {}
        for lvl in abstr.findall('w:lvl', NS):
            ilvl = int(lvl.attrib.get(f'{{{NS["w"]}}}ilvl', '0'))
            fmt_el = lvl.find('w:numFmt', NS)
            fmt = fmt_el.attrib.get(f'{{{NS["w"]}}}val') if fmt_el is not None else 'bullet'
            levels[ilvl] = fmt
        abstract[aid] = levels
    for num in root.findall('w:num', NS):
        num_id = num.attrib.get(f'{{{NS["w"]}}}numId')
        if not num_id:
            continue
        abs_el = num.find('w:abstractNumId', NS)
        if abs_el is None:
            continue
        abstract_id = abs_el.attrib.get(f'{{{NS["w"]}}}val')
        if not abstract_id:
            continue
        numbering_map[num_id] = abstract.get(abstract_id, {})
    return numbering_map


def paragraph_text(para: ET.Element) -> str:
    parts: list[str] = []
    for elem in para.iter():
        if elem.tag == f'{{{NS["w"]}}}t':
            parts.append(elem.text or '')
        elif elem.tag == f'{{{NS["w"]}}}tab':
            parts.append('\t')
        elif elem.tag == f'{{{NS["w"]}}}br':
            parts.append('\n')
    return ''.join(parts)


def docx_to_markdown(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        document = ET.fromstring(zf.read('word/document.xml'))
        numbering = load_numbering(zf)
    lines: list[str] = []
    counters: dict[str, dict[int, int]] = {}
    for para in document.findall('.//w:p', NS):
        raw_text = paragraph_text(para).strip()
        ppr = para.find('w:pPr', NS)
        if not raw_text:
            lines.append('')
            continue
        style = None
        if ppr is not None:
            style_el = ppr.find('w:pStyle', NS)
            if style_el is not None:
                style = style_el.attrib.get(f'{{{NS["w"]}}}val', '').lower()
        if style and style.startswith('heading'):
            try:
                level = int(style.replace('heading', '') or '1')
            except ValueError:
                level = 1
            level = max(1, min(level, 6))
            lines.append(f"{'#' * level} {raw_text}")
            continue
        num_line = None
        if ppr is not None:
            num_pr = ppr.find('w:numPr', NS)
            if num_pr is not None:
                num_id_el = num_pr.find('w:numId', NS)
                ilvl_el = num_pr.find('w:ilvl', NS)
                if num_id_el is not None:
                    num_id = num_id_el.attrib.get(f'{{{NS["w"]}}}val', '0')
                    level = int(ilvl_el.attrib.get(f'{{{NS["w"]}}}val', '0')) if ilvl_el is not None else 0
                    fmt = numbering.get(num_id, {}).get(level, 'bullet')
                    indent = '  ' * level
                    if fmt == 'decimal':
                        counters.setdefault(num_id, {})
                        for lv in list(counters[num_id]):
                            if lv > level:
                                del counters[num_id][lv]
                        counters[num_id][level] = counters[num_id].get(level, 0) + 1
                        num_line = f"{indent}{counters[num_id][level]}. {raw_text}"
                    else:
                        counters.setdefault(num_id, {})
                        for lv in list(counters[num_id]):
                            if lv >= level:
                                del counters[num_id][lv]
                        num_line = f"{indent}- {raw_text}"
        if num_line is not None:
            lines.append(num_line)
        else:
            lines.append(raw_text)
    cleaned: list[str] = []
    blank = False
    for line in lines:
        if line.strip():
            cleaned.append(line)
            blank = False
        else:
            if not blank:
                cleaned.append('')
            blank = True
    return '\n'.join(cleaned).rstrip() + '\n'


def main() -> None:
    for pdf_path in Path('Cours').rglob('*.pdf'):
        docx_path = pdf_path.with_suffix('.docx')
        if not docx_path.exists():
            print(f'No DOCX counterpart for {pdf_path}')
            continue
        md_path = pdf_path.with_suffix('.md')
        md_path.write_text(docx_to_markdown(docx_path), encoding='utf-8')
        print(f'Converted {docx_path} -> {md_path}')


if __name__ == '__main__':
    main()
