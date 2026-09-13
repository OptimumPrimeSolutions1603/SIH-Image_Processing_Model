"""Extract caption-verified rice disease figures from downloaded ICAR PDFs.

These crops are for cross-source smoke testing only. They are not a balanced dataset
and must not be mixed into training without a separate review and licensing decision.
"""

import csv
from pathlib import Path

import pypdfium2 as pdfium


ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "tmp" / "pdfs"
OUT = ROOT / "data" / "external_test" / "icar"

SOURCES = [
    {
        "pdf": "icar_kharif_2025.pdf",
        "page_index": 106,
        "crop": (180, 1286, 638, 1644),
        "label": "leaf_blast",
        "filename": "icar_kharif_2025_p91_leaf_blast.png",
        "publication_page": "91",
        "title": "ICAR Kharif Agro-Advisories for Farmers 2025",
        "url": "https://icar.gov.in/sites/default/files/Circulars/ICAR%20En-Kharif%20Agro-Advisories%20for%20Farmers%202025.pdf",
        "notes": "Captioned Leaf blast figure; extracted from a publication page.",
    },
    {
        "pdf": "icar_kharif_2025.pdf",
        "page_index": 106,
        "crop": (900, 1736, 1174, 2196),
        "label": "brown_spot",
        "filename": "icar_kharif_2025_p91_brown_spot_leaf.png",
        "publication_page": "91",
        "title": "ICAR Kharif Agro-Advisories for Farmers 2025",
        "url": "https://icar.gov.in/sites/default/files/Circulars/ICAR%20En-Kharif%20Agro-Advisories%20for%20Farmers%202025.pdf",
        "notes": "Leaf half of captioned Brown spot on leaves and panicles figure.",
    },
    {
        "pdf": "icar_annual_report_2022_23.pdf",
        "page_index": 107,
        "crop": (284, 2406, 735, 2938),
        "label": "bacterial_leaf_blight",
        "filename": "icar_annual_report_2022_23_p98_blb_untreated.png",
        "publication_page": "98",
        "title": "ICAR Annual Report 2022-23",
        "url": "https://icar.gov.in/sites/default/files/2025-04/ICAR-Annual-Report-2022-23-English.pdf",
        "notes": "Untreated-control half of a captioned BLB symptom comparison; whole-plant pot image, not a leaf close-up.",
    },
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    documents = {}
    for item in SOURCES:
        pdf_path = PDF_DIR / item["pdf"]
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)
        document = documents.setdefault(item["pdf"], pdfium.PdfDocument(pdf_path))
        page = document[item["page_index"]]
        rendered = page.render(scale=4).to_pil()
        destination = OUT / item["label"] / item["filename"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        rendered.crop(item["crop"]).save(destination)
        rows.append(
            {
                "file": str(destination.relative_to(ROOT)),
                "expected_label": item["label"],
                "source_title": item["title"],
                "publication_page": item["publication_page"],
                "source_url": item["url"],
                "notes": item["notes"],
            }
        )
    with (OUT / "sources.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Extracted {len(rows)} caption-verified examples to {OUT}")


if __name__ == "__main__":
    main()
