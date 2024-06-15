
import click
import fitz
from PIL import Image
from pathlib import Path
from tqdm import trange
import traceback

from zerollama.tasks.ocr.text_line_detection.engine.client import TLDClient
from zerollama.tasks.ocr.reading_order_detection.engine.client import RODClient
from zerollama.tasks.ocr.document_layout_analysis.engine.client import DLAClient
from zerollama.tasks.ocr.text_recognition.engine.client import TRClient

from zerollama.tasks.ocr.text_recognition.utils import get_annotated_image as tr_get_annotated_image
from zerollama.tasks.ocr.text_line_detection.utils import get_annotated_image as tld_get_annotated_image
from zerollama.tasks.ocr.reading_order_detection.utils import get_annotated_image as rod_get_annotated_image
from zerollama.tasks.ocr.document_layout_analysis.utils import get_annotated_image as dla_get_annotated_image

tld_model_name = "surya_tld"
rod_model_name = "surya_rod"
dla_model_name = "surya_dla"
tr_model_name = "surya_tr"

tld_client = TLDClient()
rod_client = RODClient()
dla_client = DLAClient()
tr_client = TRClient()


tld_client.wait_service_available(tld_model_name)
rod_client.wait_service_available(rod_model_name)
dla_client.wait_service_available(dla_model_name)
tr_client.wait_service_available(tr_model_name)


def ocr(image, lang=("zh", "en")):
    lines = tld_client.detection(tld_model_name, image)
    layout = dla_client.detection(dla_model_name, image, lines)
    order = rod_client.detection(rod_model_name, image, layout)
    text = tr_client.recognition(tr_model_name, image, lang, lines)

    return lines, layout, order, text


def annotate(image, langs, lines, layout, order, text):
    tr_image = tr_get_annotated_image(image, text, langs)
    tld_image = tld_get_annotated_image(image, lines)
    dla_image = dla_get_annotated_image(image, layout)
    rod_image = rod_get_annotated_image(image, order)
    return tr_image, tld_image, dla_image, rod_image


@click.command()
@click.argument('file_path', type=click.Path(exists=True))
def pdf2text(file_path):
    file_path = Path(file_path)
    pdf_path = file_path
    debug_path = file_path / "ocr_debug"
    debug_path.mkdir(exist_ok=True)

    for path in pdf_path.glob("*.pdf"):
        print(path.stem)

        (debug_path / path.stem).mkdir(exist_ok=True)

        with fitz.open(path) as pdf:
            for pg in trange(pdf.page_count):
                try:
                    page = pdf.load_page(pg)

                    mat = fitz.Matrix(2, 2)
                    pm = page.get_pixmap(matrix=mat, alpha=False)
                    if pm.width > 2000 or pm.height > 2000:
                        pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

                    image = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)

                    langs = ("zh", "en")

                    out = ocr(image, langs)

                    images = annotate(image, langs, *out)

                    (debug_path / path.stem / f"{pg}").mkdir(exist_ok=True)
                    for image, name in zip(images, ["tr", "tld", "dla", "rod"]):
                        image.save(debug_path / path.stem / f"{pg}" / f'{name}.jpg')

                except Exception:
                    traceback.print_exc()


@click.group()
def main():
    pass


main.add_command(pdf2text)


if __name__ == '__main__':
    main()










