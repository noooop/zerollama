
from zerollama.core.framework.zero.server import ZeroServerProcess
from zerollama.tasks.ocr.document_layout_analysis.protocol import DLA_ENGINE_CLASS
from zerollama.tasks.ocr.text_line_detection.protocol import TLD_ENGINE_CLASS
from zerollama.tasks.ocr.reading_order_detection.protocol import ROD_ENGINE_CLASS
from zerollama.tasks.ocr.text_recognition.protocol import TR_ENGINE_CLASS

if __name__ == '__main__':
    nameserver = ZeroServerProcess("zerollama.core.framework.nameserver.server:ZeroNameServer")

    engines = []
    for engine_class, name in [(DLA_ENGINE_CLASS, "surya_dla"),
                               (TLD_ENGINE_CLASS, "surya_tld"),
                               (TR_ENGINE_CLASS, "surya_tr"),
                               (ROD_ENGINE_CLASS, "surya_rod")]:
        engine = ZeroServerProcess(engine_class,
                                   server_kwargs={"name": name})
        engines.append(engine)

    nameserver.start()
    for engine in engines:
        engine.start()

    nameserver.wait()
    for engine in engines:
        engine.wait()