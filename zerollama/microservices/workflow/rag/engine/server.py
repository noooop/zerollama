
from inspect import isgenerator
from zerollama.core.framework.zero.server import Z_MethodZeroServer
from zerollama.microservices.workflow.rag.protocol import PROTOCOL
from zerollama.microservices.workflow.rag.protocol import RAGRequest, ZeroServerResponseOk, ZeroServerStreamResponseOk
from zerollama.microservices.workflow.rag.online import rag, default_qa_prompt_tmpl_str


class ZeroRAGEngine(Z_MethodZeroServer):
    def __init__(self, **kwargs):
        kwargs.pop("name", None)
        Z_MethodZeroServer.__init__(self, name="rag", protocol=PROTOCOL,
                                    port=None, do_register=True, **kwargs)

    def init(self):
        print(f"{self.__class__.__name__}: is running!", "port:", self.port)

    def z_rag(self, req):
        data = RAGRequest(**req.data)
        kwargs = data.dict()
        return_references = kwargs.pop("return_references", False)

        response, references = rag(**kwargs)

        if isgenerator(response):
            rep_id = 0
            if return_references:
                rep = ZeroServerStreamResponseOk(msg={"references": references},
                                                 snd_more=True,
                                                 rep_id=rep_id)
                self.zero_send(req, rep)
                rep_id += 1

            for msg in response:
                rep = ZeroServerStreamResponseOk(msg=msg,
                                                 snd_more=hasattr(msg, "delta_content"),
                                                 rep_id=rep_id)
                self.zero_send(req, rep)
                rep_id += 1
        else:
            msg = {"answer": response.dict()}
            if return_references:
                msg["references"] = references
            rep = ZeroServerResponseOk(msg=msg)
            self.zero_send(req, rep)

    def z_default_qa_prompt_tmpl(self, req):
        rep = ZeroServerResponseOk(msg={"default_qa_prompt_tmpl": default_qa_prompt_tmpl_str})
        self.zero_send(req, rep)


