from  mosaic_llm.mosaicllm import MosaicLLM
import mosaic_llm
import importlib
importlib.reload(mosaic_llm.mosaicllm)

mosaicllm = MosaicLLM(root="")
mosaicllm.run(query="climate")
