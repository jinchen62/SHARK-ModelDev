import argparse
from transformers import AutoModelForCausalLM
import safetensors
from iree.compiler.ir import Context
import torch
import shark_turbine.aot as aot
from shark_turbine.aot import *
from turbine_models.custom_models.sd_inference import utils
from turbine_models.model_builder import HFTransformerBuilder


parser = argparse.ArgumentParser()
parser.add_argument(
    "--hf_model_name",
    type=str,
    help="HF model name",
    default="BAAI/bge-base-en-v1.5",
)
parser.add_argument(
    "--hf_auth_token",
    type=str,
    help="The Hugging Face auth token, required",
)
parser.add_argument("--compile_to", type=str, default="linalg", help="torch, linalg, vmfb")
parser.add_argument(
    "--external_weights",
    type=str,
    default=None,
    help="saves ir/vmfb without global weights for size and readability, options [gguf, safetensors]",
)
parser.add_argument("--external_weight_path", type=str)
parser.add_argument("--device", type=str, default="cpu", help="cpu, cuda, vulkan, rocm")
# TODO: Bring in detection for target triple
parser.add_argument(
    "--iree_target_triple",
    type=str,
    default="host",
    help="Specify vulkan target triple or rocm/cuda target device.",
)
parser.add_argument("--vulkan_max_allocation", type=str, default="4294967296")


def export_bert_model(
    hf_model_name,
    hf_auth_token=None,
    compile_to="linalg",
    external_weights=None,
    external_weight_path=None,
    device=None,
    target_triple=None,
    max_alloc=None,
):
    model = HFTransformerBuilder(hf_id=hf_model_name, hf_auth_token=hf_auth_token)
    model.model.config.pad_token_id = None

    mapper = {}
    utils.save_external_weights(
        mapper, model, external_weights, external_weight_path
    )

    class BgeModel(CompiledModule):
        if external_weights:
            params = export_parameters(
                model.model, external=True, external_scope="", name_mapper=mapper.get
            )
        else:
            params = export_parameters(model.model)

        def run_forward(self, inp=AbstractTensor(1, None, dtype=torch.int64)):
            constraints = [inp.dynamic_dim(1) <= 512]
            return jittable(model.model.forward)(inp, constraints=constraints)

    import_to = "INPUT" if compile_to == "linalg" else "IMPORT"
    inst = BgeModel(context=Context(), import_to=import_to)
    module_str = str(CompiledModule.get_mlir_module(inst))

    safe_name = utils.create_safe_name(hf_model_name, "")
    if compile_to == "vmfb":
        utils.compile_to_vmfb(module_str, device, target_triple, max_alloc, safe_name)

    with open(f"{safe_name}.mlir", "w+") as f:
        f.write(module_str)
    print("Saved to", safe_name + ".mlir")


if __name__ == "__main__":
    args = parser.parse_args()
    export_bert_model(
        args.hf_model_name,
        args.hf_auth_token,
        args.compile_to,
        args.external_weights,
        args.external_weight_path,
        args.device,
        args.iree_target_triple,
        args.vulkan_max_allocation,
    )
