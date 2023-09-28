import os

import numpy as np
import torch
import safetensors.torch
import gguf


# copy from https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py#L16
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def load_model_from_file(model_path):
    print("loading model from {}".format(model_path))
    if model_path.lower().endswith(".safetensors"):
        pl_sd = safetensors.torch.load_file(model_path, device="cpu")
    else:
        pl_sd = torch.load(model_path, map_location="cpu")
    state_dict = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
    print("loading model from {} completed".format(model_path))
    return state_dict


def get_alpha_comprod(linear_start=0.00085, linear_end=0.0120, timesteps=1000):
    betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, timesteps, dtype=torch.float32) ** 2
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas.numpy(), axis=0)
    return torch.tensor(alphas_cumprod)


unused_tensors = [
    "betas",
]


def preprocess_weight(state_dict):
    new_state_dict = {}
    model = state_dict['model']

    for name, w in model.items():
        if not isinstance(w, torch.Tensor):
            continue
        skip = False
        for unused_tensor in unused_tensors:
            if name.startswith(unused_tensor):
                skip = True
                break
        if skip:
            continue
        if w.dtype == torch.bfloat16:
            w = w.to(torch.float16)
        new_state_dict[name] = w

    return new_state_dict


def convert(model_path, out_type=None, out_file=None):
    state_dict = load_model_from_file(model_path)

    weight = preprocess_weight(state_dict)

    layout_name = weight.keys()
    # output option
    # if out_type is None:
    #     weight = weight["ar_text_embedding.word_embeddings.weight"].numpy()
    #     if weight.dtype == np.float32:
    #         out_type = "f32"
    #     elif weight.dtype == np.float16:
    #         out_type = "f16"
    #     elif weight.dtype == np.float64:
    #         out_type = "f32"
    #     else:
    #         raise Exception("unsupported weight type %s" % weight.dtype)
    if out_file is None:
        out_file = os.path.splitext(os.path.basename(model_path))[0] + f"-{out_type}.gguf"
        out_file = os.path.join(os.getcwd(), out_file)
    block_count = 12
    head_count = 16
    context_length = 1024
    # PREFIX_MODE = 1
    # NUM_QUANTIZERS = 8
    # SAMPLE_RATE = 24000
    writer = gguf.GGUFWriter(out_file, "vallex")
    writer.add_name("vallex")
    writer.add_context_length(context_length)
    writer.add_tensor_data_layout("ar")  # kv tensor transform
    writer.add_head_count(head_count)
    writer.add_block_count(block_count)
    writer.add_file_type(gguf.GGUFValueType.get_type(out_type))

    for layers in weight.keys():
        data = weight[layers]
        data = data.squeeze().numpy()
        n_dims = len(data.shape)
        writer.add_tensor(name=layers, tensor=data)
    print("gguf: write header")
    writer.write_header_to_file()
    # print("gguf: write metadata")
    # writer.write_kv_data_to_file()
    print("gguf: write tensors")
    writer.write_tensors_to_file()
    writer.close()

    print(f"gguf: model successfully exported to '{out_file}'")
    print("")


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser(description="Convert Stable Diffuison model to GGML compatible file format")
    # parser.add_argument("--out_type", choices=["f32", "f16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0"], help="output format (default: based on input)")
    # parser.add_argument("--out_file", help="path to write to; default: based on input and current working directory")
    # parser.add_argument("model_path", help="model file path (*.pth, *.pt, *.ckpt, *.safetensors)")
    # args = parser.parse_args()
    convert("./vallex-checkpoint.pt", "q5_0", None)
