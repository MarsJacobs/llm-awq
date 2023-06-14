from lm_eval import evaluator, tasks
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
from transformers import LlamaConfig, LlamaForCausalLM
import torch
import argparse
import os
import json
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from awq.utils.parallel import auto_parallel
from awq.quantize.pre_quant import run_awq, apply_awq
from awq.quantize.quantizer import pseudo_quantize_model_weight, real_quantize_model_weight
from awq.utils.lm_eval_adaptor import LMEvalAdaptor
from awq.utils.lm_mmlu import mmlu_eval
from awq.utils.wiki_loader import *

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, help='path of the hf model')
parser.add_argument('--mmlu_dir', type=str, default=None, help='path of the mmlu dataset')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument("--tasks", default=None, type=str)
parser.add_argument("--output_path", default=None, type=str)
parser.add_argument('--num_fewshot', type=int, default=0)
# model config
parser.add_argument('--parallel', action='store_true',
                    help="enable model parallelism")
parser.add_argument('--auto_parallel', action='store_true',
                    help="automatically set parallel and batch_size")
# quantization config
parser.add_argument('--w_bit', type=int, default=None)
parser.add_argument('--q_group_size', type=int, default=-1)
parser.add_argument('--no_zero_point', action='store_true',
                    help="disable zero_point")
parser.add_argument('--q_backend', type=str,
                    default="fake", choices=["fake", "real"])
# save/load real quantized weights
parser.add_argument('--dump_quant', type=str, default=None,
                    help='save quantized model')
parser.add_argument('--load_quant', type=str, default=None,
                    help='load quantized model')
# apply/save/load awq
parser.add_argument('--run_awq', action='store_true',
                    help="perform awq search process")
parser.add_argument('--mse_range', action='store_true',
                    help="perform awq search process (clip)")
parser.add_argument('--dump_awq', type=str, default=None,
                    help="save the awq search results")
parser.add_argument('--load_awq', type=str, default=None,
                    help="load the awq search results")
parser.add_argument('--q_format', type=str, default="uniform",
                    help="Quantization format", choices=["uniform", "minmag"])
    
args = parser.parse_args()

if args.auto_parallel:
    gpu_list = auto_parallel(args)

# get quantization config (apart from w_bit)
q_config = {
    "zero_point": not args.no_zero_point,  # by default True
    "q_group_size": args.q_group_size,  # whether to use group quantization
    "q_format": args.q_format
}
print("Quantization config:", q_config)

# build model and tokenizer

def build_model_and_enc(model_path):
    # if not os.path.exists(model_path):  # look into ssd
    #     raise FileNotFoundError(f"{model_path} not found!")
    print(f"* Building model {model_path}")

    # all hf model
    config = AutoConfig.from_pretrained(model_path)
    enc = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if args.load_quant:  # directly load quantized weights
        # no need to really load the fp16 weights... just to get the model structure
        print("Loading pre-computed quantized weights...")
        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(model_path, config=config,
                                                         torch_dtype=torch.float16)
        real_quantize_model_weight(
            model, w_bit=args.w_bit, q_config=q_config, init_only=True)
        model = load_checkpoint_and_dispatch(
            model, args.load_quant, device_map="balanced",
            # TODO: can we remove this?
            no_split_module_classes=[
                "OPTDecoderLayer", "LlamaDecoderLayer"]
        )
    else:  # fp16 to quantized
        kwargs = {"device_map": "balanced", "torch_dtype": torch.float16}

        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, **kwargs)

        if args.run_awq:
            awq_results = run_awq(
                model, enc,
                w_bit=args.w_bit, q_config=q_config,
                n_samples=128, seqlen=512, mse_range=args.mse_range
            )
            if args.dump_awq:
                torch.save(awq_results, args.dump_awq)
                print("AWQ results saved at", args.dump_awq)

        if args.load_awq:
            print("Loading pre-computed AWQ results from", args.load_awq)
            awq_results = torch.load(args.load_awq, map_location="cpu")
            apply_awq(model, awq_results)

        # weight quantization
        if args.w_bit is not None:
            if args.q_backend == "fake":
                assert args.dump_quant is None, \
                    "Need to use real quantization to dump quantized weights"
                pseudo_quantize_model_weight(
                    model, w_bit=args.w_bit, q_config=q_config
                )
            elif args.q_backend == "real":  # real quantization
                real_quantize_model_weight(
                    model, w_bit=args.w_bit, q_config=q_config
                )
                if args.dump_quant:
                    print(
                        f"Saving the quantized model at {args.dump_quant}...")
                    torch.save(model.cpu().state_dict(), args.dump_quant)
                    exit(0)
            else:
                raise NotImplementedError

    return model, enc


def main():
    if args.output_path is not None and os.path.exists(args.output_path):
        # print(f"Results {args.output_path} already generated. Exit.")
        print(f"Results {args.output_path} already generated. Overwrite.")
        # exit()

    if args.dump_awq and os.path.exists(args.dump_awq):
        print(f"Found existing AWQ results {args.dump_awq}, exit.")
        exit()

    # a hack here to auto set model group
    model, enc = build_model_and_enc(args.model_path)

    if args.tasks == "wikitext":

        model.seqlen=2048
        testloader = get_loaders(
                args.tasks, model=model, seqlen=model.seqlen, train=False, enc=enc
            )
        ppl_score = llama_eval(model, testloader, dev=model.device)
        print(f"wiki PPL : {ppl_score}")
        
        return 

    if args.mmlu_dir is not None:
        
        mmlu_eval(model, args.mmlu_dir, args, enc)

    if args.tasks is not None:
        
        lm_eval_model = LMEvalAdaptor(args.model_path, model, enc, args.batch_size)
        
        task_names = args.tasks.split(",")

        results = evaluator.simple_evaluate(
            model=lm_eval_model,
            tasks=task_names,
            batch_size=args.batch_size,
            no_cache=True,
            num_fewshot=args.num_fewshot,
        )
        print(evaluator.make_table(results))

        if args.output_path is not None:
            os.makedirs("awq_evals", exist_ok=True)
            args.output_path = os.path.join("awq_evals", args.output_path + ".json")
            # otherwise cannot save
            results["config"]["model"] = args.model_path
            with open(args.output_path, "w") as f:
                json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
