from pycuke.asg import *
from pycuke.asg2ir import gen_ir
from pycuke.helpers import *
from pycuke import run
import pycuke.codegen as codegen
import argparse
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--eb', type=float, default=0.0001)
parser.add_argument('--block_size', type=int, default=32) 
parser.add_argument('--num_blocks', type=int, default=202500) 
parser.add_argument('--dataset', type=str, default='./apps/compressor/dataset/CESM-ATM/AEROD_v_1_1800_3600.dat')
parser.add_argument('--compressed_data', type=str, default='./apps/compressor/compressed_data.pt')
args = parser.parse_args()

def szp_compression():
    eb = Var(name='eb', dtype='float')
    num_blocks = Var(name='num_blocks')
    block_size = Var(name='block_size')
    input = Tensor((num_blocks, block_size), name='input')
    
    # quantization
    recipPrecision = Const(0.5, 'float') / eb
    result_quant = input * recipPrecision
    cond = mask_bigger(result_quant, -0.5)
    quant_before_round = mask_if_else(
        cond = cond,
        then_val = result_quant + Const(0.5, 'float'),
        else_val = result_quant - Const(0.5, 'float'),
    )
    quant_round = cast(quant_before_round, 'int')
    
    # lorenzo
    result_lorenzo = quant_round.diff1d(axis=1)

    # fixed-length endcoding
    signs_bool = mask_bigger(result_lorenzo, 0)
    signs = cast(signs_bool, 'int')
    result_abs = result_lorenzo.abs()               
    max_values = result_abs.max(axis=1)
    fixed_lengths = (max_values + 1).log2().ceil()
    fixed_lengths_mask_bool = mask_bigger(fixed_lengths, 0)
    fixed_lengths_mask = cast(fixed_lengths_mask_bool, 'int')
    effective_bits = bitpack(result_abs, fixed_lengths)
    
    # encoding_fixed_lengths = fixed_lengths.view((num_blocks, ), ([0, 1], ))
    signs_pack = bitpack(signs, fixed_lengths_mask)
    signs_view = signs_pack.view((num_blocks*block_size, ), ([0, 1], ))
    num_effective_signs = fixed_lengths_mask.reduce(func=lambda accum, x: accum + x, \
                init=lambda a: setval(0, dest=a), \
                axis=0)
    encoding_signs = split_first(signs_view, num_effective_signs)
    effective_bits_view = effective_bits.view((num_blocks*block_size, ), ([0, 1], ))
    num_effective_floats = fixed_lengths.reduce(func=lambda accum, x: accum + x, \
                init=lambda a: setval(0, dest=a), \
                axis=0)
    encoding_effective_bits = split_first(effective_bits_view, num_effective_floats)
    output = concat(concat(fixed_lengths, encoding_signs), encoding_effective_bits)

    # generate CPU code
    code = codegen.cpu.print_cpp(gen_ir(output))
    print(code) 
        
    # compile and run
    input_data = np.fromfile(args.dataset, dtype=np.float32)
    input_tensor = torch.from_numpy(input_data)
    num_blocks = helpers.get_num_blocks(args.dataset, args.block_size)
    input_tensor = input_tensor.reshape(num_blocks, args.block_size)
    x = run.cpu.compile_and_run(code, num_blocks, args.block_size, input_tensor, args.eb)
    # print(x[:100])
    torch.save(x, './apps/compressor/compressed_data.pt')

def szp_decompression():
    # reverse encoding
    eb = Var(name='eb', dtype='float')
    block_size = Var(name='block_size')
    num_blocks = Var(name='num_blocks')
    input_size = Var(name='input_size')
    input = Tensor((input_size, ), name='input', dtype='int')
    fixed_lengths = split_first(input, num_blocks)
    signs_and_bits = split_second(input, num_blocks)
    num_effective_floats = fixed_lengths.reduce(func=lambda accum, x: accum + x, \
                init=lambda a: setval(0, dest=a), \
                axis=0)
    fixed_lengths_mask_bool = mask_bigger(fixed_lengths, 0)
    fixed_lengths_mask = cast(fixed_lengths_mask_bool, 'int')
    num_effective_signs = fixed_lengths_mask.reduce(func=lambda accum, x: accum + x, \
                init=lambda a: setval(0, dest=a), \
                axis=0)
    signs_pack = split_first(signs_and_bits, num_effective_signs)
    effective_bits_pack = split_second(signs_and_bits, num_effective_signs)
    output_abs = bitunpack(effective_bits_pack, fixed_lengths, block_size)
    signs = bitunpack(signs_pack, fixed_lengths_mask, block_size)
    cond = mask_bigger(signs, 0)
    residual = mask_if_else(
        cond = cond,
        then_val = output_abs,
        else_val = output_abs * (-1),
    ) 
    
    # reverse lorenzo
    quant_round = residual.prefix_sum(axis=1)

    # # reverse quantization
    output = cast(quant_round, 'float') * (2.0*eb)
    
    # generate CPU code
    code = codegen.cpu.print_cpp(gen_ir(output))
    print(code) 
        
    # compiler and run
    input_data = torch.load(args.compressed_data)
    input_size = input_data.shape[0]
    num_blocks = 202500
    x = run.cpu.compile_and_run(code, num_blocks, args.block_size, input_size, input_data, args.eb)
    torch.save(x, './apps/compressor/decompressed_data.pt')

if __name__ == "__main__":
    # szp_compression()
    szp_decompression()