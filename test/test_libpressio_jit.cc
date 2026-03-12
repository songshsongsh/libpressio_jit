#include "gtest/gtest.h"
#include <libpressio_jit.h>
#include <libpressio_ext/cpp/libpressio.h>

TEST(libpressio_jit, poorjit_pycuke) {
    libpressio_jit_register_all();

    pressio lib;
    pressio_compressor comp = lib.get_compressor("poorjit");
    if (!comp) {
        GTEST_SKIP() << "poorjit not supported";
        return;
    }

    std::string frontend_code = R"py(
def build_graph():
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
    
    quant_round = residual.prefix_sum(axis=1)

    output = cast(quant_round, 'float') * (2.0*eb)
    return output
)py";

    int rc = comp->set_options({
        {"poorjit:generator", std::string("pycuke")},
        {"pycuke:frontend_code", frontend_code},
        {"pycuke:module_name", std::string("pycuke")},
        {"pycuke:entry_function", std::string("compile_source")},
        {"pycuke:python_paths", std::vector<std::string>{
            "/data/not_backed_up/ssong10/libpressio_dev/envs/libpressio-dev/libpressio-jit/p3z"
        }},
        {"poorjit:pkgconfig", std::vector<std::string>{"libpressio_cxx"}}
    });
}