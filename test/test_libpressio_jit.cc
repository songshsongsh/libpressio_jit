#include "gtest/gtest.h"
#include <libpressio_jit.h>
#include <libpressio_ext/cpp/libpressio.h>

// TEST(libpressio_jit, poorjit) {
//     libpressio_jit_register_all();
//     pressio lib;
//     pressio_compressor comp = lib.get_compressor("poorjit");
//     if(!comp) {
//         GTEST_SKIP() << "poorjit not supported";
//         return;
//     }

//     std::string source {R"cpp(
// #include <boost/config.hpp>
// #include "std_compat/memory.h"
// #include "libpressio_ext/cpp/compressor.h"
// #include "libpressio_ext/cpp/data.h"
// #include "libpressio_ext/cpp/options.h"
// #include "libpressio_ext/cpp/pressio.h"

// class example_compressor_plugin : public libpressio_compressor_plugin {
// public:
//   struct pressio_options get_options_impl() const override
//   {
//     struct pressio_options options;
//     set(options, "example:a", a);
//     return options;
//   }

//   struct pressio_options get_configuration_impl() const override
//   {
//     struct pressio_options options;
//     set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
//     set(options, "pressio:stability", "experimental");
//     return options;
//   }

//   struct pressio_options get_documentation_impl() const override
//   {
//     struct pressio_options options;
//     set(options, "pressio:description", R"(example jit compressor)");
//     return options;
//   }


//   int set_options_impl(struct pressio_options const& options) override
//   {
//     get(options, "example:a", &a);
//     return 0;
//   }

//   int compress_impl(const pressio_data* input,
//                     struct pressio_data* output) override
//   {
//     *output = *input;
//     return 0;
//   }

//   int decompress_impl(const pressio_data* input,
//                       struct pressio_data* output) override
//   {
//     *output = *input;
//     return 0;
//   }

//   int major_version() const override { return 0; }
//   int minor_version() const override { return 0; }
//   int patch_version() const override { return 1; }
//   const char* version() const override { return "0.0.1"; }
//   const char* prefix() const override { return "example"; }

//   pressio_options get_metrics_results_impl() const override {
//     return {};
//   }

//   std::shared_ptr<libpressio_compressor_plugin> clone() override
//   {
//     return compat::make_unique<example_compressor_plugin>(*this);
//   }

//   int32_t a = 3;
// };

// extern "C" BOOST_SYMBOL_EXPORT example_compressor_plugin plugin;
// example_compressor_plugin plugin;
//     )cpp"};
    
//     comp->set_options({
//         {"template:source", source},
//         {"poorjit:pkgconfig", std::vector<std::string>{"libpressio_cxx"}}
//     });

//     pressio_data d(pressio_data::owning(pressio_float_dtype, {10, 10}));
//     pressio_data compressed(pressio_data::empty(pressio_byte_dtype, d.dimensions()));
//     pressio_data decompressed(pressio_data::owning(d.dtype(), d.dimensions()));
//     float* f = static_cast<float*>(d.data());
//     float* dec = static_cast<float*>(decompressed.data());
//     size_t stride = d.get_dimension(0);
//     for (int j = 0; j < d.get_dimension(1); ++j) {
//     for (int i = 0; i < d.get_dimension(0); ++i) {
//         f[j*stride+i] = i*j;
//         dec[j*stride+i] = 0.0;
//     }}

//     EXPECT_NE(d, decompressed) << "expected that the data would not equal";
//     ASSERT_EQ(comp->compress(&d, &compressed), 0) << "compression failed: "<< comp->error_msg();
//     ASSERT_EQ(comp->decompress(&compressed, &decompressed), 0) << "decompression failed: "<< comp->error_msg();
//     EXPECT_EQ(d, decompressed) << "expected that the data would be equal";
// }

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
        {"pycuke:code", frontend_code},
        {"pycuke:module_name", std::string("pycuke")},
        {"pycuke:entry_function", std::string("compile_source")},
        {"pycuke:python_paths", std::vector<std::string>{
            "/path/to/p3z"
        }},
        {"poorjit:pkgconfig", std::vector<std::string>{"libpressio_cxx"}}
    });
}