#include <libpressio_jit_ext/cpp/generator.h>

#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include <mutex>
#include <string>
#include <vector>
#include <fstream>

namespace py = pybind11;

namespace libpressio_jit { namespace pycuke_generator_ns {

namespace {
void ensure_python_started() {
    static std::once_flag once;
    static std::unique_ptr<py::scoped_interpreter> guard;
    std::call_once(once, []() {
        guard = std::make_unique<py::scoped_interpreter>();
    });
}
}

class pycuke_generator final : public pressio_generator_plugin {
public:
    std::string generate_impl() override {
        try {
            ensure_python_started();

            py::gil_scoped_acquire gil;

            py::module_ sys = py::module_::import("sys");
            py::list sys_path = sys.attr("path");

            for (auto const& p : python_paths) {
                bool found = false;
                for (auto const& existing : sys_path) {
                    if (py::cast<std::string>(existing) == p) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    sys_path.attr("insert")(0, p);
                }
            }

            py::dict globals;
            py::dict locals;
            py::module_ asg = py::module_::import("pycuke.asg");
            py::module_ asg2ir = py::module_::import("pycuke.asg2ir");
            py::module_ codegen = py::module_::import("pycuke.codegen");

            globals["Var"] = asg.attr("Var");
            globals["Const"] = asg.attr("Const");
            globals["Tensor"] = asg.attr("Tensor");
            globals["setval"] = asg.attr("setval");
            globals["mask_bigger"] = asg.attr("mask_bigger");
            globals["mask_if_else"] = asg.attr("mask_if_else");
            globals["cast"] = asg.attr("cast");
            globals["bitpack"] = asg.attr("bitpack");
            globals["bitunpack"] = asg.attr("bitunpack");
            globals["split_first"] = asg.attr("split_first");
            globals["split_second"] = asg.attr("split_second");
            globals["concat"] = asg.attr("concat");
            globals["__builtins__"] = py::module_::import("builtins");

            py::exec(frontend_code, globals, locals);

            py::object build_fn;
            if (locals.contains("build_graph")) build_fn = locals["build_graph"];
            else if (globals.contains("build_graph")) build_fn = globals["build_graph"];
            else throw std::runtime_error("frontend_code must define build_graph()");

            py::object output = build_fn();

            py::object ir = asg2ir.attr("gen_ir")(output);
            py::object cpp = codegen.attr("cpu").attr("print_cpp")(ir);
            std::string code = cpp.cast<std::string>();

            return code;

        } catch (const py::error_already_set& e) {
            set_error(1, std::string("python error: ") + e.what());
            return "";
        } catch (const std::exception& e) {
            set_error(1, std::string("pycuke generator error: ") + e.what());
            return "";
        }
    }

    int set_options(pressio_options const& opts) override {
        get(opts, "pycuke:code", &frontend_code);
        get(opts, "pycuke:module_name", &module_name);
        get(opts, "pycuke:entry_function", &entry_function);
        get(opts, "pycuke:python_paths", &python_paths);

        if (frontend_code.empty()) {
            return set_error(1, "pycuke:code is required");
        }

        if (module_name.empty()) {
            module_name = "pycuke";
        }

        if (entry_function.empty()) {
            entry_function = "compile_source";
        }

        return 0;
    }

    pressio_options get_options() const override {
        pressio_options opts;
        set(opts, "pycuke:code", frontend_code);
        set(opts, "pycuke:module_name", module_name);
        set(opts, "pycuke:entry_function", entry_function);
        set(opts, "pycuke:python_paths", python_paths);
        return opts;
    }

    pressio_options get_documentation() const override {
        pressio_options opts;
        set(opts, "pressio:description",
            R"(generate full libpressio plugin source by calling a Python compiler)");
        set(opts, "pycuke:code",
            "Python frontend/DSL definition passed into the Python compiler");
        set(opts, "pycuke:entry_function",
            "Python function name to call, default = compile_source");
        set(opts, "pycuke:python_paths",
            "Additional entries inserted into sys.path before importing the module");
        return opts;
    }

    pressio_options get_configuration_impl() const override {
        pressio_options opts;
        set(opts, "pressio:thread_safe", pressio_thread_safety_serialized);
        set(opts, "pressio:stability", "experimental");
        set(opts, "pressio:highlevel",
            std::vector<std::string>{
                "pycuke:code",
            });
        return opts;
    }

    const char* prefix() const final {
        return "pycuke";
    }

    std::unique_ptr<pressio_generator_plugin> clone() override {
        return std::make_unique<pycuke_generator>(*this);
    }

private:
    std::string frontend_code;
    std::string module_name = "pycuke";
    std::string entry_function = "compile_source";
    std::vector<std::string> python_paths;
};

} // namespace pycuke_generator_ns

static pressio_register pycuke_generator_plugin(
    generator_plugins(),
    "pycuke",
    []() {
        return std::make_unique<pycuke_generator_ns::pycuke_generator>();
    }
);

} // namespace libpressio_jit