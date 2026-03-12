from ..asg import TensorOp, elementwise_op, Tensor
from ..ir import Ndarray
from ..helpers import (
    ASGTraversal,
    replace_all_ref,
    remove_decl,
    is_same_size,
    has_same_iteration_space,
)

def _orders_match(op1: TensorOp, op2: TensorOp) -> bool:
    o1 = getattr(op1, "output_order", [])
    o2 = getattr(op2, "output_order", [])
    if len(o1) != len(o2):
        return False
    for (_, l1), (_, l2) in zip(o1, o2):
        if not has_same_iteration_space(l1, l2):
            return False
    return True

def _program_wide_replace(ast_root, old_buf, new_buf):
    def action(n, _):
        if hasattr(n, "compute"):
            replace_all_ref(n.compute, old_buf, new_buf)
        if hasattr(n, "decl"):
            replace_all_ref(n.decl, old_buf, new_buf)
    ASGTraversal(action)(ast_root)

def _all_tensor_inputs(op: TensorOp):
    return [x for x in op.operators if isinstance(x, (TensorOp, Tensor))]

def memory_reuse(ast_root):
    nodes = ASGTraversal(lambda n, res: res.append(n) if isinstance(n, TensorOp) else None)(ast_root)
    ops = [n for n in nodes if isinstance(n, TensorOp)]

    changed = True
    while changed:
        changed = False

        for cons in ops:
            if cons.op_type not in elementwise_op or not isinstance(getattr(cons, "eval", None), Ndarray):
                continue

            for prod in _all_tensor_inputs(cons):
                if not isinstance(prod, TensorOp):
                    continue
                if prod.op_type not in elementwise_op or not isinstance(getattr(prod, "eval", None), Ndarray):
                    continue

                if len(prod.ref_by) != 1 or prod.ref_by[0] is not cons:
                    continue
                if not is_same_size(prod.ref_size, cons.ref_size):
                    continue
                if prod.dtype != cons.dtype:
                    continue
                if not _orders_match(prod, cons):
                    continue

                old_out = cons.eval
                new_out = prod.eval
                if old_out is new_out:
                    continue

                _program_wide_replace(ast_root, old_out, new_out)
                cons.eval = new_out
                remove_decl(cons, old_out)

                new_out.attr.setdefault("storage", []).append(old_out)

                changed = True
                break
            if changed:
                break

        if changed:
            continue

        for prod in ops:
            if prod.op_type not in elementwise_op or not isinstance(getattr(prod, "eval", None), Ndarray):
                continue

            candidates = []
            for inp in _all_tensor_inputs(prod):
                if not isinstance(getattr(inp, "eval", None), Ndarray):
                    continue
                if not is_same_size(inp._size() if isinstance(inp, Tensor) else inp.ref_size, prod.ref_size):
                    continue
                if inp.dtype != prod.dtype:
                    continue
                candidates.append(inp)
            if not candidates:
                continue

            main_inp = candidates[0]

            if isinstance(main_inp, TensorOp):
                if len(main_inp.ref_by) != 1 or main_inp.ref_by[0] is not prod:
                    continue
            else:  # Tensor
                if len(main_inp.ref_by) != 1 or main_inp.ref_by[0] is not prod:
                    continue

            if not _orders_match(prod, prod):
                continue

            old_out = prod.eval
            new_out = main_inp.eval
            if old_out is new_out:
                continue

            _program_wide_replace(ast_root, old_out, new_out)
            prod.eval = new_out
            remove_decl(prod, old_out)
            new_out.attr.setdefault("storage", []).append(old_out)

            changed = True
            break

    return ast_root
