from ..asg import *
from ..ir import *
from .. import asg, ir
from ..helpers import get_obj, get_val, rebind_iterate, flatten_remove, ir_uses, remove_decl, clear_compute, \
    ir_find_defs, same_object, flatten, ASGTraversal, replace_all_ref, has_same_iteration_space, IRTraversal, ir_find_uses

import pycuke.codegen

def _top_level_loops(body):
    return [s for s in body if isinstance(s, ir.Loop)]

def _axis_len(dobj, axis):
    base = dobj
    while isinstance(base, ir.Indexing):
        base = base.dobject
    return base.size[axis]

def _clone_loop(loop):
    import copy
    return copy.deepcopy(loop)

def _concat_fuse_parent_child(parent_node: TensorOp, child_node: TensorOp):
    assert parent_node.op_type == 'concat' and child_node.op_type == 'concat'
    axis_parent = parent_node.operators[2].val
    axis_child  = child_node.operators[2].val
    if axis_parent != axis_child:
        return False

    p_loops = _top_level_loops(parent_node.compute)
    if len(p_loops) < 2:
        return False

    def _uses_child(loop):
        from ..helpers import ir_uses
        return ir_uses(loop, child_node.eval)

    left_is_child = _uses_child(p_loops[0])
    right_is_child = _uses_child(p_loops[1]) and (not left_is_child)

    if not (left_is_child or right_is_child):
        return False

    c_loops = _top_level_loops(child_node.compute)
    if len(c_loops) < 2:
        return False

    from_left = left_is_child
    other_inp = parent_node.operators[1] if from_left else parent_node.operators[0]
    other_len = _axis_len(other_inp.eval, axis_parent)

    target_loop_idx = 0 if from_left else 1
    target_loop = p_loops[target_loop_idx]

    offset_expr = ir.Literal(0, 'int') if from_left else other_len

    new_loop_A = _clone_loop(c_loops[0])
    new_loop_B = _clone_loop(c_loops[1])

    from ..helpers import IRTraversal, get_obj, replace_all_ref

    def _patch_segment(loop_seg):
        def action(stmt, res):
            if isinstance(stmt, ir.Assignment):
                base = get_obj(stmt.lhs)
                if same_object(base, child_node.eval):
                    import copy
                    new_lhs = copy.deepcopy(stmt.lhs)

                    cur = new_lhs
                    tail = []
                    while isinstance(cur, ir.Indexing):
                        tail.append(cur)
                        cur = cur.dobject
                    new_base = parent_node.eval
                    for idx_node in reversed(tail):
                        new_base = ir.Indexing(new_base, idx_node.idx)
                    new_lhs = new_base

                    def add_offset_on_axis(ix: ir.Indexing, dim_target, cur_dim=[0]):
                        if not isinstance(ix, ir.Indexing):
                            return ix
                        new_d = add_offset_on_axis(ix.dobject, dim_target, cur_dim)
                        if cur_dim[0] == dim_target:
                            new_idx = ir.Expr(offset_expr, ix.idx, '+')
                        else:
                            new_idx = ix.idx
                        cur_dim[0] += 1
                        return ir.Indexing(new_d, new_idx)

                    new_lhs = add_offset_on_axis(new_lhs, axis_parent, [0])
                    stmt.lhs = new_lhs
            return [True, True, True, True, True]

        IRTraversal(action)(loop_seg)

    _patch_segment(new_loop_A)
    _patch_segment(new_loop_B)

    parent_body = parent_node.compute
    idx_in_body = parent_body.index(target_loop)
    parent_body[idx_in_body:idx_in_body+1] = [new_loop_A, new_loop_B]
    clear_compute(child_node)
    remove_decl(child_node, child_node.eval)

    return True

def fuse_concat(node):
    if not (isinstance(node, TensorOp) and node.op_type == 'concat'):
        return
    if isinstance(node.operators[0], TensorOp) and node.operators[0].op_type == 'concat' \
       and node.operators[0].operators[2].val == node.operators[2].val \
       and len(node.operators[0].ref_by) == 1:
        _concat_fuse_parent_child(node, node.operators[0])
    if isinstance(node.operators[1], TensorOp) and node.operators[1].op_type == 'concat' \
       and node.operators[1].operators[2].val == node.operators[2].val \
       and len(node.operators[1].ref_by) == 1:
        _concat_fuse_parent_child(node, node.operators[1])

def _deepcopy_ir(x):
    try:
        return deepcopy(x)
    except Exception:
        return x

def _match_indexing_to_tensor(idx_expr, tensor_dobj):
    if isinstance(idx_expr, Indexing):
        base = idx_expr.dobject
        subs = [idx_expr.idx]
        if isinstance(base, Indexing):
            subs.insert(0, base.idx)
            base = base.dobject
        if isinstance(base, Ndarray) and base.dobject_id == tensor_dobj.dobject_id:
            return (len(subs), subs)
    return (0, [])

def _collect_lhs_loop_iters(lhs_indexing):
    iters = []
    cur = lhs_indexing
    while isinstance(cur, Indexing):
        if isinstance(cur.idx, Scalar):
            iters.insert(0, cur.idx)
        cur = cur.dobject
    return iters

def _inline_elementwise_into_bitpack(bit_node, prod_node):
    dfs = ir_find_defs(prod_node.compute, prod_node.eval)
    if not dfs:
        return False
    last_def = dfs[-1]
    if not isinstance(last_def, Assignment):
        return False
    prod_lhs = last_def.lhs
    prod_rhs = last_def.rhs

    prod_iters = _collect_lhs_loop_iters(prod_lhs)
    if not prod_iters:
        return False

    flat_loops = [s for s in bit_node.compute if isinstance(s, Loop)]
    if not flat_loops:
        return False
    flat_loop = flat_loops[-1]

    def _action(stmt, _):
        if isinstance(stmt, Assignment):
            return [False, True]

        if isinstance(stmt, Expr):
            return [True, True, False, False, False]

        if isinstance(stmt, Indexing):
            rank, subs = _match_indexing_to_tensor(stmt, prod_node.eval)
            if rank == 0:
                return [True, True, False, False, False]

            new_rhs = _deepcopy_ir(prod_rhs)
            for old_iter, new_iter in zip(prod_iters, subs):
                rebind_iterate(new_rhs, old_iter, new_iter)

            _new = _deepcopy_ir(new_rhs)
            stmt.__class__ = type(_new)
            stmt.__dict__.clear()
            stmt.__dict__.update(_new.__dict__)
            return [False, False, False, False, False]

        return [True, True, True, True, True]
    IRTraversal(_action)(flat_loop)
    clear_compute(prod_node)
    remove_decl(prod_node, prod_node.eval)

    return True

# TODO: reimplement this with IRTraversal
def _replace_arrindex_with_scalar(ir, old, new):
    if type(ir) == list or type(ir) == tuple:
        for l in ir:
            _replace_arrindex_with_scalar(l, old, new)
    elif type(ir) == Loop:
        _replace_arrindex_with_scalar(ir.body, old, new)
    elif type(ir) == FilterLoop:
        if type(ir.cond) in (Indexing, Scalar):
            obj = get_obj(ir.cond)
            if obj.dobject_id == old.dobject_id:
                ir.cond = new
        else:
            _replace_arrindex_with_scalar(ir.cond, old, new)
        _replace_arrindex_with_scalar(ir.cond_body, old, new)
        _replace_arrindex_with_scalar(ir.body, old, new)
    elif type(ir) == Expr:
        if type(ir.left) in (Indexing, Scalar):
            obj = get_obj(ir.left)
            if obj.dobject_id == old.dobject_id:
                ir.left = new
        else:
            _replace_arrindex_with_scalar(ir.left, old, new)
        if type(ir.right) in (Indexing, Scalar):
            obj = get_obj(ir.right)
            if obj.dobject_id == old.dobject_id:
                ir.right = new
        else:
            _replace_arrindex_with_scalar(ir.right, old, new)
    elif type(ir) == Assignment:
        if type(ir.lhs) in (Indexing, Scalar):
            obj = get_obj(ir.lhs)
            if obj.dobject_id == old.dobject_id:
                ir.lhs = new
        else:
            _replace_arrindex_with_scalar(ir.lhs, old, new)
        if type(ir.rhs) in (Indexing, Scalar):
            obj = get_obj(ir.rhs)
            if obj.dobject_id == old.dobject_id:
                ir.rhs = new
        else:
            _replace_arrindex_with_scalar(ir.rhs, old, new)
    elif type(ir) == Slice:
        if type(ir.start) in (Indexing, Scalar):
            obj = get_obj(ir.start)
            if obj.dobject_id == old.dobject_id:
                ir.start = new
        else:
            _replace_arrindex_with_scalar(ir.start, old, new)

        if type(ir.stop) in (Indexing, Scalar):
            obj = get_obj(ir.stop)
            if obj.dobject_id == old.dobject_id:
                ir.stop = new
        else:
            _replace_arrindex_with_scalar(ir.stop, old, new)

        if type(ir.step) in (Indexing, Scalar):
            obj = get_obj(ir.step)
            if obj.dobject_id == old.dobject_id:
                ir.step = new
        else:
            _replace_arrindex_with_scalar(ir.step, old, new)

    elif type(ir) == Math:
        if type(ir.val) in (Indexing, Scalar):
            obj = get_obj(ir.val)
            if obj.dobject_id == old.dobject_id:
                ir.val = new
        elif type(ir.val) in (list, tuple):
            new_val = []
            for i in ir.val:
                obj = get_obj(i)
                if type(obj) in (Indexing, Scalar, Ndarray, Literal, int):
                    if obj.dobject_id == old.dobject_id:
                        new_val.append(new)
                    else:
                        new_val.append(i)
                else:
                    new_val.append(i)
            ir.val = new_val
        else:
            _replace_arrindex_with_scalar(ir.val, old, new)
    elif type(ir) == Code:
        for k in ir.outputs:
            if type(ir.outputs[k]) in (Indexing, Scalar):
                obj = get_obj(ir.outputs[k])
                if obj.dobject_id == old.dobject_id:
                    ir.outputs[k] = new
        

        # TODO: replace inputs
        

def iterate_of_same_loops(x1, x2):
    if 'loop' in x1.attr and 'loop' in x2.attr:
        return match_orders([(0, x1.attr['loop'])], [(0, x2.attr['loop'])])
    return False

def same_expr_and_scalar(x1, x2):
    l1 = x1
    l2 = x2
    if isinstance(x1, Expr):
        if isinstance(x1.right, Literal) and x1.right.val == 0:
            l1 = x1.left
    if isinstance(x2, Expr):
        if isinstance(x2.right, Literal) and x2.right.val == 0:
            l2 = x2.left
    return same_object(l1,l2)

def match_orders(order1, order2):
    if len(order1) == len(order2):
        for i in range(len(order1)):
            x1 = get_val(order1[i][1].start)
            y1 = get_val(order1[i][1].end)
            z1 = get_val(order1[i][1].step)
            x2 = get_val(order2[i][1].start)
            y2 = get_val(order2[i][1].end)
            z2 = get_val(order2[i][1].step)
            if x1 == None or not (x1 == x2 or same_object(x1, x2) or iterate_of_same_loops(x1, x2) or same_expr_and_scalar(x1, x2)):
                return False
            if y1 == None or not (y1 == y2 or same_object(y1, y2) or iterate_of_same_loops(y1, y2) or same_expr_and_scalar(y1, y2)):
                return False
            if z1 == None and not (z1 == z2 or same_object(z1, z2) or iterate_of_same_loops(z1, z2) or same_expr_and_scalar(z1, z2)):
                return False
        return True
    else:
        return False


def merge_loops(order1, order2, data, this_node, input_node):
    # if match_orders(order1, order2): #change the orders here!
    if (1 == 1):
        for i in range(len(order1)):
            nl = order1[i][1]
            ol = order2[i][1]
            rebind_iterate(order2[i][1], ol.iterate, nl.iterate)
            if i < len(order1) - 1:
                nl.body[0:0] = [s for s in flatten(ol.body) if s != order2[i + 1][1]]
            if 'loop_ofs' in ol.attr:
                if 'loop_ofs' in nl.attr:
                    nl.attr['loop_ofs'] = max(nl.attr['loop_ofs'], ol.attr['loop_ofs'])
                else:
                    nl.attr['loop_ofs'] = ol.attr['loop_ofs']
            # for key in ol.attr:
            #     if key != 'loop_ofs':
            #         nl.attr[key] = ol.attr[key]

        dfs = ir_find_defs(order2[-1][1].body, data)
        if len(dfs) > 0:
            if ir_uses(dfs[-1], data):
                df = Scalar(data.dtype)
                input_node.decl.append(Decl(df))
            else:
                df = dfs[-1].rhs
                flatten_remove(order2[-1][1].body, dfs[-1])
            # print(pycuke.codegen.cpu.to_string(order1[-1][1]))
            if type(order1[-1][1]) == FilterLoop and data.dobject_id == get_obj(order1[-1][1].cond).dobject_id:
                order1[-1][1].cond_body.extend(order2[-1][1].body)
            else:
                j = len(order1[-1][1].body)
                for i in range(len(order1[-1][1].body)):
                    if ir_uses(order1[-1][1].body[i], data):
                        j = i
                        break
                order1[-1][1].body[j:j] = order2[-1][1].body
            _replace_arrindex_with_scalar(order1[-1][1], data, df)
            clear_compute(input_node)
            remove_decl(input_node, input_node.eval)
            if type(df) in (Scalar, Ndarray):
                pre_eval = input_node.eval
                input_node.eval = df
                # input_node.eval.attr['node'] = input_node
                if 'cache' in input_node.eval.attr:
                    cur = input_node.eval
                    while 'cache' in cur.attr:
                        cur = cur.attr['cache']
                    cur.attr['cache'] = pre_eval
                else:
                    input_node.eval.attr['cache'] = pre_eval

                if 'storage' in input_node.eval.attr:
                    input_node.eval.attr['storage'].append(pre_eval)
                else:
                    input_node.eval.attr['storage'] = [pre_eval]
        else:
            if type(order1[-1][1]) == FilterLoop and data == get_obj(order1[-1][1].cond):
                order1[-1][1].cond_body.extend(order2[-1][1].body)
            else:
                j = len(order1[-1][1].body)
                for i in range(len(order1[-1][1].body)):
                    if ir_uses(order1[-1][1].body[i], data):
                        j = i
                        break
                order1[-1][1].body[j:j] = order2[-1][1].body
                clear_compute(input_node)


######################################Satrt Matcha###################################
def _replace_statement(irs, old, new, body):
    if type(irs)==list:
        for ir in irs:
            _replace_statement(ir, old, new, irs)
    if type(irs)==Loop or type(irs)==FilterLoop:
        for ir in irs.body:
            _replace_statement(ir, old, new, irs.body)
    if type(irs)==Assignment:
        if new==None:
            if body.count(old)>0:
                body.remove(old)
        else:
            for i, n in enumerate(body):
                if n == old:
                    body[i] = new

def merge_sum_apply(sum_order, apply_order, apply_output, sum_node, apply_node):
    
    sum_loop = sum_order[0][1]
    apply_loop = apply_order[0][1]
    rebind_iterate(sum_loop, sum_loop.iterate, apply_loop.iterate)
    apply_output=get_obj(apply_output)
    assert(len(sum_node.compute)==2)
    assert(sum_loop==sum_node.compute[1])
    apply_loop_body = apply_loop.body
    sum_loop_body = sum_loop.body
    sum_init = sum_node.compute[0]
    
    #Revise counter
    if hasattr(apply_node, "counter"):
        apply_node.counter.decl.clear()
        apply_node.counter.compute.clear()
        counter_stmt = ir_find_defs(apply_loop_body, apply_node.counter.eval)
        flatten_remove(apply_loop_body, counter_stmt[0])
    
    #Revise Apply Node
    apply_data_defs_stmts = ir_find_defs(apply_loop_body, apply_output)
    sum_data_uses_stmts = ir_find_uses(sum_loop_body, apply_output)
    eliminated_var = sum_data_uses_stmts[-1].rhs
    assert(get_obj(apply_data_defs_stmts[-1].lhs).dobject_id == get_obj(sum_data_uses_stmts[-1].rhs).dobject_id)
    
    # print(pycuke.codegen.cpu.to_string(sum_data_uses_stmts[0].lhs))
    new_assign = Assignment(sum_data_uses_stmts[-1].lhs, apply_data_defs_stmts[-1].rhs, '+')
    _replace_statement(apply_loop_body, apply_data_defs_stmts[-1], new_assign, [])
    _replace_statement(apply_loop_body, apply_data_defs_stmts[0], None, [])
    _replace_statement(sum_loop_body, sum_data_uses_stmts[-1], None, [])

    # sum_data_uses_stmts[0].rhs = apply_data_defs_stmts[0].rhs
    # apply_data_defs_stmts[0].lhs = sum_data_uses_stmts[0].lhs
    # apply_data_defs_stmts[0].op = '+'

    remove_decl(sum_node, get_obj(eliminated_var))
    remove_decl(apply_node, get_obj(eliminated_var))
    
    clear_compute(sum_node)
    sum_node.compute.extend([sum_init])
    sum_node.compute.extend(apply_node.compute)
    apply_loop.attr = sum_loop.attr

    #Clear Apply Node
    clear_compute(apply_node)


def fuse_sum_apply(sum_node, sum_order, apply_node):
    merge_sum_apply(sum_order, apply_node.output_order, apply_node.eval, sum_node, apply_node)
    # if(apply_node.operators[1 + 2 * apply_node.nparams]==None):
    #     merge_loops(sum_order, apply_node.output_order, apply_node.eval, sum_node, apply_node)
    # else:
######################################End Matcha###################################


def fuse_operators(op1, order1, op2):
    if len(order1) > 0:
        merge_loops(order1, op2.output_order, op2.eval, op1, op2)
    else:
        dfs = ir_find_defs(op2.compute, op2.eval)
        if len(dfs) > 0:
            if not ir_uses(dfs[-1], op2.eval):
                df = dfs[-1].rhs
                flatten_remove(op2.compute, dfs[-1])
                op1.compute[0:0] = op2.compute
                _replace_arrindex_with_scalar(op1.compute, op2.eval, df)
                clear_compute(op2)
                remove_decl(op2, op2.eval)

def basic_rule(node, res):
    if type(node) == TensorOp and node.op_type in elementwise_op:
        if type(node.operators[0]) == TensorOp and node.operators[0].op_type in (
                elementwise_op + ['apply', 'einsum', 'setval', 'view', 'count_leading_zeros']) and len(
            node.operators[0].ref_by) == 1:
            fuse_operators(node, node.input_orders[0], node.operators[0])
        # if type(node.operators[0]) == TensorOp and 
        #   node.operators[0].op_type in (elementwise_op + ['apply', 'einsum', 'setval', 'view']) and 
        #   len(node.operators[0].ref_by) == 1:
        #     fuse_operators(node, node.input_orders[0], node.operators[0])
        #if type(node.operators[0]) == TensorOp and node.operators[0].op_type in (elementwise_op + ['apply', 'einsum', 'setval', 'view']) and len(node.operators[0].ref_by) == 1:

        if node.op_type in binary_elw:
            if type(node.operators[1]) == TensorOp and node.operators[1].op_type in (
                    elementwise_op + ['apply', 'einsum', 'setval', 'view']) and len(
                node.operators[1].ref_by) == 1:
                fuse_operators(node, node.input_orders[1], node.operators[1])

    elif type(node) == TensorOp and node.op_type == 'apply':
        cond = node.operators[1 + 2 * node.nparams]
        if cond != None:
            this_loop = node.output_order[0]
            fuse_operators(node, [this_loop], cond)

    elif type(node) == TensorOp and node.op_type == 'index':
        if type(node.operators[1]) == TensorOp and node.operators[1].op_type in (
                elementwise_op + ['setval']) and len(node.operators[1].ref_by) == 1:
            assert len(node.operators[1]._size()) == 0
            dfs = ir_find_defs(node.operators[1].compute, node.operators[1].eval)
            if len(dfs) > 0:
                if not ir_uses(dfs[-1], node.operators[1].eval):
                    df = dfs[-1].rhs
                    rebind_iterate(node.eval, node.operators[1].eval, df)
                    clear_compute(node.operators[1])
                    remove_decl(node.operators[1], node.operators[1].eval)

    elif type(node) == TensorOp and 'op_name' in node.attr and node.attr['op_name'] == 'sum':
        if type(node.operators[0]) == TensorOp and node.operators[0].op_type in (
                elementwise_op + ['apply', 'setval']) and len(node.operators[0].ref_by) == 1:
            fuse_sum_apply(node, node.input_orders[0], node.operators[0])
            #fuse_operators(node, node.input_orders[0], node.operators[0])
    
    elif type(node) == TensorOp and node.op_type == 'inline':
        for i in range(3, len(node.operators), 2):
            if type(node.operators[i]) == TensorOp and node.operators[i].op_type in (elementwise_op + ['setval']):
                assert len(node.operators[i]._size()) == 0
                dfs = ir_find_defs(node.operators[i].compute, node.operators[i].eval)
                if len(dfs) > 0:
                    if not ir_uses(dfs[-1], node.operators[i].eval):
                        df = dfs[-1].rhs
                        replace_all_ref(node.compute[0], node.operators[i].eval, df)
                        clear_compute(node.operators[i])
                        remove_decl(node.operators[i], node.operators[i].eval)
    
    elif type(node) == TensorOp and node.op_type == 'aggr':
        if type(node.operators[0]) == TensorOp and node.operators[0].op_type in (
                elementwise_op + ['apply', 'setval']) and len(node.operators[0].ref_by) == 1:
            fuse_operators(node, node.output_order, node.operators[0])

    # elif type(node) == TensorOp and node.op_type == 'reduce':
    #     if type(node.operators[0]) == TensorOp and node.operators[0].op_type in (
    #             elementwise_op + ['apply', 'setval', 'view']) and len(node.operators[0].ref_by) == 1:
    #         fuse_operators(node, node.output_order, node.operators[0])

    elif type(node) == TensorOp and node.op_type == 'reduce':
        prod = node.operators[0]
        if type(node.operators[0]) == TensorOp and node.operators[0].op_type in (
            elementwise_op + math_op + ['apply', 'setval', 'view', 'cast']) and len(node.operators[0].ref_by) == 1:
            fuse_operators(node, node.output_order, node.operators[0])

    elif type(node) == TensorOp and node.op_type == 'scan':
        if type(node.operators[0]) == TensorOp and node.operators[0].op_type in (
            elementwise_op + ['apply', 'setval', 'view', 'cast']) and len(node.operators[0].ref_by) == 1:
            fuse_operators(node, node.output_order, node.operators[0])

    elif type(node) == TensorOp and node.op_type == 'count_leading_zeros':
        if type(node.operators[0]) == TensorOp and node.operators[0].op_type in (
            elementwise_op + ['apply', 'setval', 'view', 'cast']) and len(node.operators[0].ref_by) == 1:
            fuse_operators(node, node.output_order, node.operators[0])

    elif type(node) == TensorOp and node.op_type == 'bitpack':
        inp = node.operators[0]
        if isinstance(inp, TensorOp) and inp.op_type in (elementwise_op + ['apply','setval','view','cast']) and len(inp.ref_by) == 1:
                ok = _inline_elementwise_into_bitpack(node, inp)
                if not ok:
                    order = node.input_orders[0] if len(node.input_orders) > 0 else []
                    fuse_operators(node, order, inp)
    elif type(node) == TensorOp and node.op_type == 'concat':
        fuse_concat(node)

