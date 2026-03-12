from __future__ import annotations
import copy
from . import ir
from . import asg
from . import helpers


def num_unbind(index):
    if type(index) == ir.Indexing:
        return num_unbind(index.dobject) + num_unbind(index.idx)
    elif type(index) == ir.Literal and index.val == -1:
        return 1
    else:
        return 0

def bind(object: ir.Indexing | ir.Ndarray | ir.Slice, subscripts: list | tuple, attrs = None):
    new_index = copy.deepcopy(object)
    # if attrs == None:
    #     attrs = [{} for _ in range(len(subscripts))]
    j = 0
    if type(new_index) == ir.Indexing:
        indices = [new_index]
        while type(indices[-1].dobject) == ir.Indexing:
            indices.append(indices[-1].dobject)
        indices.reverse()
        i = 0
        while i < len(indices) and j < len(subscripts):
            index = indices[i]
            i += 1
            while type(index.idx) == ir.Indexing:
                index = index.idx
            assert type(index.idx) in (ir.Scalar, ir.Literal, ir.Expr)
            if type(index.idx) == ir.Scalar or type(index.idx) == ir.Expr or (type(index.idx) == ir.Literal and index.idx.val != -1):
                continue
            idx = subscripts[j]
            if type(idx) in (ir.Scalar, ir.Literal, ir.Indexing, ir.Expr):
                index.idx = idx
            elif type(idx) in (ir.Ndarray, ir.Slice):
                index.idx = ir.Indexing(idx, ir.Literal(-1, 'int'))
            else:
                raise TypeError('idx type error when binding')
            # index.attr.update(attrs[j])
            index.refresh_size()
            j += 1

    while j < len(subscripts):
        idx = subscripts[j]
        if type(idx) in (ir.Scalar, ir.Literal, ir.Indexing, ir.Expr):
            new_index = ir.Indexing(new_index, idx)
        elif type(idx) in (ir.Ndarray, ir.Slice):
            new_index = ir.Indexing(new_index, ir.Indexing(idx, ir.Literal(-1, 'int')))
        else:
            raise TypeError('incorrect idx type!')
        # new_index.attr.update(attrs[j])
        j += 1
    if type(new_index) == ir.Indexing:
        new_index.refresh_size()
    return new_index



def get_slice(index: (ir.Indexing, ir.Ndarray, ir.Slice)):
    if type(index) == ir.Indexing:
        x = get_slice(index.dobject)
        if x != None:
            return x
        else:
            y = get_slice(index.idx)
            if y != None:
                return y
            else:
                if type(index.dobject) == ir.Slice and type(index.idx) == ir.Literal and index.idx.val == -1:
                    return index.dobject
    return None


def replace_output(stmt, old, new):
    if type(stmt) == list or type(stmt) == tuple:
        for l in stmt:
            replace_output(l, old, new)
    elif type(stmt) == ir.Loop:
        replace_output(stmt.body, old, new)
    elif type(stmt) == ir.Assignment:
        if stmt.lhs == old:
            stmt.lhs = new
        else:
            replace_output(stmt.lhs, old, new)
    elif type(stmt) == ir.Indexing:
        if stmt.dobject == old:
            stmt.dobject = new
        else:
            replace_output(stmt.dobject, old, new)
    elif type(stmt) == ir.Ndarray:
        if stmt == old:
            stmt = new
        


def list_product(l):
    res = ir.Literal(1, dtype='int')
    for x in l:
        if not type(x) in (tuple, list):
            res = ir.Expr(res, x, '*')
        else:
            res = ir.Expr(res, list_product(x), '*')
    return res

def get_subdims(x, dims, sizes):
    res = dict()
    y = x
    for i in range(len(sizes)):
        d = dims[len(sizes) - 1 - i]
        s = sizes[len(sizes) - 1 - i]
        if not type(d) in (tuple, list):
            if d != -1:
                if d not in res:
                    res[d] = [(ir.Expr(y, s, '%') if i < len(sizes) - 1 else y, s)]
                else:
                    res[d].insert(0, (ir.Expr(y, s, '%') if i < len(sizes) - 1 else y, s))
        else:
            ts = list_product(s)
            sd = get_subdims(ir.Expr(y, ts, '%') if i < len(sizes) - 1 else y, d, s)
            for k in sd:
                if k not in res:
                    res[k] = sd[k]
                else:
                    res[k][0:0] = sd[k]
        y = ir.Expr(y, s, '/')

    return res

def resolve_view(node, subscripts):
    if isinstance(node, asg.Tensor):
        if 'dim_map' in node.attr and 'size_map' in node.attr:
            dim_map = node.attr['dim_map']
            size_map = node.attr['size_map']
            assert len(dim_map) == len(subscripts)
            assert helpers.list_same_size(dim_map, size_map)

            orig_subscripts = dict()
            for i in range(len(dim_map)):
                d = dim_map[i]
                if type(d) in (list, tuple):
                    t = get_subdims(subscripts[i], d, size_map[i])
                    for k in t:
                        if k != -1:
                            if k in orig_subscripts:
                                orig_subscripts[k].extend(t[k])
                            else:
                                orig_subscripts[k] = t[k]
                elif d != -1: # remove fake axes
                    if d in orig_subscripts:
                        orig_subscripts[d].append((subscripts[i], size_map[i]))
                    else:
                        orig_subscripts[d] = [(subscripts[i], size_map[i])]

            res_subscripts = []
            for k in sorted(orig_subscripts.keys()):
                sg = [tmp[0] for tmp in orig_subscripts[k]]
                ds = [tmp[1] for tmp in orig_subscripts[k]]
                orig_s = sg[0]

                for j in range(1, len(sg)):
                    if type(orig_s) in (ir.Scalar, ir.Literal, ir.Indexing, ir.Ndarray, ir.Expr):
                        if type(sg[j]) in (ir.Scalar, ir.Literal, ir.Indexing, ir.Ndarray, ir.Expr):
                            orig_s = ir.Expr(ir.Expr(orig_s, ds[j], '*'), sg[j], '+')
                        elif type(sg[j]) == ir.Slice:
                            start = ir.Expr(ir.Expr(orig_s, ds[j], '*'), sg[j].start, '+')
                            stop = ir.Expr(ir.Expr(orig_s, ds[j], '*'), sg[j].stop, '+')
                            orig_s = ir.Slice(start, stop, sg[j].step)
                        else:
                            raise TypeError('idx type error')
                    elif type(orig_s) == ir.Slice:
                        if type(sg[j]) in (ir.Scalar, ir.Literal, ir.Indexing, ir.Ndarray, ir.Expr):
                            start = ir.Expr(ir.Expr(orig_s.start, ds[j], '*'), sg[j], '+')
                            stop = ir.Expr(
                                ir.Expr(ir.Expr(ir.Expr(orig_s.stop, 1, '-'), ds[j], '*'), sg[j], '+'), 1,
                                '+')
                            step = ir.Expr(orig_s.step, ds[j], '*')
                            orig_s = ir.Slice(start, stop, step)
                        elif type(sg[j]) == ir.Slice:
                            start = ir.Expr(ir.Expr(orig_s.start, ds[j], '*'), sg[j].start, '+')
                            stop = ir.Expr(ir.Expr(ir.Expr(orig_s.stop, 1, '-'), ds[j], '*'), sg[j].stop, '+')
                            assert (orig_s.step == 1 or orig_s.step.val == 1) and (sg[j].step == 1 or sg[
                                j].step.val == 1), 'view does not support non-continuous slice of slice'
                            orig_s = ir.Slice(start, stop, 1)
                        else:
                            raise TypeError('idx type error')
                    else:
                        raise TypeError('orig type error')
                res_subscripts.append(orig_s)
            return res_subscripts
    return subscripts


def gen_ir(node):
    assert isinstance(node, asg.ASTNode)
    if node.eval or len(node.decl) > 0 or (type(node) == asg.TensorOp and len(node.compute) > 0):
        return node
    if type(node) == asg.Const:
        if node.dtype != 'slice':
            assert type(node.val) == int or type(node.val) == float
            node.eval = ir.Literal(node.val, node.dtype)
        else:
            gen_ir(node.val.start)
            gen_ir(node.val.stop)
            gen_ir(node.val.step)
            node.eval = ir.Slice(node.val.start.eval, node.val.stop.eval, node.val.step.eval)


    elif type(node) == asg.Var or (type(node) == asg.Tensor and len(node._size()) == 0):
        node.eval = ir.Scalar(node.dtype, node.name)
        for key in node.attr:
            node.eval.attr[key] = node.attr[key]
        node.decl = [ir.Decl(node.eval)]

    elif type(node) == asg.Tensor and len(node._size()) > 0:
        # convert AST sizes to IR sizes
        size = helpers.get_ir_of_size(node._size())
        node.eval = ir.Ndarray(node.dtype, size, node.name)
        for key in node.attr:
            node.eval.attr[key] = node.attr[key]
        node.decl = [ir.Decl(node.eval)]

    elif type(node) == asg.TensorOp:
        if node.op_type in asg.arith_op or node.op_type in asg.cmp_op:
            # arith_op and cmp_op are binary operations, we generate the two operands first
            gen_ir(node.operators[0])
            gen_ir(node.operators[1])
            assert isinstance(node.operators[0], asg.Tensor) and isinstance(node.operators[1], asg.Tensor)

            if node.op_type in asg.arith_op:
                op = asg.arith_op[node.op_type]
            else:
                op = node.op_type

            if len(node._size()) > 0:  # if output has >=1 dimensions, it should be stored in an Ndarray
                size = helpers.get_ir_of_size(node._size())
                node.eval = ir.Ndarray(node.dtype, size)

                node.decl = [ir.Decl(node.eval)]


                left_levels = len(node.operators[0]._size())
                right_levels = len(node.operators[1]._size())
                max_levels = max(left_levels, right_levels)
                assert max_levels == len(size)

                lhs = node.operators[0].eval
                rhs = node.operators[1].eval
                res = node.eval
                compute = node.compute

                lhs_subscripts = []
                rhs_subscripts = []
                res_subscripts = []

                if compute == []:
                    par_loop = None
                else:
                    par_loop = compute[0]
                for level in range(max_levels):

                    # handle out of bound slicing
                    # left_slice = get_slice(lhs)
                    # right_slice = get_slice(rhs)
                    # left_attr = {}
                    # if left_slice != None and type(left_slice.start) == ir.Literal:
                    #     if left_slice.start.val < 0:
                    #         left_ofs = -left_slice.start.val
                    #         left_attr['slice_ofs'] = left_ofs
                    #     else:
                    #         left_ofs = 0
                    # else:
                    #     left_ofs = 0
                    # right_attr = {}
                    # if right_slice != None and type(right_slice.start) == ir.Literal:
                    #     if right_slice.start.val < 0:
                    #         right_ofs = -right_slice.start.val
                    #         right_attr['slice_ofs'] = right_ofs
                    #     else:
                    #         right_ofs = 0
                    # else:
                    #     right_ofs = 0
                    
                    # pre_loop = ir.Loop(0, size[level], 1, [])
                    if node.attr.get('scan') and level == node.operators[-1]: 
                        if not node.attr.get('inclusive'):
                            pre_loop = ir.Loop(0, size[level].val-1, 1, [])
                        else:
                            pre_loop = ir.Loop(1, size[level].val, 1, []) 
                    else:
                        pre_loop = ir.Loop(0, size[level], 1, [])
                        # add openmp
                        if level == 0:                  
                            pre_loop.attr['ptype'] = 'naive'
                            pre_loop.attr['plevel'] = 0
                            pre_loop.attr['nprocs'] = {0: [16]} 

                    
                    # loop_ofs = max(left_ofs, right_ofs)
                    # if loop_ofs > 0:
                    #     pre_loop.attr['loop_ofs'] = loop_ofs

                    if level < left_levels:
                        lhs_subscripts.append(pre_loop.iterate)
                        node.input_orders[0].append((level, pre_loop))
                    if node.op_type in asg.cmp_op and level < right_levels:
                        rhs_subscripts.append(pre_loop.iterate)
                        node.input_orders[1].append((level, pre_loop))
                    if node.op_type in asg.arith_op and right_levels!=0 and right_levels < left_levels and level >= right_levels:
                        if node.attr.get('inclusive') and level == node.operators[-1]:
                            rhs_subscripts.append(ir.Scalar(dtype = pre_loop.iterate.dtype, name = pre_loop.iterate.__name__+'-1'))
                        else:
                            rhs_subscripts.append(pre_loop.iterate)
                        node.input_orders[1].append((level, pre_loop))
                    if node.op_type in asg.arith_op and right_levels!=0 and right_levels >= left_levels and level < right_levels:
                        if node.attr.get('inclusive') and level == node.operators[-1]:
                            rhs_subscripts.append(ir.Scalar(dtype = pre_loop.iterate.dtype, name = pre_loop.iterate.__name__+'-1'))
                        else:
                            rhs_subscripts.append(pre_loop.iterate)
                        node.input_orders[1].append((level, pre_loop))
                    if node.attr.get('scan') and level == node.operators[-1]:
                        if not node.attr.get('inclusive'):
                            res_subscripts.append(ir.Scalar(dtype = pre_loop.iterate.dtype, name = '(1)+'+pre_loop.iterate.__name__))
                        else:
                            res_subscripts.append(ir.Scalar(dtype = pre_loop.iterate.dtype, name = pre_loop.iterate.__name__))
                    else:
                        res_subscripts.append(pre_loop.iterate)
                    node.output_order.append((level, pre_loop))
                    pre_loop.attr['output_axis'] = level
                    if par_loop:
                        pre_loop.attr['parent_loop'] = par_loop
                    else:
                        par_loop = pre_loop
                    compute.append(pre_loop)
                    compute = pre_loop.body

                lhs = bind(lhs, resolve_view(node.operators[0], lhs_subscripts))
                rhs = bind(rhs, resolve_view(node.operators[1], rhs_subscripts))
                res = bind(res, res_subscripts)
                compute.append(ir.Assignment(res, ir.Expr(lhs, rhs, op)))
            else:
                node.eval = ir.Expr(node.operators[0].eval, node.operators[1].eval, op)

        elif node.op_type in asg.math_op:
            gen_ir(node.operators[0])

            if len(node._size()) > 0:
                size = helpers.get_ir_of_size(node._size())
                node.eval = ir.Ndarray(node.dtype, size)


                node.decl = [ir.Decl(node.eval)]

                res = node.eval
                val = node.operators[0].eval
                levels = len(size)
                compute = node.compute

                subscripts = []
                for level in range(levels):
                    # sl = get_slice(val)
                    # attr = {}
                    # if sl != None and type(sl.start) == ir.Literal:
                    #     if sl.start.val < 0:
                    #         ofs = -sl.start.val
                    #         attr['slice_ofs'] = ofs
                    #     else:
                    #         ofs = 0
                    # else:
                    #     ofs = 0
                    if hasattr(size[level], "val"):
                        end_val = size[level].val
                    else:
                        end_val = size[level]
                    pre_loop = ir.Loop(0, end_val, 1, [])
                    # add openmp
                    if level == 0:                  
                        pre_loop.attr['ptype'] = 'naive'
                        pre_loop.attr['plevel'] = 0
                        pre_loop.attr['nprocs'] = {0: [16]} 
                    # if ofs > 0:
                    #     pre_loop.attr['loop_ofs'] = ofs

                    subscripts.append(pre_loop.iterate)
                    node.input_orders[0].append((level, pre_loop))
                    node.output_order.append((level, pre_loop))
                    pre_loop.attr['output_axis'] = level
                    compute.append(pre_loop)
                    compute = pre_loop.body

                val = bind(val, resolve_view(node.operators[0], subscripts))
                res = bind(res, subscripts)

                compute.append(ir.Assignment(res, ir.Math(val, node.op_type)))
            else:
                node.eval = ir.Math(node.operators[0].eval, node.op_type)

        elif node.op_type == 'setval': 
            if type(node.operators[0]) == asg.Tensor:
                node.operators[0].attr['is_arg'] = False

            gen_ir(node.operators[0])
            gen_ir(node.operators[1])

            node.eval = node.operators[0].eval

            if helpers.is_scalar(node.operators[1]):
                val = node.operators[1].eval

                if len(node.ref_size) > 0:
                    size = helpers.get_ir_of_size(node.ref_size)
                    pre_loop = ir.Loop(0, size[0], 1, [])
                    node.compute = [pre_loop]
                    res = bind(node.eval, [pre_loop.iterate])
                    for i in range(1, len(size)):
                        loop = ir.Loop(0, size[i], 1, [])
                        pre_loop.body.append(loop)
                        pre_loop = loop
                        res = bind(res, [pre_loop.iterate])

                    assign = ir.Assignment(res, val)
                    pre_loop.body.append(assign)
                else:
                    node.compute = [ir.Assignment(node.eval, val)]

                l = node.compute[0]
                for i in range(len(node.eval.size)):
                    node.output_order.append((i, l))
                    l.attr['output_axis'] = i
                    l = l.body[0]
            else:
                node.operators[1].decl = [d for d in node.operators[1].decl if d.dobject != node.operators[1].eval]
                # find all defs and replace them with new node eval
                for dfs in helpers.ir_find_defs(node.operators[1].compute, node.operators[1].eval):
                    if isinstance(dfs.lhs, ir.Indexing):
                        temp = dfs.lhs
                        idx_list = []
                        while isinstance(temp, ir.Indexing):
                            idx_list.append(temp.idx)
                            temp = temp.dobject
                        idx_list.reverse()
                        res = bind(node.eval, idx_list)
                        helpers.replace_all_ref(node.operators[1].compute, dfs.lhs, res)
                    else:
                        replace_output(node.operators[1].compute, node.operators[1].eval, node.eval)
                node.operators[1].eval = node.eval
                node.output_order = node.operators[1].output_order

        elif node.op_type == 'einsum':
            gen_ir(node.operators[0])
            if node.operators[1] != None:
                gen_ir(node.operators[1])
            node.input_orders[0] = []
            node.input_orders[1] = []

            exp = node.operators[2]
            inputs, output = exp.split('->')
            input1, input2 = inputs.split(',')
            all_indices = ''.join(sorted(set(input1 + input2)))
            all_loops = []
            mapping = {}

            reduce_begins = len(output)

            for i in output:
                pos1 = input1.find(i)
                pos2 = input2.find(i)
                if (pos1 >= 0 and pos2 < 0):
                    mapping[i] = len(all_loops)
                    l = ir.Loop(0, node.operators[0].eval.ref_size(pos1), 1, [])
                    all_loops.append(l)
                    node.input_orders[0].append((len(node.input_orders[0]), l))
                elif (pos1 < 0 and pos2 >= 0):
                    mapping[i] = len(all_loops)
                    l = ir.Loop(0, node.operators[1].eval.ref_size(pos2), 1, [])
                    all_loops.append(l)
                    node.input_orders[1].append((len(node.input_orders[1]), l))

            for i in all_indices:
                if i in output:
                    continue
                pos1 = input1.find(i)
                pos2 = input2.find(i)
                if (pos1 >= 0 and pos2 < 0):
                    mapping[i] = len(all_loops)
                    l = ir.Loop(0, node.operators[0].eval.ref_size(pos1), 1, [])
                    all_loops.append(l)
                    node.input_orders[0].append((len(node.input_orders[0]), l))
                elif (pos1 < 0 and pos2 >= 0):
                    mapping[i] = len(all_loops)
                    l = ir.Loop(0, node.operators[1].eval.ref_size(pos2), 1, [])
                    all_loops.append(l)
                    node.input_orders[1].append((len(node.input_orders[1]), l))

            for i in all_indices:
                pos1 = input1.find(i)
                pos2 = input2.find(i)
                if pos1 >= 0 and pos2 >= 0:
                    mapping[i] = len(all_loops)
                    l = ir.Loop(0, node.operators[0].eval.ref_size(pos1), 1, [])
                    all_loops.append(l)
                    node.input_orders[0].append((len(node.input_orders[0]), l))
                    node.input_orders[1].append((len(node.input_orders[1]), l))

            for i in all_indices:
                pos1 = input1.find(i)
                pos2 = input2.find(i)
                if pos1 < 0 and pos2 < 0:
                    raise IndexError('index not found!')

            for i in range(reduce_begins, len(all_loops)):
                all_loops[i].attr['ptype'] = 'reduction'

            op1 = node.operators[0].eval
            op1_subscripts = []
            for i in input1:
                op1_subscripts.append(all_loops[mapping[i]].iterate)
            op1 = bind(op1, resolve_view(node.operators[0], op1_subscripts))

            if node.operators[1] != None:
                op2 = node.operators[1].eval
                op2_subscripts = []
                for i in input2:
                    op2_subscripts.append(all_loops[mapping[i]].iterate)
                op2 = bind(op2, resolve_view(node.operators[1], op2_subscripts))
            else:
                op2 = None

            size = helpers.get_ir_of_size(node._size())
            if len(size) > 0:
                node.eval = ir.Ndarray(node.dtype, size)
            else:
                node.eval = ir.Scalar(node.dtype)
            node.decl = [ir.Decl(node.eval)]
            res = node.eval
            for i in output:
                res = bind(res, [all_loops[mapping[i]].iterate])

            if op2 != None:
                expr = ir.Expr(op1, op2, '*')
            else:
                expr = op1
            if reduce_begins == len(all_loops):
                body = ir.Assignment(res, expr)
            else:
                body = ir.Assignment(res, expr, '+')
            init = ir.Assignment(res, 0)
            if reduce_begins == 0:
                node.compute.append(init)
            pre_loop = all_loops[0]
            node.compute.append(pre_loop)
            for i in range(1, len(all_loops)):
                if reduce_begins == i:
                    init.attr['parent_loop'] = pre_loop
                    pre_loop.body.append(init)
                loop = all_loops[i]
                loop.attr['parent_loop'] = pre_loop
                pre_loop.body.append(loop)
                pre_loop = loop
            body.attr['parent_loop'] = pre_loop
            pre_loop.body.append(body)

            l = node.compute[0]
            for i in range(len(node.eval.size)):
                node.output_order.append((i, l))
                l.attr['output_axis'] = i
                l = l.body[0]

        elif node.op_type == 'view':
            gen_ir(node.operators[0])
            node.eval = node.operators[0].eval
            dim_map = []
            size_map = []
            if 'size_map' in node.operators[0].attr:
                ref_size1 = node.operators[0].attr['size_map']
            else:
                ref_size1 = helpers.get_ir_of_size(node.operators[0].ref_size)
            ref_size2 = helpers.get_ir_of_size(node.ref_size)
            for i in range(len(node.operators[1])):
                s = node.operators[1][i]
                if type(s) in (list, tuple):
                    d = []
                    si = []
                    for ss in s:
                        assert type(ss) == asg.Const
                        val = ss.val
                        si.append(ref_size1[val])
                        if 'dim_map' in node.operators[0].attr:
                            val = node.operators[0].attr['dim_map'][val]
                        d.append(val)
                    dim_map.append(d)
                    size_map.append(si)
                else:
                    assert type(s) == asg.Const
                    val = s.val
                    if 'dim_map' in node.operators[0].attr:
                        val = node.operators[0].attr['dim_map'][val]
                    dim_map.append(val)
                    size_map.append(ref_size2[i])
            node.attr['dim_map'] = dim_map
            node.attr['size_map'] = size_map

        elif node.op_type == 'mask_if_else':
            gen_ir(node.operators[0])  # cond
            gen_ir(node.operators[1])  # then_val
            gen_ir(node.operators[2])  # else_val

            cond = node.operators[0]
            then_val = node.operators[1]
            else_val = node.operators[2]

            dtype = then_val.dtype
            size = helpers.get_ir_of_size(then_val._size())

            node.eval = ir.Ndarray(dtype, size)
            node.decl = [ir.Decl(node.eval)]

            subscripts = []
            compute = node.compute
            for i in range(len(size)):
                loop = ir.Loop(0, size[i], 1, [])
                compute.append(loop)
                subscripts.append(loop.iterate)
                node.output_order.append((i, loop))
                loop.attr['output_axis'] = i
                compute = loop.body

            lhs = bind(cond.eval, subscripts)
            tval = bind(then_val.eval, subscripts)
            fval = bind(else_val.eval, subscripts)
            res = bind(node.eval, subscripts)

            # compute.append(ir.Assignment(res, ir.IfExpr(lhs, tval, fval)))
           
            compute.append(ir.IfElse(lhs, 
                ir.Assignment(res, tval), 
                ir.Assignment(res, fval)
            ))

        elif node.op_type == 'if_else':
            cond, then_val, else_val = node.operators
            gen_ir(cond)
            gen_ir(then_val)
            gen_ir(else_val)

            node.eval = ir.Ndarray(then_val.dtype, then_val._size())
            node.decl = [ir.Decl(node.eval)]

            cond_expr = cond.eval
            then_code = then_val.compute
            else_code = else_val.compute
            node.compute = [ir.IfElse(cond_expr, then_code, else_code)]
            
        elif node.op_type == 'bitpack':
            input_tensor = node.operators[0]
            bitwidth = node.operators[1]

            gen_ir(input_tensor)
            gen_ir(bitwidth)

            input_eval = input_tensor.eval
            bw_eval = bitwidth.eval

            size = helpers.get_ir_of_size(input_tensor._size())
            assert len(size) == 2, "bitpack: only supports 2D input"
            rows, cols = size[0], size[1]
            flat_len = ir.Expr(rows, cols, '*')  # rows * cols

            # output: same 2D shape, dtype int (each cell stores a 32-bit word in int32 container)
            node.eval = ir.Ndarray('int', size)
            node.decl = [ir.Decl(node.eval)]

            suffix = str(node.eval.dobject_id)

            # reservoir must be uint64_t to avoid UB and truncation
            bits = ir.Scalar('uint64_t', f'bits_{suffix}')
            bit_count = ir.Scalar('int', f'bitcnt_{suffix}')
            out_idx = ir.Scalar('int', f'outidx_{suffix}')
            node.decl += [ir.Decl(bits), ir.Decl(bit_count), ir.Decl(out_idx)]

            # mask temp (uint64)
            mask_tmp = ir.Scalar('uint64_t', f'mask_{suffix}')
            node.decl.append(ir.Decl(mask_tmp))

            # init
            node.compute.append(ir.Assignment(bits, ir.Cast(ir.Literal(0, 'int'), 'uint64_t')))
            node.compute.append(ir.Assignment(bit_count, ir.Literal(0, 'int')))
            node.compute.append(ir.Assignment(out_idx, ir.Literal(0, 'int')))

            # helper: output[out_idx] as 2D (row-major)
            out_i = lambda lin: ir.Expr(lin, cols, '/')
            out_j = lambda lin: ir.Expr(lin, cols, '%')
            out_at = lambda lin: ir.Indexing(ir.Indexing(node.eval, out_i(lin)), out_j(lin))

            # constants
            mask32_u64 = ir.Cast(ir.Literal(0xFFFFFFFF, 'int'), 'uint64_t')
            one_u64 = ir.Cast(ir.Literal(1, 'int'), 'uint64_t')

            # loops
            loop_r = ir.Loop(0, rows, 1, [])
            loop_c = ir.Loop(0, cols, 1, [])
            loop_r.body.append(loop_c)

            r = loop_r.iterate
            c = loop_c.iterate

            in_val_i32 = ir.Indexing(ir.Indexing(input_eval, r), c)

            # per-row bitwidth: bval (int)
            if isinstance(bitwidth, asg.Const):
                bval = ir.Literal(int(bitwidth.val), 'int')
            else:
                bval = ir.Indexing(bw_eval, r)

            # set mask_tmp based on bval:
            # if bval == 0 -> mask_tmp = 0
            # else if bval == 32 -> mask_tmp = 0xFFFFFFFF
            # else -> (1ULL<<bval)-1
            cond_bw0 = ir.Expr(bval, ir.Literal(0, 'int'), '==')
            cond_bw32 = ir.Expr(bval, ir.Literal(32, 'int'), '==')
            mask_dyn = ir.Expr(ir.Expr(one_u64, bval, '<<'), one_u64, '-')  # (1ULL<<b)-1

            set_mask = ir.IfElse(
                cond_bw0,
                [ir.Assignment(mask_tmp, ir.Cast(ir.Literal(0, 'int'), 'uint64_t'))],
                [ir.IfElse(cond_bw32,
                           [ir.Assignment(mask_tmp, mask32_u64)],
                           [ir.Assignment(mask_tmp, mask_dyn)])]
            )
            loop_c.body.append(set_mask)

            # v64 = (uint64)in_val & mask_tmp
            v64 = ir.Expr(ir.Cast(in_val_i32, 'uint64_t'), mask_tmp, '&')

            # bits |= (v64 << bit_count)
            merged = ir.Expr(bits, ir.Expr(v64, bit_count, '<<'), '|')
            loop_c.body.append(ir.Assignment(bits, merged))

            # bit_count += bval
            loop_c.body.append(ir.Assignment(bit_count, ir.Expr(bit_count, bval, '+')))

            # while (bit_count >= 32) flush one 32-bit word
            # (we implement as if (>=32) then flush once; since bval<=32 and we flush promptly, once is enough.)
            cond_flush = ir.Expr(bit_count, ir.Literal(32, 'int'), '>=')
            flush_body = [
                # write low 32 bits
                ir.Assignment(out_at(out_idx), ir.Cast(ir.Expr(bits, mask32_u64, '&'), 'int')),
                ir.Assignment(out_idx, ir.Expr(out_idx, ir.Literal(1, 'int'), '+')),
                ir.Assignment(bit_count, ir.Expr(bit_count, ir.Literal(32, 'int'), '-')),
                ir.Assignment(bits, ir.Expr(bits, ir.Literal(32, 'int'), '>>')),
            ]
            loop_c.body.append(ir.IfElse(cond_flush, flush_body, []))

            node.compute.append(loop_r)

            # final flush if remaining bits > 0
            final_cond = ir.Expr(bit_count, ir.Literal(0, 'int'), '>')
            node.compute.append(
                ir.IfElse(
                    final_cond,
                    [
                        ir.Assignment(out_at(out_idx), ir.Cast(ir.Expr(bits, mask32_u64, '&'), 'int')),
                        ir.Assignment(out_idx, ir.Expr(out_idx, ir.Literal(1, 'int'), '+')),
                    ],
                    []
                )
            )

            # zero-fill rest
            loop_z = ir.Loop(out_idx, flat_len, 1, [])
            k = loop_z.iterate
            loop_z.body.append(ir.Assignment(out_at(k), ir.Literal(0, 'int')))
            node.compute.append(loop_z)

        elif node.op_type == 'bitunpack':
            packed = node.operators[0]      # 1D int tensor: stream of 32-bit words (stored in int32)
            bitwidth = node.operators[1]    # 1D int tensor: per-row bw (<=32)
            block_size = node.operators[2]  # scalar int

            gen_ir(packed)
            gen_ir(bitwidth)
            gen_ir(block_size)

            packed_eval = packed.eval
            bw_eval = bitwidth.eval
            bs_eval = block_size.eval  # scalar

            bw_size = helpers.get_ir_of_size(bitwidth._size())
            assert len(bw_size) == 1, "bitunpack: bitwidth must be 1D"
            rows = bw_size[0]
            cols = bs_eval

            node.eval = ir.Ndarray('int', [rows, cols])
            node.decl = [ir.Decl(node.eval)]

            suffix = str(node.eval.dobject_id)

            # reservoir must be uint64_t
            bits = ir.Scalar('uint64_t', f'ubits_{suffix}')
            bit_count = ir.Scalar('int', f'ubitcnt_{suffix}')
            in_idx = ir.Scalar('int', f'uinidx_{suffix}')
            node.decl += [ir.Decl(bits), ir.Decl(bit_count), ir.Decl(in_idx)]

            # mask temp
            mask_tmp = ir.Scalar('uint64_t', f'umask_{suffix}')
            node.decl.append(ir.Decl(mask_tmp))

            # init
            node.compute.append(ir.Assignment(bits, ir.Cast(ir.Literal(0, 'int'), 'uint64_t')))
            node.compute.append(ir.Assignment(bit_count, ir.Literal(0, 'int')))
            node.compute.append(ir.Assignment(in_idx, ir.Literal(0, 'int')))

            # constants
            mask32_u64 = ir.Cast(ir.Literal(0xFFFFFFFF, 'int'), 'uint64_t')
            one_u64 = ir.Cast(ir.Literal(1, 'int'), 'uint64_t')
            zero_u64 = ir.Cast(ir.Literal(0, 'int'), 'uint64_t')

            loop_r = ir.Loop(0, rows, 1, [])
            loop_c = ir.Loop(0, cols, 1, [])
            loop_r.body.append(loop_c)

            r = loop_r.iterate
            c = loop_c.iterate

            out_cell = ir.Indexing(ir.Indexing(node.eval, r), c)
            bval = ir.Indexing(bw_eval, r)  # int

            # bw==0 => output 0 (do not consume)
            cond_bw0 = ir.Expr(bval, ir.Literal(0, 'int'), '==')
            bw0_body = [ir.Assignment(out_cell, ir.Literal(0, 'int'))]

            # helper: read next packed word (as uint64 32-bit)
            packed_word_i32 = ir.Indexing(packed_eval, in_idx)
            word_u32 = ir.Expr(packed_word_i32, ir.Literal(0xFFFFFFFF, 'int'), '&')
            word_u64 = ir.Cast(word_u32, 'uint64_t')

            def refill_once_body():
                # bits |= (word_u64 << bit_count); bit_count += 32; in_idx++
                return [
                    ir.Assignment(bits, ir.Expr(bits, ir.Expr(word_u64, bit_count, '<<'), '|')),
                    ir.Assignment(bit_count, ir.Expr(bit_count, ir.Literal(32, 'int'), '+')),
                    ir.Assignment(in_idx, ir.Expr(in_idx, ir.Literal(1, 'int'), '+')),
                ]

            # ensure reservoir has at least bval bits (bval<=32 => at most one refill needed; keep two for safety)
            cond_need1 = ir.Expr(bit_count, bval, '<')
            refill1 = ir.IfElse(cond_need1, refill_once_body(), [])
            cond_need2 = ir.Expr(bit_count, bval, '<')
            refill2 = ir.IfElse(cond_need2, refill_once_body(), [])

            # set mask_tmp based on bval
            cond_bw32 = ir.Expr(bval, ir.Literal(32, 'int'), '==')
            mask_dyn = ir.Expr(ir.Expr(one_u64, bval, '<<'), one_u64, '-')  # (1ULL<<b)-1

            set_mask = ir.IfElse(
                cond_bw32,
                [ir.Assignment(mask_tmp, mask32_u64)],
                [ir.Assignment(mask_tmp, mask_dyn)]
            )

            # main body for bw>0:
            # refill if needed
            # val = bits & mask_tmp
            # out = (int)val
            # bits >>= bval; bit_count -= bval
            main_body = []
            main_body.append(refill1)
            main_body.append(refill2)
            main_body.append(set_mask)
            val_u64 = ir.Expr(bits, mask_tmp, '&')
            main_body.append(ir.Assignment(out_cell, ir.Cast(val_u64, 'int')))
            main_body.append(ir.Assignment(bits, ir.Expr(bits, bval, '>>')))
            main_body.append(ir.Assignment(bit_count, ir.Expr(bit_count, bval, '-')))

            loop_c.body.append(ir.IfElse(cond_bw0, bw0_body, main_body))
            node.compute.append(loop_r)

        elif node.op_type == 'count_leading_zeros':
            gen_ir(node.operators[0])
            ref_size = node.operators[0]._size()
            dtype = node.operators[0].dtype
            size = helpers.get_ir_of_size(ref_size)
            node.eval = ir.Ndarray('int', size)
            node.decl = [ir.Decl(node.eval)]

            lhs = node.operators[0].eval
            rhs = node.eval
            compute = node.compute

            subscripts = []
            par_loop = None
            for level in range(len(size)):
                loop = ir.Loop(0, size[level], 1, [])
                subscripts.append(loop.iterate)
                node.input_orders[0].append((level, loop))
                node.output_order.append((level, loop))
                loop.attr['output_axis'] = level
                if par_loop:
                    loop.attr['parent_loop'] = par_loop
                compute.append(loop)
                par_loop = loop
                compute = loop.body

            lhs_indexed = bind(lhs, subscripts)
            rhs_indexed = bind(rhs, subscripts)

            temp = ir.Scalar('int', name='clz_temp')
            count = ir.Scalar('int', name='clz_count')
            node.decl += [ir.Decl(temp), ir.Decl(count)]

            compute.append(ir.Assignment(temp, lhs_indexed))
            compute.append(ir.Assignment(count, ir.Literal(0, 'int')))

            loop_var = ir.Scalar('int', name='i')
            bit_loop = ir.Loop(0, ir.Literal(32, 'int'), 1, [])
            bit_loop.iterate = loop_var

            shift_amt = ir.Expr(ir.Literal(31, 'int'), loop_var, '-')
            mask = ir.Expr(ir.Literal(1, 'int'), shift_amt, '<<')
            bit = ir.Expr(temp, mask, '&')

            if_stmt = ir.IfElse(
                bit,
                [ir.Break()],
                [ir.Assignment(count, ir.Expr(count, ir.Literal(1, 'int'), '+'))]
            )
            bit_loop.body.append(if_stmt)
            compute.append(bit_loop)
            compute.append(ir.Assignment(rhs_indexed, count))

        elif node.op_type == 'filter_rows':
            input_tensor, mask_tensor = node.operators

            gen_ir(input_tensor)
            gen_ir(mask_tensor)

            input_eval = input_tensor.eval
            mask_eval = mask_tensor.eval
            dtype = input_tensor.dtype

            # 2D Tensor: shape [M, N]
            assert len(input_tensor._size()) == 2
            M, N = input_tensor._size()
            size = helpers.get_ir_of_size([mask_tensor.ref_size[0], input_tensor._size()[1]])

            node.eval = ir.Ndarray(dtype, size)
            node.decl = [ir.Decl(node.eval)]

            mask_loop = ir.Loop(0, helpers.get_ir_of_size(mask_tensor._size())[0], 1, [])
            j_loop = ir.Loop(0, helpers.get_ir_of_size([N])[0], 1, [])

            dst_row = ir.Scalar('int', 'dst_row')
            node.decl.append(ir.Decl(dst_row))
            node.compute = [ir.Assignment(dst_row, ir.Literal(0, 'int'))]

            cond = ir.Indexing(mask_eval, mask_loop.iterate)
            input_row = mask_loop.iterate

            i = input_row
            j = j_loop.iterate

            input_val = ir.Indexing(ir.Indexing(input_eval, i), j)
            output_val = ir.Indexing(ir.Indexing(node.eval, dst_row), j)

            # only store when mask[i] is true
            if_body = [
                j_loop,
                ir.Assignment(dst_row, ir.Expr(dst_row, ir.Literal(1, 'int'), '+'))
            ]
            j_loop.body.append(ir.Assignment(output_val, input_val))
            mask_loop.body.append(ir.IfElse(cond, if_body, []))

            node.compute.append(mask_loop)

            # set output axis info
            node.output_order = [(0, mask_loop), (1, j_loop)]


        elif node.op_type == 'index':
            gen_ir(node.operators[0])
            subscripts = []
            for op in node.operators[1:]:
                gen_ir(op)
                subscripts.append(op.eval)
            for i in range(len(subscripts), len(node.operators[0].ref_size)):
                op = asg.Const(slice(0, node.operators[0].ref_size[i], 1), 'slice')
                gen_ir(op)
                subscripts.append(op.eval)

            real_subscripts = resolve_view(node.operators[0], subscripts)
            assert len(real_subscripts) == len(node.operators[0].eval.size)
            node.eval = bind(node.operators[0].eval, real_subscripts)

            for key in node.operators[0].eval.attr:
                node.eval.attr[key] = node.operators[0].eval.attr[key]

            if 'dim_map' in node.operators[0].attr:
                dim_map = [node.operators[0].attr['dim_map'][i] for i in range(len(subscripts)) if type(subscripts[i]) == ir.Slice]
                size_map = [node.operators[0].attr['size_map'][i] for i in range(len(subscripts)) if type(subscripts[i]) == ir.Slice]
                tmp = list(range(len(real_subscripts)))
                j = 0
                for i in range(len(real_subscripts)):
                    if type(real_subscripts[i]) == ir.Slice:
                        tmp[i] = j
                        j += 1
                    else:
                        tmp[i] = None
                node.attr['dim_map'] = []
                node.attr['size_map'] = []
                for i in range(len(dim_map)):
                    d = dim_map[i]
                    if d == -1:
                        node.attr['dim_map'].append(-1)
                        node.attr['size_map'].append(size_map[i])
                    else:
                        if tmp[d] is not None:
                            node.attr['dim_map'].append(tmp[d])
                            node.attr['size_map'].append(size_map[i])


        elif node.op_type == 'apply':

            # operators: func, data (node.nparams), axis (node.nparams), out_ofs, cond, items (node.nparams), ret, counter

            # evaluate data, axis, out_ofs, cond
            for i in range(1, 3 + 2 * node.nparams):
                if node.operators[i] != None:
                    gen_ir(node.operators[i])

            primary_axis = node.operators[1 + node.nparams].eval.val
            sizes =  helpers.get_ir_of_size(node.operators[1].ref_size)

            # this is the loop that iterates over the axis of the primary (first) tensor input
            cond = node.operators[2 + 2 * node.nparams]
            if cond == None:
                outer_loop = ir.Loop(0, sizes[primary_axis], 1, [])
            else:
                outer_loop = ir.FilterLoop(0, sizes[primary_axis], 1,
                                        cond.eval, [], [])
                # gen ir for the counter
                gen_ir(node.operators[-1])


            nn = []
            for i in range(node.nparams):
                data = node.operators[1 + i]
                axis = node.operators[1 + node.nparams + i].eval.val

                # number of unbind axes in the eval of input data
                n = num_unbind(data.eval)
                nn.append(n)

                subscripts = []

                for j in range(axis):
                    op = asg.Const(slice(0, data.ref_size[j], 1), 'slice')
                    gen_ir(op)
                    subscripts.append(op.eval)

                subscripts.append(outer_loop.iterate)

                for j in range(axis+1, len(data.ref_size)):
                    op = asg.Const(slice(0, data.ref_size[j], 1), 'slice')
                    gen_ir(op)
                    subscripts.append(op.eval)

                real_subscripts = resolve_view(data, subscripts)

                item = node.operators[3 + 2 * node.nparams + i]
                item.eval = bind(data.eval, real_subscripts)

                if 'dim_map' in data.attr and 'size_map' in data.attr:
                    dim_map = [data.attr['dim_map'][ii] for ii in range(len(subscripts)) if
                               type(subscripts[ii]) == ir.Slice]
                    size_map = [data.attr['size_map'][ii] for ii in range(len(subscripts)) if
                                type(subscripts[ii]) == ir.Slice]
                    item = node.operators[3 + 2 * node.nparams + i]
                    tmp = list(range(len(real_subscripts)))
                    j = 0
                    for k in range(len(real_subscripts)):
                        if type(real_subscripts[k]) == ir.Slice:
                            tmp[k] = j
                            j += 1
                        else:
                            tmp[k] = None
                    item.attr['dim_map'] = []
                    item.attr['size_map'] = []
                    for k in range(len(dim_map)):
                        d = dim_map[k]
                        if d == -1:
                            item.attr['dim_map'].append(-1)
                            item.attr['size_map'].append(size_map[k])
                        else:
                            if tmp[d] is not None:
                                item.attr['dim_map'].append(tmp[d])
                                item.attr['size_map'].append(size_map[k])

            # since input items of func has been generated and indexed, we can generate the IR of the func
            ret = node.operators[-2]
            gen_ir(ret)

            # get the input orders
            for i in range(min(len(ret.input_orders), node.nparams)):
                n = nn[i]
                l = node.input_orders[1 + i]
                axis = node.operators[1 + node.nparams + i].eval.val
                # TODO: (Yihua) I don't remember how this is implemented, Yihua can you add some comments to explain it?
                # TODO: input orders may need update for views
                if axis >= n:
                    for j in range(axis):
                        l.append((len(l), ret.input_orders[i][j][1]))
                    l.append((len(l), outer_loop))
                    for j in range(axis, len(ret.input_orders[i])):
                        l.append((len(l), ret.input_orders[i][j][1]))
                else:
                    l.append((len(l), outer_loop))
                    for j in range(len(ret.input_orders[i])):
                        l.append((len(l), ret.input_orders[i][j][1]))

            def action(n, res):
                if isinstance(n, asg.Tensor) and not 'scope' in n.attr:
                    res.extend(n.compute)
                    if True:#helpers.depend_on_item(n, outer_loop.iterate): # TODO: (Yihua) check if n depends on items, if not we don't need to put it in the loop body
                        for nn in n.compute:
                            nn.attr['parent_loop'] = outer_loop
                        outer_loop.body.append(n.compute)
                        n.attr['scope'] = outer_loop.body

            t = helpers.ASGTraversal(action)
            ret_compute = t(ret)

            size = helpers.get_ir_of_size(node.ref_size)
            node.eval = ir.Ndarray(ret.eval.dtype, size)
            node.decl.append(ir.Decl(node.eval))

            out_ofs = node.operators[1 + 2 * node.nparams]
            res = bind(node.eval, [outer_loop.iterate]) if out_ofs == None else node.eval
            ret_eval = ret.attr['eval'] if 'eval' in ret.attr else ret.eval
            helpers.replace_all_ref(ret_compute, ret_eval, res)
            helpers.remove_decl(ret, ret_eval)

            # if there is no compute in the func, we simply assign the result to itself, so that later the lhs of the assignment will be changed to the output array
            if len(ret_compute) == 0:
                ret_compute.append(ir.Assignment(res, ret.eval))
                for nn in ret_compute:
                    nn.attr['parent_loop'] = outer_loop
                outer_loop.body.extend(ret_compute)

            node.compute = [outer_loop]

            # if there is an offset for output storage
            if out_ofs != None:
                assert type(ret_compute[-1]) in (ir.Loop, ir.Assignment)
                l = ret_compute[-1]
                while (type(l) == ir.Loop):
                    l = l.body[-1]
                # But the index to the node.eval in res is incorrect, we need to change it according to the offset
                helpers.rebind_iterate(l.lhs, ret_compute[-1].iterate,
                               ir.Expr(ir.Indexing(out_ofs.eval, outer_loop.iterate), ret_compute[-1].iterate, '+'))
            # ret.eval is removed from the decl
            node.decl = [d for d in node.decl if d.dobject != ret.eval]

            if cond != None:
                counter = node.operators[-1].eval
                counter.attr['loop'] = outer_loop
                node.compute = [ir.Assignment(counter, 0)] + node.compute
                outer_loop.body.append(ir.Assignment(counter, 1, '+'))
                assert type(ret_compute[-1]) in (ir.Loop, ir.Assignment)
                l = ret_compute[-1]
                while (type(l) == ir.Loop):
                    l = l.body[-1]
                helpers.rebind_iterate(l.lhs, outer_loop.iterate, counter)
                node.attr['eval'] = node.eval

                subscripts = [ir.Slice(ir.Literal(0, counter.dtype), counter, ir.Literal(1, counter.dtype))]
                for i in range(1, len(node.ref_size)):
                    op = asg.Const(slice(0, node.ref_size[i], 1), 'slice')
                    gen_ir(op)
                    subscripts.append(op.eval)
                node.eval = bind(node.eval, subscripts)

                node.attr['is_set'] = True
            # TODO: (Yihua) I don't remember how this is implemented. What is 'is_set' used for? Yihua can you add some comments to explain it.
            elif 'is_set' in node.operators[1].attr:
                size[primary_axis] = node.operators[1].eval.size[primary_axis]
                node.attr['is_set'] = True

            node.output_order = [(0, outer_loop)]
            outer_loop.attr['output_axis'] = 0
            if hasattr(ret, 'output_order'):
                for i in range(len(ret.output_order)):
                    node.output_order.append((i + 1, ret.output_order[i][1]))
                    ret.output_order[i][1].attr['output_axis'] = i + 1

        elif node.op_type == 'reduce':
            # TODO: add input_orders for reduce, and aggr
            gen_ir(node.operators[0])  # input data
            gen_ir(node.operators[3])  # axis
            axis = node.operators[3].eval.val

            size = helpers.get_ir_of_size(node._size())
            if len(size) > 0:
                node.eval = ir.Ndarray(node.dtype, size)
            else:
                node.eval = ir.Scalar(node.dtype)
            
            gen_ir(node.operators[2])  # init
            # the decl of node.eval should be added to the init
            node.operators[2].decl.append(ir.Decl(node.eval))

            outer_loop = ir.Loop(0, node.operators[0].eval.size[axis], 1, [], 'reduction')

            item1 = node.operators[4]
            item2 = node.operators[5]
            
            item1.eval = node.eval
            item2.eval = node.operators[0].eval
                        
            n = num_unbind(item2.eval)
            for i in range(n, axis):
                item2.eval = ir.Indexing(item2.eval, ir.Literal(-1, 'int'))
            if axis > n:
                item2.eval = ir.Indexing(item2.eval, outer_loop.iterate)
            else:
                item2.eval = bind(item2.eval, [outer_loop.iterate])
            item2.decl = []
            item1.decl = []

            ret = node.operators[-1]
            gen_ir(ret)
            
            # choose the compute body produced by ret
            if len(ret.output_order) > 0:
                compute = ret.output_order[-1][1].body
            else:
                compute = ret.compute
            # ---- FIX: if ret is scalar expr, compute will be empty; write back explicitly ----
            if (compute is None) or (len(compute) == 0):
                # ret.eval is the expression for the updated accumulator
                outer_loop.body = [ir.Assignment(node.eval, ret.eval)]
            else:
                outer_loop.body = compute[:]
                compute.clear()
                
            # compute = ret.output_order[-1][1].body if len(ret.output_order) > 0 else ret.compute
            # outer_loop.body = compute[:]
            # compute.clear()

            # merge init into node.compute
            init = node.operators[2].output_order[-1][1].body if len(node.operators[2].output_order) > 0 else \
            node.operators[2].compute
            # assert len(node.operators[2].output_order) == len(ret.output_order)
            for i in range(len(node.operators[2].output_order)):
                # assert has_same_iteration_space(node.operators[2].output_order[i][1], ret.output_order[i][1])
                helpers.rebind_iterate(init, node.operators[2].output_order[i][1].iterate, ret.output_order[i][1].iterate)
                node.output_order.append((i, ret.output_order[i][1]))
                ret.output_order[i][1].attr['output_axis'] = i
            compute.extend(init)
            node.operators[2].compute.clear()
            compute.append(outer_loop)
            
            def action(node, res):
                if isinstance(node, asg.Tensor):
                    res.extend(node.compute)
                    node.compute.clear()

            t = helpers.ASGTraversal(action)
            ret_compute = t(ret)
            node.compute.extend(ret_compute)
            
            # replace_output(node.compute, ret.eval, node.eval)
            helpers.replace_all_ref(node.compute, ret.eval, node.eval)
            node.decl = [d for d in node.decl if d.dobject != ret.eval]


        elif node.op_type == 'scan':
            gen_ir(node.operators[0])  # input data
            gen_ir(node.operators[3])  # axis
            axis = node.operators[3].eval.val
            size = helpers.get_ir_of_size(node._size())
            if len(size) > 0:
                node.eval = ir.Ndarray(node.dtype, size)
            else:
                node.eval = ir.Scalar(node.dtype)

            inclusive = node.attr.get('inclusive', False)

            if not inclusive:
                gen_ir(node.operators[2])  # init
                node.operators[2].decl.append(ir.Decl(node.eval))
            else:
                node.decl.append(ir.Decl(node.eval))
                loop_vars = [ir.Scalar('int', f'_l{i}') for i in range(len(size))]
                copy_loop = node.compute
                for d in range(len(size)):
                    loop = ir.Loop(0, size[d], 1, [])
                    loop.attr['output_axis'] = d
                    copy_loop.append(loop)
                    copy_loop = loop.body

                lhs = bind(node.eval, loop_vars)
                rhs = bind(node.operators[0].eval, loop_vars)
                copy_loop.append(ir.Assignment(lhs, rhs))

            real_sizes = helpers.get_ir_of_size(node.operators[0]._size())
            loop_extent = real_sizes[axis]
            outer_loop = ir.Loop(0, loop_extent, 1, [], 'scan')

            item1 = node.operators[4]
            item2 = node.operators[5]
            
            item1.eval = node.eval
            item2.eval = node.operators[0].eval
                        
            n = num_unbind(item2.eval)
            item2.decl = []
            item1.decl = []

            ret = node.operators[-1]
            ret.attr['scan'] = True
            ret.attr['inclusive'] = inclusive
            ret.operators.append(axis)
            gen_ir(ret)
            compute = ret.output_order[-1][1].body if len(ret.output_order) > 0 else ret.compute

            def action(node, res):
                if isinstance(node, asg.Tensor):
                    res.extend(node.compute)
                    node.compute.clear()

            t = helpers.ASGTraversal(action)
            ret_compute = t(ret)
            node.compute.extend(ret_compute)
            
            helpers.replace_all_ref(node.compute, ret.eval, node.eval)
            node.decl = [d for d in node.decl if d.dobject != ret.eval]


        elif node.op_type == 'diff1d':
            gen_ir(node.operators[0])
            input_tensor = node.operators[0]
            input_eval = input_tensor.eval
            axis = node.operators[1].val

            size = helpers.get_ir_of_size(input_tensor._size())

            node.eval = ir.Ndarray(input_tensor.dtype, size)
            node.decl = [ir.Decl(node.eval)]

            ndim = len(size)

            # helper: bind tensor with indices
            def index(tensor, indices):
                return bind(tensor, indices)

            # helper: get loop end expression safely
            def loop_end(dim):
                s = size[dim]
                return s.val if hasattr(s, 'val') else s

            # Build outer loops for dims < axis
            node.compute = []
            outer = node.compute
            prefix_iters = [None] * ndim  # store iter expr for each dim used in current scope

            for i in range(axis):
                loop = ir.Loop(0, loop_end(i), 1, [])
                loop.attr['output_axis'] = i
                outer.append(loop)
                prefix_iters[i] = loop.iterate
                outer = loop.body

            # helper: build tail loops (dims > axis) under a given body list, return tail iters and innermost body list
            def build_tail_loops(body_list):
                tail_iters = [None] * ndim
                cur = body_list
                for i in range(axis + 1, ndim):
                    lp = ir.Loop(0, loop_end(i), 1, [])
                    lp.attr['output_axis'] = i
                    cur.append(lp)
                    tail_iters[i] = lp.iterate
                    cur = lp.body
                return tail_iters, cur

            # ---- Case 1: j = 0  (copy input) ----
            j0_body = []
            tail_iters0, inner0 = build_tail_loops(j0_body)

            idx0 = []
            for d in range(ndim):
                if d < axis:
                    idx0.append(prefix_iters[d])
                elif d == axis:
                    idx0.append(ir.Literal(0, 'int'))
                else:
                    idx0.append(tail_iters0[d])

            inner0.append(ir.Assignment(index(node.eval, idx0), index(input_eval, idx0)))
            outer.extend(j0_body)

            # ---- Case 2: j = 1..end-1 (difference) ----
            diff_loop = ir.Loop(1, loop_end(axis), 1, [])
            diff_loop.attr['output_axis'] = axis

            # inside diff_loop, build tail loops and put the diff assignment in the innermost body
            tail_iters1, inner1 = build_tail_loops(diff_loop.body)

            idx_j = []
            idx_jm1 = []
            for d in range(ndim):
                if d < axis:
                    idx_j.append(prefix_iters[d])
                    idx_jm1.append(prefix_iters[d])
                elif d == axis:
                    idx_j.append(diff_loop.iterate)
                    idx_jm1.append(ir.Expr(diff_loop.iterate, ir.Literal(1, 'int'), '-'))
                else:
                    idx_j.append(tail_iters1[d])
                    idx_jm1.append(tail_iters1[d])

            lhs = index(node.eval, idx_j)
            rhs_l = index(input_eval, idx_j)
            rhs_r = index(input_eval, idx_jm1)
            inner1.append(ir.Assignment(lhs, ir.Expr(rhs_l, rhs_r, '-')))

            outer.append(diff_loop)
            
        elif node.op_type == 'concat':
            gen_ir(node.operators[0])  # tensor a
            gen_ir(node.operators[1])  # tensor b
            gen_ir(node.operators[2])  # axis

            a = node.operators[0].eval
            b = node.operators[1].eval
            axis = node.operators[2].val

            size = helpers.get_ir_of_size(node._size())
            node.eval = ir.Ndarray(node.dtype, size)
            node.decl = [ir.Decl(node.eval)]

            compute = node.compute

            loop_stack = []
            for i in range(len(size)):
                if i == axis:
                    loop = ir.Loop(0, a.size[i], 1, [])
                    loop_stack.append(loop)
                    node.input_orders[0].append((i, loop))
                else:
                    loop = ir.Loop(0, size[i], 1, [])
                    loop_stack.append(loop)
                    node.input_orders[0].append((i, loop))
                if i == 0:
                    compute.append(loop)
                else:
                    loop_stack[-2].body.append(loop)

            # copy from a
            lhs = bind(node.eval, [l.iterate for l in loop_stack])
            rhs = bind(a, [l.iterate for l in loop_stack])
            loop_stack[-1].body.append(ir.Assignment(lhs, rhs))

            # copy from b
            loop_stack_b = []
            for i in range(len(size)):
                if i == axis:
                    loop = ir.Loop(0, b.size[i], 1, [])
                    loop_stack_b.append(loop)
                    node.input_orders[1].append((i, loop))
                else:
                    loop = ir.Loop(0, size[i], 1, [])
                    loop_stack_b.append(loop)
                    node.input_orders[1].append((i, loop))
                if i == 0:
                    compute.append(loop)
                else:
                    loop_stack_b[-2].body.append(loop)

            lhs = bind(node.eval, [loop.iterate if i != axis else ir.Expr(loop.iterate, a.size[i], '+')
                                for i, loop in enumerate(loop_stack_b)])
            rhs = bind(b, [loop.iterate for loop in loop_stack_b])
            loop_stack_b[-1].body.append(ir.Assignment(lhs, rhs))

        elif node.op_type in ('split_first', 'split_second'):
            gen_ir(node.operators[0])  # source tensor
            gen_ir(node.operators[1])  # split_point

            src      = node.operators[0].eval
            split_pt = node.operators[1].eval

            size = helpers.get_ir_of_size(node._size())
            node.eval = ir.Ndarray(node.dtype, size)
            node.decl = [ir.Decl(node.eval)]

            loop = ir.Loop(0, size[0], 1, [])
            node.output_order.append((0, loop))
            loop.attr['output_axis'] = 0
            node.compute.append(loop)

            src_idx = ir.Expr(loop.iterate, split_pt, '+') if node.op_type == 'split_second' else loop.iterate
            lhs = bind(node.eval, [loop.iterate])
            rhs_subs = resolve_view(node.operators[0], [src_idx])
            rhs = bind(src, rhs_subs)
            loop.body.append(ir.Assignment(lhs, rhs))

        elif node.op_type == 'aggr':
            gen_ir(node.operators[0])  # input tensor
            gen_ir(node.operators[3])  # indices
            gen_ir(node.operators[4])  # axis
            axis = node.operators[4].eval.val
            size = helpers.get_ir_of_size(node._size())
            node.eval = ir.Ndarray(node.dtype, size)
            gen_ir(node.operators[2])  # init
            node.operators[2].decl.append(ir.Decl(node.eval))

            # compute
            outer_loop = ir.Loop(0, node.operators[0].eval.size[axis], 1, [], 'reduction')

            item1 = node.operators[6]
            item2 = node.operators[7]
            item1.eval = ir.Indexing(node.eval, ir.Indexing(node.operators[3].eval, outer_loop.iterate))
            item2.eval = node.operators[0].eval
            for i in range(axis):
                item2.eval = ir.Indexing(item2.eval, ir.Literal(-1, 'int'))
            item2.eval = ir.Indexing(item2.eval, outer_loop.iterate)
            item2.decl = []
            item1.decl = []

            ret = node.operators[-1]
            gen_ir(ret)

            def action(node, res):
                if isinstance(node, asg.Tensor):
                    res.extend(node.compute)
                    node.compute.clear()

            ret_compute = helpers.ASGTraversal(action)(ret)
            
            for nn in ret_compute:
                nn.attr['parent_loop'] = outer_loop
            outer_loop.body.extend(ret_compute)
            node.compute.append(outer_loop)

            replace_output(node.compute, ret.eval, item1.eval)
            node.decl = [d for d in node.decl if d.dobject != ret.eval]

            node.output_order = [(0, outer_loop)]
            for i in range(len(ret.output_order)):
                node.output_order.append((i + 1, ret.output_order[i][1]))
                ret.output_order[i][1].attr['output_axis'] = i + 1

        elif node.op_type == 'inline':
            src = node.operators[0]
            num_output = node.operators[1].val
            outputs_keyvalue = []
            inputs_keyvalue = []
            for i in range(2, len(node.operators), 2):
                gen_ir(node.operators[i+1])
                if i<=num_output*2:
                    gen_ir(node.operators[i+1])
                    outputs_keyvalue.append((node.operators[i], node.operators[i+1].eval))
                else:
                    gen_ir(node.operators[i+1])
                    inputs_keyvalue.append((node.operators[i], node.operators[i+1].eval))
            node.eval = node.operators[3].eval
            node.compute = [ir.Code(src, dict(outputs_keyvalue), dict(inputs_keyvalue))]

        elif node.op_type == 'size':
            gen_ir(node.operators[0])
            gen_ir(node.operators[1])

            axis = node.operators[1].eval.val
            node.eval = ir.Scalar(node.operators[0]._size()[0].dtype)
            node.decl = [ir.Decl(node.eval)]
            node.compute = [ir.Assignment(node.eval, node.operators[0].eval.size[axis])]

        elif node.op_type == 'cast':
            src = node.operators[0]
            to_dtype = node.operators[1]
            gen_ir(src)
            size = helpers.get_ir_of_size(node._size())
            if len(size) > 0:
                node.eval = ir.Ndarray(to_dtype, size)
                node.decl = [ir.Decl(node.eval)]
                subs = []
                compute = node.compute
                for i in range(len(size)):
                    loop = ir.Loop(0, size[i], 1, [])
                    compute.append(loop)
                    subs.append(loop.iterate)
                    node.output_order.append((i, loop))
                    loop.attr['output_axis'] = i
                    compute = loop.body
                lhs = bind(node.eval, subs)
                rhs = bind(src.eval, subs)
                compute.append(ir.Assignment(lhs, ir.Cast(rhs, to_dtype)))
            else:
                node.eval = ir.Scalar(to_dtype)
                node.decl = [ir.Decl(node.eval)]
                node.compute = [ir.Assignment(node.eval, ir.Cast(src.eval, to_dtype))]

        elif node.op_type == 'unfilter_rows':
            compact, mask = node.operators
            gen_ir(compact)
            gen_ir(mask)

            M = helpers.get_ir_of_size(mask._size())[0]
            N = helpers.get_ir_of_size([compact._size()[1]])[0]
            node.eval = ir.Ndarray(compact.dtype, [M, N], node.name)
            node.decl = [ir.Decl(node.eval)]

            src_row = ir.Scalar('int', 'src_row')
            node.decl.append(ir.Decl(src_row))
            node.compute = [ir.Assignment(src_row, ir.Literal(0, 'int'))]

            i_loop = ir.Loop(0, M, 1, [])
            j_loop = ir.Loop(0, N, 1, [])

            i = i_loop.iterate
            j = j_loop.iterate
            cond = ir.Indexing(mask.eval, i)

            out_ij = ir.Indexing(ir.Indexing(node.eval, i), j)
            in_ij  = ir.Indexing(ir.Indexing(compact.eval, src_row), j)

            j_loop.body.append(ir.Assignment(out_ij, in_ij))
            then_body = [j_loop, ir.Assignment(src_row, ir.Expr(src_row, ir.Literal(1, 'int'), '+'))]

            i_loop.body.append(ir.IfElse(cond, then_body, []))
            node.compute.append(i_loop)

            node.output_order = [(0, i_loop), (1, j_loop)]

        elif node.op_type == 'prefix_sum':
            src = node.operators[0]
            axis = node.operators[1]   # Const int
            inc  = node.operators[2]   # Const 0/1

            gen_ir(src)
            gen_ir(axis)
            gen_ir(inc)

            size = helpers.get_ir_of_size(src._size())
            assert len(size) == 2, "prefix_sum: currently only supports 2D tensor"

            rows, cols = size[0], size[1]

            # output same shape
            node.eval = ir.Ndarray(src.dtype, size)
            node.decl = [ir.Decl(node.eval)]
            node.compute = []

            assert isinstance(axis, asg.Const), "prefix_sum: axis must be Const for now"
            assert isinstance(inc, asg.Const), "prefix_sum: inclusive must be Const for now"
            ax = axis.val
            inclusive = (inc.val == 1)

            # helper: read src[i,j] with view support
            def load_src(subs):
                # subs: list of ir expr [i, j]
                real_subs = resolve_view(src, subs)   # may rewrite indices if src is view
                return bind(src.eval, real_subs)

            # helper: out[i,j]
            def out_at(i_expr, j_expr):
                return bind(node.eval, [i_expr, j_expr])

            one = ir.Literal(1, 'int')
            zero = ir.Literal(0, 'int')

            if ax == 1:
                # prefix along columns within each row
                loop_r = ir.Loop(0, rows, 1, [])
                loop_c = ir.Loop(0, cols, 1, [])
                loop_r.body.append(loop_c)

                r = loop_r.iterate
                c = loop_c.iterate

                out_rc = out_at(r, c)

                cond0 = ir.Expr(c, zero, '==')

                if inclusive:
                    # out[r,0] = src[r,0]
                    then0 = [ir.Assignment(out_rc, load_src([r, c]))]
                    # out[r,c] = out[r,c-1] + src[r,c]
                    prev = out_at(r, ir.Expr(c, one, '-'))
                    else0 = [ir.Assignment(out_rc, ir.Expr(prev, load_src([r, c]), '+'))]
                else:
                    # exclusive: out[r,0] = 0
                    then0 = [ir.Assignment(out_rc, zero)]
                    # out[r,c] = out[r,c-1] + src[r,c-1]
                    prev = out_at(r, ir.Expr(c, one, '-'))
                    src_prev = load_src([r, ir.Expr(c, one, '-')])
                    else0 = [ir.Assignment(out_rc, ir.Expr(prev, src_prev, '+'))]

                loop_c.body.append(ir.IfElse(cond0, then0, else0))
                node.compute.append(loop_r)

            elif ax == 0:
                # prefix along rows within each column
                loop_c = ir.Loop(0, cols, 1, [])
                loop_r = ir.Loop(0, rows, 1, [])
                loop_c.body.append(loop_r)

                c = loop_c.iterate
                r = loop_r.iterate

                out_rc = out_at(r, c)
                cond0 = ir.Expr(r, zero, '==')

                if inclusive:
                    then0 = [ir.Assignment(out_rc, load_src([r, c]))]
                    prev = out_at(ir.Expr(r, one, '-'), c)
                    else0 = [ir.Assignment(out_rc, ir.Expr(prev, load_src([r, c]), '+'))]
                else:
                    then0 = [ir.Assignment(out_rc, zero)]
                    prev = out_at(ir.Expr(r, one, '-'), c)
                    src_prev = load_src([ir.Expr(r, one, '-'), c])
                    else0 = [ir.Assignment(out_rc, ir.Expr(prev, src_prev, '+'))]

                loop_r.body.append(ir.IfElse(cond0, then0, else0))
                node.compute.append(loop_c)

            else:
                raise NotImplementedError("prefix_sum: axis must be 0 or 1 for 2D")
        
        
        # TODO: (Lihan) what does this do? what is the storage attribute?
        # storage attr stores all the other representations of current node.eval, it is used in parallelize.py
        node.eval.attr['storage'] = []

    return node
