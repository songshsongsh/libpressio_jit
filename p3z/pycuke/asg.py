from __future__ import annotations
import inspect
from . import helpers

MIN_INT = -2147483648
MAX_INT = 2147483647

arith_op = {'add': '+', 'sub': '-', 'mul': '*', 'floordiv': '//', 'truediv': '/', 'mod': '%', 'leftequal': '<=', 'rightequal': '>=',
            'equal': '==', 'notequal': '!=', 'larger': '>', 'smaller': '<', 'shift_left': '<<', 'shift_right': '>>', 'bitwise_and': '&',
            'bitwise_or': '|', 'bitwise_xor': '^'}
bitwise_op = {'shift_left': '<<', 'shift_right': '>>', 'bitwise_and': '&', 'bitwise_or': '|', 'bitwise_xor': '^'}
math_op = ['round', 'abs', 'nbits', 'ceil', 'log2']
cmp_op = ['bigger', 'smaller', 'mask_bigger', 'mask_smaller']
func_op = ['apply', 'reduce', 'aggr', 'scan', 'diff1d', 'prefix_sum']
other_op = ['setval', 'einsum', 'index', 'inline', 'size', 'norm', 'view', 'mask_if_else', 'if_else', 'concat', 'split_first', 'split_second',
            'bitpack', 'bitunpack', 'count_leading_zeros', 'filter_rows', 'cast']

binary_elw = list(arith_op.keys()) + cmp_op
unary_elw = math_op
elementwise_op = binary_elw + unary_elw

int_types = ['int', 'int32_t', 'int64_t']
float_types = ['float', 'double']


def new_op(func):
    def wrapper_func(*args, **kwargs):
        _res = func(*args, **kwargs)
        _res.attr['op_name'] = func.__name__
        return _res
    return wrapper_func

def bigger(x, y):
    return TensorOp('bigger', x, y)

def mask_bigger(x, y):
    return TensorOp('mask_bigger', x, y)

def smaller(x, y):
    return TensorOp('smaller', x, y)

def mask_smaller(x, y):
    return TensorOp('mask_smaller', x, y)

def mask_if_else(cond, then_val, else_val):
    return TensorOp('mask_if_else', cond, then_val, else_val)

def if_else(cond, then_val, else_val):
    return TensorOp('if_else', cond, then_val, else_val)

def bitpack(tensor, bitwidths):
    if isinstance(bitwidths, int):
        bitwidths = Const(bitwidths, 'int')
    return TensorOp('bitpack', tensor, bitwidths)

def count_leading_zeros(x):
    return TensorOp('count_leading_zeros', x)

def filter_rows(input_tensor, mask_tensor):
    return TensorOp('filter_rows', input_tensor, mask_tensor)

def bitunpack(packed, bitwidths, block_size):
    if isinstance(bitwidths, int):
        bitwidths = Const(bitwidths, 'int')
    return TensorOp('bitunpack', packed, bitwidths, block_size)

def unfilter_rows(compact_tensor, mask_tensor):
    return TensorOp('unfilter_rows', compact_tensor, mask_tensor)


def cast(x, to_dtype: str):
    # to_dtype: 'int', 'int32_t', 'int64_t', 'uint32_t', 'float', 'double', 'bool'
    return TensorOp('cast', x, to_dtype)

# By default, the output of func should have the same size for any input, but they can have different sizes in the first dim if out_ofss is provided
def apply(func, data: (list, tuple), axes=None, out_ofs=None, cond=None):
    assert callable(func)
    nparam = len(inspect.signature(func).parameters)
    assert len(data) == nparam
    if axes == None:
        axes = []
    while (len(axes) < nparam):
        axes.append(Const(0, 'int'))
    return TensorOp('apply', func, *data, *axes, out_ofs, cond)


def setval(val, name='', dest=None):
    if isinstance(val, Tensor):
        if dest == None:
            if len(val.ref_size) > 0:
                if name == '':
                    res = Tensor(val.ref_size, dtype=val.dtype)
                else:
                    res = Tensor(val.ref_size, name=name, dtype=val.dtype)
            else:
                if name == '':
                    res = Var(dtype=val.dtype)
                else:
                    res = Var(name=name, dtype=val.dtype)
        else:
            res = dest
    else:
        if type(val) == int:
            if dest == None:
                if name == '':
                    res = Var(dtype='int')
                else:
                    res = Var(name=name, dtype='int')
            else:
                res = dest
        elif type(val) == float:
            if dest == None:
                if name == '':
                    res = Var(dtype='float')
                else:
                    res = Var(name=name, dtype='float')
            else:
                res = dest

    res.attr['is_arg'] = False
    return TensorOp('setval', res, val)

def inline(src, output=[], inputs=[]):
    return TensorOp('inline', src, len(output), *[*output, *inputs])

def einsum(exp: str, tensor1, tensor2):
    return TensorOp('einsum', tensor1, tensor2, exp)

def concat(a, b, axis=0):
    return TensorOp('concat', a, b, Const(axis, 'int'))

def split_first(tensor, split_point):
    if isinstance(split_point, int):
        split_point = Const(split_point, 'int')
    return TensorOp('split_first', tensor, split_point)

def split_second(tensor, split_point):
    if isinstance(split_point, int):
        split_point = Const(split_point, 'int')
    return TensorOp('split_second', tensor, split_point)

'''
decl: declarations of the node (e.g., type and initial values)
compute: computations of the node (e.g., function calls)
output_order: order of nodes in the output
input_orders: order of nodes in the input
eval: execute the functions in compute list
ref_by: referenced by other nodes
id: unique id
attr: attributes of the node (e.g., is_arg, dynamic_size)
'''
class ASTNode:
    nuniq = 0

    def __init__(self):
        self.decl = []
        self.compute = []
        self.output_order = []
        self.input_orders = []
        self.eval = None
        self.ref_by = []
        self.id = ASTNode.nuniq
        ASTNode.nuniq += 1
        self.attr = {}


class Tensor(ASTNode):
    def __init__(self, size: list | tuple, dtype='float', name=None):
        super().__init__()
        self.ref_size = []
        for s in size:
            if helpers.is_int_var(s):
                self.ref_size.append(s)
            elif type(s) == int:
                self.ref_size.append(Const(s, 'int'))
            else:
                raise TypeError('tensor dimensions must be int or a scalar int variable')
        self.dtype = dtype
        self.name = name
        self.attr['is_arg'] = True

    def named(self, name: str):
        self.name = name
        return self

    def __sub__(self, other):
        return TensorOp('sub', self, other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __add__(self, other):
        return TensorOp('add', self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        return TensorOp('mul', self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return TensorOp('truediv', self, other)

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def __floordiv__(self, other):
        return TensorOp('floordiv', self, other)

    def __rfloordiv__(self, other):
        return self.__floordiv__(other)
    
    def __lshift__(self, other):
        return TensorOp('shift_left', self, other)

    def __rlshift__(self, other):
        return self.__lshift__(other)

    def __rshift__(self, other):
        return TensorOp('shift_right', self, other)

    def __rrshift__(self, other):
        return self.__rshift__(other)

    def __and__(self, other):
        return TensorOp('bitwise_and', self, other)

    def __rand__(self, other):
        return self.__and__(other)

    def __or__(self, other):
        return TensorOp('bitwise_or', self, other)

    def __ror__(self, other):
        return self.__or__(other)
    
    def __xor__(self, other):
        return TensorOp('bitwise_xor', self, other)

    def __rxor__(self, other):
        return self.__xor__(other)
    
    def __matmul__(self, other):
        return TensorOp('einsum', self, other, 'ij,jk->ik')

    def __getitem__(self, idx):
        assert isinstance(idx, (int, slice, Tensor, tuple))
        return TensorOp('index', self, idx)


    def apply(self, func, axis=0, out_ofs=None, cond=None):
        assert callable(func)
        return TensorOp('apply', func, self, axis, out_ofs, cond)

    def reduce(self, func, init, axis=0):
        if callable(func) and callable(init):
            return TensorOp('reduce', self, func, init, axis)
        else:
            raise TypeError('reduce must use a callable function')

    def scan(self, func, init, axis=0, inclusive=False):
        if callable(func) and callable(init):
            op = TensorOp('scan', self, func, init, axis)
            op.attr['inclusive'] = inclusive
            return op
        else:
            raise TypeError('scan must use a callable function')
        
    def diff1d(x, axis=1):
        return TensorOp('diff1d', x, Const(axis, 'int'))

    @new_op
    def sum(self, axis=0):
        s1 = ''
        rs = ''
        for i in range(len(self._size())):
            s1 += chr(ord('i') + i)
            if i != axis:
                rs += chr(ord('i') + i)
        return einsum(f'{s1},->{rs}', self, None)

    def max(self, axis=0):
        func = lambda x, y: bigger(x, y)
        if self.dtype == 'uint32_t':
           init = lambda x: setval(0, dest=x) 
        else:
            init = lambda x: setval(MIN_INT, dest=x)
        return self.reduce(func, init, axis)

    def min(self, axis=0):
        func = lambda x, y: smaller(x, y)
        init = lambda x: setval(MAX_INT, dest=x)
        return self.reduce(func, init, axis)

    def aggr(self, func, init, indices, axis=0, size=None):
        if callable(func) and callable(init):
            op = TensorOp('aggr', self, func, init, indices, axis, size)
            return op
        else:
            raise TypeError('aggr must use a callable function')

    def aggr_sum(self, indices, axis=0, size=None):
        func = lambda x, y: x + y
        init = lambda: setval(0)
        return self.aggr(func, init, indices, axis, size)

    def aggr_max(self, indices, axis=0, size=None):
        func = lambda x, y: bigger(x, y)
        init = lambda: setval(MIN_INT)
        return self.aggr(func, init, indices, axis, size)

    def aggr_min(self, indices, axis=0, size=None):
        func = lambda x, y: smaller(x, y)
        init = lambda: setval(MAX_INT)
        return self.aggr(func, init, indices, axis, size)

    def prefix_sum(self, axis=0, inclusive=True):
        if type(axis) == Const:
            axis = axis.val
        if type(axis) == int:
            axis = Const(axis, 'int')
        inc = Const(1 if inclusive else 0, 'int')
        return TensorOp('prefix_sum', self, axis, inc)    

    # def prefix_sum2(self, axis=0, inclusive=True):
    #     assert type(axis) == int or (type(axis) == Const and axis.dtype in int_types), 'axis must be an integer or a Const'
    #     if type(axis) == Const:
    #         axis = axis.val

    #     assert len(self.ref_size) > 0, 'input must have at least one dimension'
    #     data = self
    #     size = []
    #     if not inclusive:
    #         size[0] = self.ref_size[axis] + 1

    #     for i in range(len(self.ref_size)):
    #         if axis != i:
    #             size.append(self.ref_size[i])

    #     res = Tensor(size, dtype=self.dtype)

    #     for i in range(axis):
    #         data = data[:]
    #         res = res[:]

    #     if inclusive:
    #         return setval(data[:] + res[-1:size[axis] - 1], dest=res)
    #     else:
    #         return setval(data[-1:data._size()[axis]] + res[-1:size[axis] - 1], dest=res)

    def _size(self):
        return self.ref_size

    def size(self, axis):
        return TensorOp('size', self, axis)

    def round(self):
        return TensorOp('round', self)

    def abs(self):
        return TensorOp('abs', self)
    
    def ceil(self):
        return TensorOp('ceil', self)
    
    def log2(self):
        return TensorOp('log2', self)

    def nbits(self):
        return TensorOp('nbits', self)

    def view(self, sizes, dims):
        return TensorOp('view', self, sizes, dims)


class Var(Tensor):
    def __init__(self, dtype='int', name=None):
        super().__init__([], dtype, name)


# const is var without name
class Const(Var):
    def __init__(self, val, dtype):
        super().__init__(dtype)
        # slice is considered constant because once the slice is created its start, stop, step cannot be reassigned
        # however, start, stop, step themselves can be variables
        if dtype == 'slice':
            if type(val.start) == int:
                start = Const(val.start, 'int')
            else:
                start = val.start
            if type(val.stop) == int:
                stop = Const(val.stop, 'int')
            else:
                stop = val.stop
            if type(val.step) == int:
                step = Const(val.step, 'int')
            else:
                step = val.step
            assert helpers.is_int_var(start)
            assert helpers.is_int_var(stop)
            assert helpers.is_int_var(step)
            self.val = slice(start, stop, step)
        else:
            self.val = val



class TensorOp(Tensor):
    Types = func_op + list(arith_op.keys()) + math_op + cmp_op + other_op
    def __init__(self, op_type, *operators):
        assert op_type in TensorOp.Types
        self.op_type = op_type

        # TODO: infer result data type
        self.operators = []
        for opr in operators:
            self.operators.append(opr)
            if isinstance(opr, ASTNode) and (not op_type in ('setval', 'inline')):
                opr.ref_by.append(self)

        if op_type in arith_op or op_type in cmp_op:
            if op_type in ('mask_bigger, mask_smaller'):
                dtype = 'bool'
            elif op_type in bitwise_op:
                dtype = 'uint32_t'
            else:
                dtype = operators[0].dtype
            if type(self.operators[0]) == int:
                # assert(dtype in int_types)
                self.operators[0] = Const(self.operators[0], dtype)
            elif type(operators[0]) == float:
                # assert(dtype in float_types)
                self.operators[0] = Const(self.operators[0], dtype)
            if type(self.operators[1]) == int:
                # assert(dtype in int_types)
                self.operators[1] = Const(self.operators[1], dtype)
            elif type(operators[1]) == float:
                # assert(dtype in float_types)
                self.operators[1] = Const(self.operators[1], dtype)
            # assert helpers.broadcastable(self.operators[0]._size(), self.operators[1]._size())
            if (len(self.operators[0]._size()) > len(self.operators[1]._size())):
                ref_size = self.operators[0]._size()
            else:
                ref_size = self.operators[1]._size()

        elif op_type == 'einsum':
            dtype = operators[0].dtype
            exp = self.operators[2]
            inputs, output = exp.split('->')
            input1, input2 = inputs.split(',')
            op1_size = self.operators[0]._size()
            if self.operators[1] != None:
                op2_size = self.operators[1]._size()
            else:
                op2_size = []
            ref_size = []
            for i in output:
                pos1 = input1.find(i)
                if pos1 >= 0:
                    ref_size.append(op1_size[pos1])
                else:
                    pos2 = input2.find(i)
                    if pos2 >= 0:
                        ref_size.append(op2_size[pos2])
                    else:
                        raise IndexError('index not found!')

        elif op_type == 'view':
            dtype = operators[0].dtype
            sizes = []
            for s in operators[1]:
                if (type(s) == Const and s.dtype in int_types) or helpers.is_int_var(s):
                    sizes.append(s)
                elif type(s) == int:
                    sizes.append(Const(s, 'int'))
                else:
                    raise TypeError('tensor dimensions must be int or a scalar int variable')
            dims = []
            # view dims must be int literal or const
            for s in operators[2]:
                if type(s) == Const and s.dtype in int_types:
                    dims.append(s)
                elif type(s) == int:
                    dims.append(Const(s, 'int'))
                elif type(s) in (list, tuple):
                    d = []
                    for ss in s:
                        if type(ss) == Const and ss.dtype in int_types:
                            d.append(ss)
                        elif type(ss) == int:
                            d.append(Const(ss, 'int'))
                    dims.append(d)
                else:
                    raise TypeError('tensor dimensions must be int or a scalar int variable')
            # TODO: check if sizes is consistent with ref_size of the original tensor
            ref_size = sizes
            self.operators.pop()
            self.operators[1] = dims

        elif op_type == 'mask_if_else':
            cond, then_val, else_val = self.operators
            assert cond._size() == then_val._size() == else_val._size()
            dtype = then_val.dtype
            ref_size = then_val._size()
            
        elif op_type == 'if_else':
            dtype = operators[1].dtype
            ref_size = operators[1]._size()
            
        elif op_type == 'count_leading_zeros':
            dtype = 'int'
            ref_size = self.operators[0]._size() 

        elif op_type == 'filter_rows':
            input_tensor, mask_tensor = self.operators
            assert len(input_tensor._size()) == 2
            assert len(mask_tensor._size()) == 1
            # assert helpers.has_same_value(input_tensor._size()[0], mask_tensor._size()[0])
            dtype = input_tensor.dtype
            ref_size = [input_tensor._size()[0], input_tensor._size()[1]]

        elif op_type == 'cast':
            # operators: (src_tensor, to_dtype:str)
            src = self.operators[0]
            to_dtype = self.operators[1]
            dtype = to_dtype
            ref_size = src._size()

        elif op_type == 'concat':
            dtype = self.operators[0].dtype
            assert self.operators[0].dtype == self.operators[1].dtype
            axis = self.operators[2].val

            shape_a = self.operators[0]._size()
            shape_b = self.operators[1]._size()
            assert len(shape_a) == len(shape_b)
            for i in range(len(shape_a)):
                if i != axis:
                    assert helpers.has_same_value(shape_a[i], shape_b[i])
            ref_size = []
            for i in range(len(shape_a)):
                if i == axis:
                    ref_size.append(shape_a[i] + shape_b[i])
                else:
                    ref_size.append(shape_a[i])

        elif op_type == 'split_first':
            assert len(self.operators[0]._size()) == 1, 'split only supports 1D tensors'
            dtype    = self.operators[0].dtype
            ref_size = [self.operators[1]]  

        elif op_type == 'split_second':
            assert len(self.operators[0]._size()) == 1, 'split only supports 1D tensors'
            dtype    = self.operators[0].dtype
            ref_size = [self.operators[0]._size()[0] - self.operators[1]]
            
        elif op_type == 'bitpack':
            input_tensor = self.operators[0]
            bitwidth = self.operators[1]
            assert len(input_tensor._size()) == 2, "bitpack: input must be a 2D tensor"
            assert input_tensor.dtype in int_types, "bitpack: input dtype must be int/int32_t/int64_t"
            if isinstance(bitwidth, int):
                self.operators[1] = Const(bitwidth, 'int')
                bitwidth = self.operators[1]
            if isinstance(bitwidth, Const):
                assert bitwidth.dtype in int_types, "bitpack: bitwidth Const must be int"
            else:
                assert helpers.is_1dint_tensor(bitwidth), "bitpack: bitwidth must be a 1D int tensor (per-row)"
                assert helpers.has_same_value(input_tensor._size()[0], bitwidth._size()[0]), \
                    "bitpack: bitwidth length must equal input rows"
            dtype = 'int'
            ref_size = input_tensor._size()

        elif op_type == 'bitunpack':
            packed = self.operators[0]      # 1D packed stream (int/int64)
            bitwidth = self.operators[1]    # 1D per-row bitwidths
            block_size = self.operators[2]  # scalar int Var/Const
            assert len(packed._size()) == 1, "bitunpack: packed must be a 1D tensor"
            assert packed.dtype in int_types, "bitunpack: packed dtype must be int/int32_t/int64_t"
            # bitwidth must be 1D int tensor, length = rows
            assert helpers.is_1dint_tensor(bitwidth), "bitunpack: bitwidth must be a 1D int tensor (per-row)"
            # block_size must be scalar int var or int const
            if isinstance(block_size, int):
                self.operators[2] = Const(block_size, 'int')
                block_size = self.operators[2]
            if isinstance(block_size, Const):
                assert block_size.dtype in int_types, "bitunpack: block_size Const must be int"
            else:
                # scalar int Var: ref_size should be []
                assert getattr(block_size, 'dtype', None) in int_types and len(getattr(block_size, 'ref_size', [])) == 0, \
                    "bitunpack: block_size must be a scalar int variable"
            # output is 2D (rows, block_size)
            dtype = 'int'
            ref_size = (bitwidth._size()[0], block_size)

        elif op_type == 'index':
            dtype = operators[0].dtype
            
            if not type(operators[1]) in (list, tuple):
                self.operators[1] = [operators[1]]
            ref_size = self.operators[0]._size()[len(self.operators[1]):]

            new_size = []
            new_idx = []
            size_dtype = operators[0]._size()[0].dtype
            for i in range(len(self.operators[1])):
                idx = self.operators[1][i]
                if type(idx) == slice:
                    start = idx.start
                    if start == None:
                        start = Const(0, size_dtype)
                    elif type(start) == int:
                        start = Const(start, size_dtype)
                    stop = idx.stop
                    if stop == None:
                        stop = self.operators[0].ref_size[i]
                    elif type(stop) == int:
                        stop = Const(stop, size_dtype)
                    step = idx.step
                    if step == None:
                        step = Const(1, size_dtype)
                    elif type(step) == int:
                        step = Const(step, size_dtype)

                    idx = Const(slice(start, stop, step), 'slice')

                    if step.val == 1:
                        if type(start) == Const and start.val==0:
                            csize = [helpers.eval_const_expr(stop)]
                        else:
                            csize = [helpers.eval_const_expr(stop - start)]
                    else:
                        csize = [helpers.eval_const_expr(stop - start) // step]
                elif helpers.is_1dint_tensor(idx):
                    csize = [idx.ref_size[0]]
                elif helpers.is_int_var(idx) or type(idx) == int:
                    csize = []
                    if type(idx) == int:
                        idx = Const(idx, size_dtype)
                else:
                    raise TypeError('index data type error!')

                new_size.extend(csize)
                new_idx.append(idx)

            ref_size[0:0] = new_size
            self.operators.pop()
            self.operators.extend(new_idx)

        elif op_type == 'apply':
            func = self.operators[0]
            self.nparams = len(inspect.signature(func).parameters)

            # check the axes of the n parameters, an axis must be an int or a int Const
            for i in range(self.nparams):
                axis = operators[1 + self.nparams + i]
                if type(axis) == int:
                    self.operators[1 + self.nparams + i] = Const(axis, 'int')
                else:
                    assert type(axis) == Const and axis.dtype in int_types, 'invalid axis'

            data = []
            # this is the primary axis size, i.e., number of loop iterations
            axis_size = self.operators[1].ref_size[self.operators[1 + self.nparams].val]
            for i in range(1, 1 + self.nparams):
                data_size = self.operators[i].ref_size
                axis = self.operators[self.nparams + i].val
                # every input item should have the same size as the primary axis size
                assert helpers.has_same_value(axis_size, data_size[axis])
                item_size = data_size[:axis] + data_size[axis + 1:]
                if (len(item_size) > 0):
                    item = Tensor(item_size, self.operators[i].dtype)
                else:
                    item = Var(self.operators[i].dtype)
                item.attr['is_arg'] = False
                data.append(item)

            # call function on an item of the input data
            ret = self.operators[0](*data)
            dtype = ret.dtype
            ret.ref_by.append(self)

            # handle output with offsets
            out_ofs = self.operators[1 + 2 * self.nparams]
            if out_ofs == None:
                ref_size = [axis_size] + ret.ref_size
            else:
                ref_size = [out_ofs[axis_size]] + ret.ref_size[1:]

            self.operators.extend(data)
            self.operators.append(ret)

            # handle conditional apply
            cond = self.operators[2 + 2 * self.nparams]
            if cond != None:
                assert helpers.is_1d_tensor(cond)
                assert helpers.has_same_value(axis_size, cond.ref_size[0])
                self.counter = setval(0)
                self.operators.append(self.counter)
                # set dynamic_size attribute for memory allocation optimization
                ref_size[0].attr['dynamic_size'] = True
            else:
                self.operators.append(None)
 
        elif op_type == 'reduce':
            assert type(self.operators[3]) == int
            axis = self.operators[3]
            self.operators[3] = Const(axis, 'int')
            ref_size = self.operators[0]._size()[:axis] + self.operators[0]._size()[axis + 1:]
            dtype = self.operators[0].dtype
            if (len(ref_size) > 0):
                item1 = Tensor(ref_size, self.operators[0].dtype)
                item2 = Tensor(ref_size, self.operators[0].dtype)
            else:
                item1 = Var(self.operators[0].dtype)
                item2 = Var(self.operators[0].dtype)
            item1.attr['is_arg'] = False
            item2.attr['is_arg'] = False
            self.operators.append(item1)
            self.operators.append(item2)
            self.operators.append(self.operators[1](item1, item2))
            
        elif op_type == 'scan':
            assert type(self.operators[3]) == int
            axis = self.operators[3]
            self.operators[3] = Const(axis, 'int')
            ref_size = self.operators[0]._size()
            # ref_size = self.operators[0]._size()[:axis] + self.operators[0]._size()[axis + 1:]
            dtype = self.operators[0].dtype
            if (len(ref_size) > 0):
                item1 = Tensor(ref_size, self.operators[0].dtype)
                item2 = Tensor(ref_size, self.operators[0].dtype)
            else:
                item1 = Var(self.operators[0].dtype)
                item2 = Var(self.operators[0].dtype)
            item1.attr['is_arg'] = False
            item2.attr['is_arg'] = False
            self.operators.append(item1)
            self.operators.append(item2)
            self.operators.append(self.operators[1](item1, item2))
            
        elif op_type == 'diff1d':
            dtype = self.operators[0].dtype
            ref_size = self.operators[0]._size()
            axis = self.operators[1]
            if isinstance(axis, int):
                axis = Const(axis, 'int')
                self.operators[1] = axis
                
        elif op_type == 'aggr':
            dtype = operators[0].dtype
            assert helpers.is_1dint_tensor(self.operators[3])
            assert type(self.operators[4]) == int
            axis = self.operators[4]
            self.operators[4] = Const(axis, 'int')
            if self.operators[5] == None:
                self.operators[5] = self.operators[3].ref_size[0]
            else:
                assert helpers.is_int_var(self.operators[5])
                if type(self.operators[5]) == int:
                    self.operators[5] = Const(self.operators[5], 'int')
            ref_size = [self.operators[5]] + self.operators[0]._size()[:axis] + self.operators[0]._size()[axis + 1:]
            if (len(ref_size) > 1):
                item1 = Tensor(ref_size[1:], self.operators[0].dtype)
                item2 = Tensor(ref_size[1:], self.operators[0].dtype)
            else:
                item1 = Var(self.operators[0].dtype)
                item2 = Var(self.operators[0].dtype)
            item1.attr['is_arg'] = False
            item2.attr['is_arg'] = False
            self.operators.append(item1)
            self.operators.append(item2)
            self.operators.append(self.operators[1](item1, item2))

        elif op_type in math_op:
            dtype = self.operators[0].dtype
            ref_size = self.operators[0]._size()
            if op_type == 'round':
                dtype = 'int'
            elif op_type == 'abs':
                dtype = self.operators[0].dtype
            elif op_type == 'ceil':
                dtype = 'int'
            elif op_type == 'log2':
                dtype = 'float'

        elif op_type == 'setval':   
            dtype = self.operators[0].dtype
            ref_size = self.operators[0]._size()
            if (helpers.is_scalar(self.operators[1])):
                if type(self.operators[1]) == int:
                    # assert(dtype in int_types)
                    self.operators[1] = Const(self.operators[1], dtype)
                elif type(self.operators[1]) == float:
                    assert(dtype in float_types)
                    self.operators[1] = Const(self.operators[1], dtype)

            else:
                assert self.operators[0].dtype == self.operators[1].dtype

        elif op_type == 'inline':
            src = self.operators[0]
            if type(self.operators[1]) == int:
                self.operators[1] = Const(self.operators[1], 'int')
            num_output = self.operators[1]
            
            dtype = self.operators[2][1].dtype
            ref_size = self.operators[2][1]._size()

            keyvalue = []
            for op in self.operators[2:]:
                keyvalue.append(op[0])
                keyvalue.append(op[1])
            self.operators.clear()
            self.operators.append(src)
            self.operators.append(num_output)
            self.operators.extend(keyvalue)

        elif op_type == 'size':
            dtype = self.operators[0]._size()[0].dtype
            ref_size = []
            if type(self.operators[1]) == int:
                self.operators[1] = Const(self.operators[1], 'int')
            axis = self.operators[1].val
            assert axis < len(self.operators[0]._size())

        elif op_type == 'unfilter_rows':
            compact_tensor, mask_tensor = self.operators
            assert len(compact_tensor._size()) == 2
            assert len(mask_tensor._size()) == 1
            dtype = compact_tensor.dtype
            ref_size = [mask_tensor._size()[0], compact_tensor._size()[1]]
        
        elif op_type == 'prefix_sum':
            x = self.operators[0]
            axis = self.operators[1] if len(self.operators) > 1 else Const(0, 'int')
            inc  = self.operators[2] if len(self.operators) > 2 else Const(1, 'int')

            if isinstance(axis, int):
                self.operators[1] = Const(axis, 'int')
                axis = self.operators[1]
            if isinstance(inc, bool):
                self.operators[2] = Const(1 if inc else 0, 'int')
                inc = self.operators[2]
            if isinstance(inc, int):
                self.operators[2] = Const(inc, 'int')
                inc = self.operators[2]

            assert isinstance(axis, Const) and axis.dtype in int_types
            assert isinstance(inc, Const) and inc.dtype in int_types and inc.val in (0, 1)
            assert len(x._size()) >= 1
            assert 0 <= axis.val < len(x._size())

            dtype = x.dtype
            ref_size = x._size()
        
        super().__init__(ref_size, dtype, name = f'{op_type}_' + '_'.join([op.name if (hasattr(op, 'name') and op.name != None) else '' for op in self.operators]))
        
        # call the init function for reduce and aggr
        if self.op_type in ('reduce', 'aggr', 'scan'):
            self.operators[2] = self.operators[2](self)

        self.input_orders = [[] for o in self.operators]





