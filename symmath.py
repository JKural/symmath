import weakref
import numbers
from fractions import Fraction

import numpy as np


class Symbol:

    symbols = weakref.WeakValueDictionary()

    def __init__(self, name, pretty_name):
        self.name = name
        self.pretty_name = pretty_name
        self.size = 1

    @classmethod
    def create(cls, *args):
        return cls.symbols.setdefault((cls, *args), cls(*args))
    
    @property
    def arity(self):
        return 0

    @staticmethod
    def _apply_op(op, instance, argument, reversed):
        if isinstance(argument, Symbol) or isinstance(argument, numbers.Real):
            if isinstance(argument, numbers.Real):
                argument = const(argument)
            lhs, rhs = instance, argument
            if reversed:
                lhs, rhs = rhs, lhs
            return op.create(lhs, rhs)
        else:
            return NotImplemented

    def __pos__(self):
        return Pos.create(self)

    def __neg__(self):
        return Neg.create(self)

    def __add__(self, other):
        return Symbol._apply_op(Add, self, other, reversed=False)

    def __radd__(self, other):
        return Symbol._apply_op(Add, self, other, reversed=True)

    def __sub__(self, other):
        return Symbol._apply_op(Sub, self, other, reversed=False)

    def __rsub__(self, other):
        return Symbol._apply_op(Sub, self, other, reversed=True)

    def __mul__(self, other):
        return Symbol._apply_op(Mul, self, other, reversed=False)

    def __rmul__(self, other):
        return Symbol._apply_op(Mul, self, other, reversed=True)

    def __truediv__(self, other):
        return Symbol._apply_op(Div, self, other, reversed=False)

    def __rtruediv__(self, other):
        return Symbol._apply_op(Div, self, other, reversed=True)

    def __pow__(self, other):
        return Symbol._apply_op(Pow, self, other, reversed=False)

    def __rpow__(self, other):
        return Symbol._apply_op(Pow, self, other, reversed=True)
    
    def __repr__(self):
        return print_symbol(self)
    
    def __str__(self):
        return pretty_print_symbol(self)


def _apply_fun(fun, x):
    if isinstance(x, Symbol):
        return fun.create(x)
    elif isinstance(x, numbers.Real):
        return fun.create(const(x))
    else:
        raise TypeError(f"Expected number or Symbol, got {type(x)} instead")


def exp(x):
    return _apply_fun(Exp, x)


def log(x):
    return _apply_fun(Log, x)


def sin(x):
    return _apply_fun(Sin, x)


def cos(x):
    return _apply_fun(Cos, x)


def tan(x):
    return _apply_fun(Tan, x)


def asin(x):
    return _apply_fun(Asin, x)


def acos(x):
    return _apply_fun(Acos, x)


def atan(x):
    return _apply_fun(Atan, x)


class Function(Symbol):

    def __init__(self, name, pretty_name, fun, args):
        super().__init__(name=name, pretty_name=pretty_name)
        self.fun = fun
        self.args = args
        self.size = 1 + sum(arg.size for arg in self.args)
        self.vars = set()
        for arg in self.args:
            try:
                self.vars |= arg.vars
            except AttributeError:
                if isinstance(arg, Variable):
                    self.vars.add(arg)
        self.vars = frozenset(self.vars)
    
    @property
    def arity(self):
        return len(self.args)

    def print_scheme(self, *tokens):
        return f"{self.name}({', '.join(tokens)})"

    def pretty_print_scheme(self, *token_precedence_pairs):
        return f"{self.pretty_name}({', '.join([token for token, _ in token_precedence_pairs])})"
    
    def __call__(self, *args):
        try:
            var_dict = {var.id: args[var.id] for var in self.vars}
        except KeyError:
            raise TypeError("Not enough arguments to fill all variables")
        eval = Evaluate(var_dict)
        return eval(self)


class Operator(Function):

    def __init__(self, name, pretty_name, fun, precedence, associativity, args):
        super().__init__(name=name, pretty_name=pretty_name, fun=fun, args=args)
        self.precedence = precedence
        self.associativity = associativity


class UnaryOperator(Operator):
    def __init__(self, name, pretty_name, fun, precedence, associativity, arg):
        super().__init__(
            name=name,
            pretty_name=pretty_name,
            fun=fun,
            precedence=precedence,
            associativity=associativity,
            args=(arg,),
        )
    
    def pretty_print_scheme(self, token_precedence_pair):
        token, precedence = token_precedence_pair
        if precedence > self.precedence:
            token = f"({token})"
        
        if self.associativity == "left":
            return f"{token}{self.pretty_name}"
        elif self.associativity == "right":
            return f"{self.pretty_name}{token}"
        else:
            assert False


class BinaryOperator(Operator):
    def __init__(self, name, pretty_name, fun, precedence, associativity, lhs, rhs):
        super().__init__(
            name=name,
            pretty_name=pretty_name,
            fun=fun,
            precedence=precedence,
            associativity=associativity,
            args=(lhs, rhs),
        )
    
    def pretty_print_scheme(self, lhs_token_precedence_pair, rhs_token_precedence_pair):
        lhs_token, lhs_precedence = lhs_token_precedence_pair
        rhs_token, rhs_precedence = rhs_token_precedence_pair
        if self.associativity == "left":
            if lhs_precedence > self.precedence:
                lhs_token = f"({lhs_token})"
            if rhs_precedence >= self.precedence:
                rhs_token = f"({rhs_token})"
        elif self.associativity == "right":
            if lhs_precedence >= self.precedence:
                lhs_token = f"({lhs_token})"
            if rhs_precedence > self.precedence:
                rhs_token = f"({rhs_token})"
        return f"{lhs_token}{self.pretty_name}{rhs_token}"


class Constant(Symbol):

    def __init__(self, name, pretty_name, dtype, value):
        if type(value) is not dtype:
            raise TypeError(
                f"'value' is expected to be of type {dtype}, got {type(value)} instead"
            )
        super().__init__(name=name, pretty_name=pretty_name)
        self.dtype = dtype
        self.value = value
    
    def print_scheme(self):
        return f"{self.name}: {self.value}"
    
    def pretty_print_scheme(self):
        return f"{self.value}"


class Variable(Symbol):

    def __init__(self, id):
        if type(id) is not int:
            raise TypeError(
                f"'id' is expected to be of type {int}, got {type(id)} instead"
            )
        super().__init__(name="Variable", pretty_name="var")
        self.id = id
    
    def print_scheme(self, *_):
        return f"{self.name}[{self.id}]"
    
    def pretty_print_scheme(self, *_):
        return f"x_{self.id}"


class Integer(Constant):

    def __init__(self, value):
        super().__init__(name="Integer", pretty_name="Z", dtype=int, value=value)


class Rational(Constant):

    def __init__(self, value):
        super().__init__(name="Rational", pretty_name="Q", dtype=Fraction, value=value)


class Real(Constant):

    def __init__(self, value):
        super().__init__(name="Real", pretty_name="R", dtype=float, value=value)


class Pos(UnaryOperator):

    def __init__(self, arg):
        super().__init__(
            name="Pos", pretty_name="+", fun=np.positive, precedence=3, associativity="right", arg=arg
        )
    
    def derivative_scheme(self, arg):
        return +arg


class Neg(UnaryOperator):

    def __init__(self, arg):
        super().__init__(
            name="Neg", pretty_name="-", fun=np.negative, precedence=3, associativity="right", arg=arg
        )
    
    def derivative_scheme(self, arg):
        return -arg
    

class Add(BinaryOperator):

    def __init__(self, lhs, rhs):
        super().__init__(
            name="Add",
            pretty_name="+",
            fun=np.add,
            precedence=5,
            associativity="left",
            lhs=lhs,
            rhs=rhs,
        )
    
    def derivative_scheme(self, arg0, arg1):
        return arg0 + arg1


class Sub(BinaryOperator):
    def __init__(self, lhs, rhs):
        super().__init__(
            name="Sub",
            pretty_name="-",
            fun=np.subtract,
            precedence=5,
            associativity="left",
            lhs=lhs,
            rhs=rhs,
        )

    def derivative_scheme(self, arg0, arg1):
        return arg0 - arg1


class Mul(BinaryOperator):

    def __init__(self, lhs, rhs):
        super().__init__(
            name="Mul",
            pretty_name="*",
            fun=np.multiply,
            precedence=4,
            associativity="left",
            lhs=lhs,
            rhs=rhs,
        )

    def derivative_scheme(self, arg0, arg1):
        return arg0 * self.args[1] + self.args[0] * arg1


class Div(BinaryOperator):

    def __init__(self, lhs, rhs):
        super().__init__(
            name="Div",
            pretty_name="/",
            fun=np.true_divide,
            precedence=4,
            associativity="left",
            lhs=lhs,
            rhs=rhs,
        )
    
    def derivative_scheme(self, arg0, arg1):
        return (arg0 * self.args[1] - self.args[0] * arg1) / self.args[1] ** 2


class Pow(BinaryOperator):

    def __init__(self, lhs, rhs):
        super().__init__(
            name="Pow",
            pretty_name="**",
            fun=np.power,
            precedence=2,
            associativity="right",
            lhs=lhs,
            rhs=rhs,
        )
    
    def derivative_scheme(self, arg0, arg1):
        return self.args[1] * self.args[0] ** (self.args[1] - 1) * arg0 + self * log(self.args[0]) * arg1


class Exp(Function):

    def __init__(self, arg):
        super().__init__(name="Exp", pretty_name="exp", fun=np.exp, args=(arg,))
    
    def derivative_scheme(self, arg):
        return self * arg


class Log(Function):

    def __init__(self, arg):
        super().__init__(name="Log", pretty_name="log", fun=np.log, args=(arg,))
    
    def derivative_scheme(self, arg):
        return arg / self.args[0]


class Sin(Function):

    def __init__(self, arg):
        super().__init__(name="Sin", pretty_name="sin", fun=np.sin, args=(arg,))
    
    def derivative_scheme(self, arg):
        return cos(self.args[0]) * arg


class Cos(Function):

    def __init__(self, arg):
        super().__init__(name="Cos", pretty_name="cos", fun=np.cos, args=(arg,))
    
    def derivative_scheme(self, arg):
        return -sin(self.args[0]) * arg


class Tan(Function):

    def __init__(self, arg):
        super().__init__(name="Tan", pretty_name="tan", fun=np.tan, args=(arg,))
    
    def derivative_scheme(self, arg):
        return arg / cos(self.args[0]) ** 2


class Asin(Function):

    def __init__(self, arg):
        super().__init__(name="Asin", pretty_name="asin", fun=np.arcsin, args=(arg,))
    
    def derivative_scheme(self, arg):
        return arg / (1 - self.args[0] ** 2) ** Fraction(1, 2)


class Acos(Function):

    def __init__(self, arg):
        super().__init__(name="Acos", pretty_name="acos", fun=np.arccos, args=(arg,))
    
    def derivative_scheme(self, arg):
        return -arg / (1 - self.args[0] ** 2) ** Fraction(1, 2)


class Atan(Function):

    def __init__(self, arg):
        super().__init__(name="Atan", pretty_name="atan", fun=np.arctan, args=(arg,))
    
    def derivative_scheme(self, arg):
        return arg / (1 + self.args[0] ** 2)


def const(x):
    if isinstance(x, numbers.Integral):
        return Integer.create(int(x))
    elif isinstance(x, numbers.Rational):
        return Rational.create(Fraction(x))
    elif isinstance(x, numbers.Real):
        return Real.create(float(x))
    else:
        raise TypeError("x not of Integral, Rational or Real types")


def var(id):
    return Variable.create(int(id))

class Apply_on_symbol:

    def __init__(self, function, *, projection=None, auto_clear_cache=False):
        self.function = function
        self.projection = projection
        self.auto_clear_cache = auto_clear_cache
        self.cache = weakref.WeakKeyDictionary()
    
    def __call__(self, symbol):
        stack = [(symbol, 0)]
        values = []
        while stack:
            sym, argn = stack.pop()
            try:
                value = self.cache[sym]
            except KeyError:
                if argn < sym.arity:
                    stack.append((sym, argn+1))
                    stack.append((sym.args[argn], 0))
                    continue
                elif argn == sym.arity:
                    n = len(values) - argn
                    value = self.function(sym, *values[n:])
                    del values[n:]
                else:
                    assert False
                self.cache[sym] = value
            values.append(value)
        assert len(values) == 1
        if self.auto_clear_cache:
            self.cache.clear()
        if self.projection:
            return self.projection(values[0])
        else:
            return values[0]
    
    def clear_cache(self):
        self.cache.clear()

def _print_symbol_helper(symbol, *args):
    return symbol.print_scheme(*args)

print_symbol = Apply_on_symbol(_print_symbol_helper, auto_clear_cache=True)

def _pretty_print_symbol_helper(symbol, *args):
    try:
        precedence = symbol.precedence
    except AttributeError:
        precedence = -1
    return symbol.pretty_print_scheme(*args), precedence

pretty_print_symbol = Apply_on_symbol(_pretty_print_symbol_helper, projection=lambda x: x[0], auto_clear_cache=True)

class Derivative(Apply_on_symbol):

    def __init__(self, n, /):
        self.n = n
        super().__init__(self._derivative_helper, projection=reduce)
    
    def _derivative_helper(self, symbol, *args):
        try:
            if var(self.n) in symbol.vars:
                return symbol.derivative_scheme(*args)
            else:
                return const(0)
        except AttributeError:
            if isinstance(symbol, Variable) and self.n == symbol.id:
                return const(1)
            else:
                return const(0)

def matches(symbol, pattern, names=None):
    if names is None:
        names = {}
    try:
        head, *rest = pattern
    except TypeError:
        head = pattern
    if isinstance(head, str):
        if head not in names:
            names[head] = symbol
        return names[head] == symbol, names
    elif isinstance(head, numbers.Real):
        return symbol == const(head), names
    elif issubclass(head, Function):
        if type(symbol) != head:
            return False, names
        try:
            if len(symbol.args) != len(rest):
                raise ValueError("Invalid pattern")
        except NameError:
            raise ValueError("Invalid pattern")
        result = True
        for i in range(len(symbol.args)):
            tmp_result, tmp_names = matches(symbol.args[i], rest[i], names)
            result &= tmp_result
            names |= tmp_names
            if not result:
                break
        return result, names
    else:
        raise ValueError("Invalid pattern")

def build_from_pattern(pattern, names):
    try:
        head, *rest = pattern
    except TypeError:
        head = pattern
    if isinstance(head, str):
        return names[head]
    elif isinstance(head, numbers.Real):
        return const(head)
    elif issubclass(head, Function):
        args = [build_from_pattern(sub_pattern, names) for sub_pattern in rest]
        return head.create(*args)
    else:
        raise ValueError("Invalid pattern")

def replace_patterns(symbol, replacement_dict):
    for pattern in replacement_dict.keys():
        matched, names = matches(symbol, pattern)
        if matched:
            return build_from_pattern(replacement_dict[pattern], names)
    return symbol

reduction_patterns = {
    Pos: {
        (Pos, "x"): "x",
    },

    Neg: {
        (Neg, (Neg, "x")): "x",
    },

    Add: {
        (Add, "x", 0): "x",
        (Add, 0, "x"): "x",
        (Add, "x", (Neg, "y")): (Sub, "x", "y"),
        (Add, (Neg, "x"), "y"): (Sub, "y", "x"),
        (Add, "x", (Add, "y", "z")): (Add, (Add, "x", "y"), "z"),
        (Add, (Mul, "x", "y"), (Mul, "x", "z")): (Mul, "x", (Add, "y", "z")),
        (Add, (Mul, "x", "z"), (Mul, "y", "z")): (Mul, (Add, "x", "y"), "z"),
        (Add, (Log, "x"), (Log, "y")): (Log, (Mul, "x", "y")),
        (Add, (Pow, (Sin, "x"), 2), (Pow, (Cos, "x"), 2)): 1,
        (Add, (Pow, (Cos, "x"), 2), (Pow, (Sin, "x"), 2)): 1,
    },

    Sub: {
        (Sub, "x", 0): "x",
        (Sub, 0, "x"): (Neg, "x"),
        (Sub, "x", "x"): 0,
        (Sub, "x", (Neg, "y")): (Add, "x", "y"),
        (Sub, (Log, "x"), (Log, "y")): (Log, (Div, "x", "y")),
        (Sub, 1, (Pow, (Sin, "x"), 2)): (Pow, (Cos, "x"), 2),
        (Sub, 1, (Pow, (Cos, "x"), 2)): (Pow, (Sin, "x"), 2),
    },

    Mul: {
        (Mul, "x", 0): 0,
        (Mul, 0, "x"): 0,
        (Mul, "x", 1): "x",
        (Mul, 1, "x"): "x",
        (Mul, "x", (Mul, "y", "z")): (Mul, (Mul, "x", "y"), "z"),
        (Mul, "x", (Div, 1, "y")): (Div, "x", "y"),
        (Mul, (Div, 1, "x"), "y"): (Div, "y", "x"),
        (Mul, "x", (Div, "y", "x")): "y",
        (Mul, (Div, "y", "x"), "x"): "y",
        (Mul, (Exp, "x"), (Exp, "y")): (Exp, (Add, "x", "y")),
        (Mul, (Pow, "x", "y"), (Pow, "x", "z")): (Pow, "x", (Add, "y", "z")),
        (Mul, (Pow, "x", "z"), (Pow, "y", "z")): (Pow, (Mul, "x", "y"), "z"),
        (Mul, (Cos, "x"), (Tan, "x")): (Sin, "x"),
        (Mul, (Tan, "x"), (Cos, "x")): (Sin, "x"),
    },

    Div: {
        (Div, 0, "x"): 0,
        (Div, "x", 1): "x",
        (Div, "x", "x"): 1,
        (Div, (Exp, "x"), (Exp, "y")): (Exp, (Sub, "x", "y")),
        (Div, (Pow, "x", "y"), (Pow, "x", "z")): (Pow, "x", (Sub, "y", "z")),
        (Div, (Pow, "x", "z"), (Pow, "y", "z")): (Pow, (Div, "x", "y"), "z"),
        (Div, (Sin, "x"), (Cos, "x")): (Tan, "x"),
    },

    Pow: {
        (Pow, "x", 0): 1,
        (Pow, 0, "x"): 0,
        (Pow, "x", 1): "x",
        (Pow, (Pow, "x", "y"), "z"): (Pow, "x", (Mul, "y", "z")),
    }
}

def _reduce_helper(symbol, *args):
    try:
        reduction_pattern = reduction_patterns[type(symbol)]
    except KeyError:
        reduction_pattern = {}
    if args:
        if all(isinstance(arg, Constant) for arg in args):
            symbol = const(symbol.fun(*(arg.value for arg in args)))
        else:
            symbol = symbol.create(*args)
    return replace_patterns(symbol, reduction_pattern)

reduce_iteration = Apply_on_symbol(_reduce_helper)

def reduce(symbol):
    size = symbol.size
    symbol = reduce_iteration(symbol)
    while size > symbol.size:
        size = symbol.size
        symbol = reduce_iteration(symbol)
    return symbol

class Evaluate(Apply_on_symbol):

    def __init__(self, var_dict):
        super().__init__(self._evaluate_helper)
        self.var_dict = var_dict

    def _evaluate_helper(self, symbol, *args):
        if isinstance(symbol, Constant):
            return symbol.value
        elif isinstance(symbol, Variable):
            return self.var_dict[symbol.id]
        else:
            return symbol.fun(*args)
