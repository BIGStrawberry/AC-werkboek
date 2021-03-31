#!/usr/bin/env python

"""Geneste objecten om simpele wiskundige formules te representeren, met als doel deze te differentiëren en integreren."""

__author__      = "Brian van der Bijl"
__copyright__   = "Copyright 2020, Hogeschool Utrecht"

from collections import OrderedDict
import math
from IPython.display import display, Math, Markdown
from ac_exceptions import VariableError
from ac_latex import latex_ratio

def parentheses(string, outer):
    if outer:
        return string
    else:
        return "(" + string + ")"


class Function():
    def __init__(self, label, body, deriv_order=0):
        if isinstance(body, Function):
            raise SyntaxError("Nested functions not supported")
        self.body = body
        self.label = label
        self.deriv_order = deriv_order


    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__ and self.body == other.body


    def __str__(self):
        return "Function(label='" + self.label + "',body=" + str(self.body) + ",deriv_order=" + str(self.deriv_order) + ")"


    def simplify(self):
        return Function(self.label, self.body.simplify(), self.deriv_order)


    def complexity(self):
        return self.body.complexity()


    def variables(self):
        return [self.body.variables()]


    def deriv(self):
        if self.body.deriv():
            return Function(self.label, self.body.deriv().simplify(), self.deriv_order + 1)
        else:
            return None


    def integrate(self, wrt):
        return Function(self.label, Sum(self.body.integrate(wrt).simplify(), Variable('C')), self.deriv_order - 1)


    def eval(self, variables):
        return self.body.eval(variables)


    def to_latex(self, jupyter_display=True):
        if self.deriv_order >= 0:
            cont = self.label + "^{" + r"\prime" * self.deriv_order + "}"
        else:
            cont = self.label.upper()

        if self.body.variables():
            cont += "(" + ','.join(list(OrderedDict.fromkeys(self.body.variables()))) + ")"

        cont += " = " + self.body.simplify().to_latex(outer=True)

        if cont and jupyter_display:
            display(Math(cont))
            display(Markdown("<details><pre>$" + cont + "$</pre></details>"))
            return None
        else:
            return cont


class Variable():
    def __init__(self, label):
        self.label = label


    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__ and self.label == other.label


    def __str__(self):
        return "Variable(label='" + self.label + "')"


    def variables(self):
        if self.label != 'C':
            return [self.label]
        else:
            return []


    def deriv(self): # pylint: disable=R0201
        return Constant(1)


    def simplify(self):
        return self


    def complexity(self): # pylint: disable=R0201
        return 1


    def eval(self, variables):
        if self.label in variables:
            return variables[self.label]
        elif self.label == 'C': # C defaults to zero if not given
            return 0
        else:
            raise VariableError("Evaluating " + self.label + ", but no value provided")


    def to_latex(self, jupyter_display=False):
        cont = self.label
        if cont and jupyter_display:
            display(Math(cont))
            display(Markdown("<details><pre>$" + cont + "$</pre></details>"))
            return None
        else:
            return cont


class Power():
    def __init__(self, base, exponent):
        self.base = base
        self.exponent = exponent
        self.one = exponent == 0
        self.negative_exponent = self.exponent < 0


    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__ and self.base == other.base and self.exponent == other.exponent


    def __str__(self):
        return "Power(base=" + str(self.base) + ",exponent=" + str(self.exponent) + ")"


    def variables(self):
        return self.base.variables()


    def simplify(self):
        base = self.base.simplify()
        if self.exponent == 0:
            return Constant(1)
        elif self.exponent == 1:
            return self.base
        elif isinstance(base, Power):
            return Power(base.base, base.exponent * self.exponent)
        else:
            return self


    def complexity(self):
        return 1 + self.base.complexity()


    def eval(self, variables):
        return self.base.eval(variables) ** self.exponent


    def to_latex(self, jupyter_display=False):
        if self.exponent == 0:
            cont = "1"
        elif self.exponent == 1:
            cont = self.base.to_latex()
        elif hasattr(self.base, "trig_fn"):
            cont = r"\text{" + self.base.trig_fn + "}^{" + str(self.exponent) + "}" + self.base.argument.to_latex()
        elif self.exponent < 0:
            cont = r"\frac{1}{" + Power(self.base, abs(self.exponent)).to_latex() + "}"
        else:
            cont = self.base.to_latex() + "^{" + latex_ratio(self.exponent) + "}"

        if cont and jupyter_display:
            display(Math(cont))
            display(Markdown("<details><pre>$" + cont + "$</pre></details>"))
            return None
        else:
            return cont


class Negative():
    def __init__(self, inverse):
        self.inverse = inverse
        self.zero = inverse == 0


    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__ and self.inverse == other.inverse


    def __str__(self):
        return "Negative(inverse=" + str(self.inverse) + ")"


    def variables(self):
        return self.inverse.variables()


    def deriv(self):
        return Negative(self.inverse.deriv())


    def integrate(self, wrt):
        return Negative(self.inverse.integrate(wrt))


    def simplify(self):
        if isinstance(self.inverse, Negative):
            return self.inverse.inverse
        elif isinstance(self.inverse.simplify(), Constant) and self.inverse.simplify().value == 0:
            return Constant(0)
        else:
            return Negative(self.inverse.simplify())


    def complexity(self):
        return 1 + self.inverse.complexity()


    def eval(self, variables):
        return - self.inverse.eval(variables)


    def to_latex(self, jupyter_display=False):
        cont = "-" + self.inverse.to_latex()
        if cont and jupyter_display:
            display(Math(cont))
            display(Markdown("<details><pre>$" + cont + "$</pre></details>"))
            return None
        else:
            return cont


class Constant():
    def __new__(cls, value):
        if value >= 0:
            return super().__new__(cls)
        else:
            return Negative(Constant(abs(value)))


    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__ and self.value == other.value


    def __init__(self, value):
        if value >= 0:
            self.value = value
            self.zero = value == 0
            self.one = value == 1
        else:
            raise Exception("Should not happen")


    def __str__(self):
        return "Constant(value=" + str(self.value) + ")"


    def variables(self): # pylint: disable=R0201
        return []


    def deriv(self): # pylint: disable=R0201
        return Constant(0)


    def simplify(self):
        return self


    def complexity(self): # pylint: disable=R0201
        return 1


    def eval(self, variables):
        del variables
        return self.value


    def to_latex(self, jupyter_display=False):
        cont = latex_ratio(self.value)
        if cont and jupyter_display:
            display(Math(cont))
            display(Markdown("<details><pre>$" + cont + "$</pre></details>"))
            return None
        else:
            return cont


class Product():
    def __init__(self, left, right):
        self.left = left
        self.right = right


    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__ and self.left == other.left\
                                                                   and self.right == other.right


    def __str__(self):
        return "Product(left=" + str(self.left) + ",right=" + str(self.right) + ")"


    def variables(self):
        return self.left.variables() + self.right.variables()


    def simplify(self):
        left = self.left.simplify()
        right = self.right.simplify()
        if (hasattr(left, "zero") and left.zero) or (hasattr(right, "zero") and right.zero):
            result = Constant(0)
        elif hasattr(left, "one") and left.one:
            result = right.simplify()
        elif hasattr(right, "one") and right.one:
            result = left.simplify()
        elif isinstance(left, Constant) and isinstance(right, Constant):
            result = Constant(left.value * right.value)
        elif isinstance(left, Negative):
            result = Negative(Product(left.inverse, right))
        elif isinstance(right, Power) and isinstance(left, Power) and left.base == right.base:
            result = Power(left.base, left.exponent + right.exponent)
        elif isinstance(right, Power) and isinstance(left, Power) and left.exponent == right.exponent:
            result = Power(Product(left.base, right.base), left.exponent)
        elif isinstance(left, Power) and left.base == right:
            result = Power(right, left.exponent + 1)
        elif isinstance(right, Power) and left == right.base:
            result = Power(left, right.exponent + 1)
        elif isinstance(left, Product) and isinstance(right, Power) and left.right == right.base:
            result = Product(left.left, Power(right.base, right.exponent + 1))
        elif left == right:
            result = Power(left, 2)
        else:
            result = Product(self.left.simplify(), self.right.simplify())
        return result


    def complexity(self):
        return 1 + max(self.left.complexity(), self.right.complexity())


    def eval(self, variables):
        return self.left.eval(variables) * self.right.eval(variables)


    def to_latex(self, jupyter_display=False):
        if isinstance(self.left, Constant) and isinstance(self.right.simplify(), Variable):
            cont = self.left.to_latex() + self.right.to_latex()
        elif hasattr(self.right, "negative_exponent") and self.right.negative_exponent:
            cont = r"\frac{" + self.left.to_latex() + "}{" + Power(self.right.base, abs(self.right.exponent)).to_latex() + "}"
        else:
            cont = (self.left.to_latex() + r" \  " + self.right.to_latex())

        if cont and jupyter_display:
            display(Math(cont))
            display(Markdown("<details><pre>$" + cont + "$</pre></details>"))
            return None
        else:
            return cont


class Sum():
    def __init__(self, left, right):
        self.left = left
        self.right = right


    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__ and self.left == other.left\
                                                                   and self.right == other.right


    def __str__(self):
        return "Sum(left=" + str(self.left) + ",right=" + str(self.right) + ")"


    def variables(self):
        return self.left.variables() + self.right.variables()


    def simplify(self):
        left = self.left.simplify()
        right = self.right.simplify()
        if hasattr(left, "zero") and left.zero:
            return right.simplify()
        elif hasattr(right, "zero") and right.zero:
            return left.simplify()
        elif isinstance(left, Constant) and isinstance(right, Constant):
            return Constant(left.value + right.value)
        else:
            return Sum(self.left.simplify(), self.right.simplify())


    def complexity(self):
        return 1 + max(self.left.complexity(), self.right.complexity())


    def eval(self, variables):
        return self.left.eval(variables) + self.right.eval(variables)


    def to_latex(self, jupyter_display=False, outer=False):
        if isinstance(self.right, Negative):
            cont = parentheses((self.left.to_latex()) + " - " + (self.right.inverse.to_latex()), outer)
        else:
            cont = parentheses((self.left.to_latex()) + " + " + (self.right.to_latex()), outer)
        if cont and jupyter_display:
            display(Math(cont))
            display(Markdown("<details><pre>$" + cont + "$</pre></details>"))
            return None
        else:
            return cont


class Sin():
    def __init__(self, argument):
        self.argument = argument
        self.trig_fn = "sin"


    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__ and self.argument == other.argument


    def __str__(self):
        return "Sin(argument=" + str(self.argument) + ")"


    def variables(self):
        return self.argument.variables()


    def simplify(self):
        return self


    def complexity(self):
        return 1 + self.argument.complexity()


    def eval(self, variables):
        return math.sin(self.argument.eval(variables))


    def to_latex(self, jupyter_display=False):
        cont = r"\text{sin}(" + self.argument.to_latex() + ")"
        if cont and jupyter_display:
            display(Math(cont))
            display(Markdown("<details><pre>$" + cont + "$</pre></details>"))
            return None
        else:
            return cont


class Cos():
    def __init__(self, argument):
        self.argument = argument
        self.trig_fn = "cos"


    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__ and self.argument == other.argument


    def __str__(self):
        return "Cos(argument=" + str(self.argument) + ")"


    def variables(self):
        return self.argument.variables()


    def simplify(self):
        return self


    def complexity(self):
        return 1 + self.argument.complexity()


    def eval(self, variables):
        return math.cos(self.argument.eval(variables))


    def to_latex(self, jupyter_display=False):
        cont = r"\text{cos}(" + self.argument.to_latex() + ")"
        if cont and jupyter_display:
            display(Math(cont))
            display(Markdown("<details><pre>$" + cont + "$</pre></details>"))
            return None
        else:
            return cont


class Tan():
    def __init__(self, argument):
        self.argument = argument
        self.trig_fn = "tan"


    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__ and self.argument == other.argument


    def __str__(self):
        return "Tan(argument=" + str(self.argument) + ")"


    def variables(self):
        return self.argument.variables()


    def simplify(self):
        return self


    def complexity(self):
        return 1 + self.argument.complexity()


    def eval(self, variables):
        return math.tan(self.argument.eval(variables))


    def to_latex(self, jupyter_display=False):
        cont = r"\text{tan}(" + self.argument.to_latex() + ")"
        if cont and jupyter_display:
            display(Math(cont))
            display(Markdown("<details><pre>$" + cont + "$</pre></details>"))
            return None
        else:
            return cont


class Cot():
    def __init__(self, argument):
        self.argument = argument
        self.trig_fn = "cot"


    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__ and self.argument == other.argument


    def __str__(self):
        return "Cot(argument=" + str(self.argument) + ")"


    def variables(self):
        return self.argument.variables()


    def simplify(self):
        return self


    def complexity(self):
        return 1 + self.argument.complexity()


    def deriv(self):
        return Negative(Product(self.argument.deriv(),
                                Power(Csc(Variable('x')), 2)))


    def eval(self, variables):
        return 1 / math.tan(self.argument.eval(variables))


    def to_latex(self, jupyter_display=False):
        cont = r"\text{cot}(" + self.argument.to_latex() + ")"
        if cont and jupyter_display:
            display(Math(cont))
            display(Markdown("<details><pre>$" + cont + "$</pre></details>"))
            return None
        else:
            return cont


class Sec():
    def __init__(self, argument):
        self.argument = argument
        self.trig_fn = "sec"


    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__ and self.argument == other.argument


    def __str__(self):
        return "Sec(argument=" + str(self.argument) + ")"


    def variables(self):
        return self.argument.variables()


    def simplify(self):
        return self


    def complexity(self):
        return 1 + self.argument.complexity()


    def deriv(self):
        return Product(self.argument.deriv(),
                       Product(Sec(self.argument), Tan(self.argument)))


    def eval(self, variables):
        return 1 / math.cos(self.argument.eval(variables))


    def to_latex(self, jupyter_display=False):
        cont = r"\text{sec}(" + self.argument.to_latex() + ")"
        if cont and jupyter_display:
            display(Math(cont))
            display(Markdown("<details><pre>$" + cont + "$</pre></details>"))
            return None
        else:
            return cont


class Csc():
    def __init__(self, argument):
        self.argument = argument
        self.trig_fn = "csc"


    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__ and self.argument == other.argument


    def __str__(self):
        return "Csc(argument=" + str(self.argument) + ")"


    def variables(self):
        return self.argument.variables()


    def simplify(self):
        return self


    def complexity(self):
        return 1 + self.argument.complexity()


    def deriv(self):
        return Negative(Product(self.argument.deriv(),
                                Product(Csc(self.argument), Cot(self.argument))))


    def eval(self, variables):
        return 1 / math.sin(self.argument.eval(variables))


    def to_latex(self, jupyter_display=False):
        cont = r"\text{csc}(" + self.argument.to_latex() + ")"
        if cont and jupyter_display:
            display(Math(cont))
            display(Markdown("<details><pre>$" + cont + "$</pre></details>"))
            return None
        else:
            return cont


class E(): # pylint: disable=C0103
    def __init__(self, exponent):
        self.exponent = exponent


    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__ and self.exponent == other.exponent


    def __str__(self):
        return "E(exponent=" + str(self.exponent) + ")"


    def variables(self):
        return self.exponent.variables()


    def simplify(self):
        return self


    def complexity(self):
        return 1 + self.exponent.complexity()


    def eval(self, variables):
        return math.exp(self.exponent.eval(variables))


    def to_latex(self, jupyter_display=False):
        cont = "e^{" + self.exponent.to_latex() + "}"
        if cont and jupyter_display:
            display(Math(cont))
            display(Markdown("<details><pre>$" + cont + "$</pre></details>"))
            return None
        else:
            return cont


class Exponent():
    def __init__(self, base, exponent):
        self.base = base
        self.exponent = exponent


    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__ and self.base == other.base\
                                                                   and self.exponent == other.exponent


    def __str__(self):
        return "Exponent(base=" + str(self.base) + ",exponent=" + str(self.exponent) + ")"


    def variables(self):
        return self.base.variables()+self.exponent.variables()


    def simplify(self):
        return self


    def complexity(self):
        return 1 + max(self.base.complexity(), self.exponent.complexity())


    def eval(self, variables):
        return self.base.eval(variables) ** self.exponent.eval(variables)


    def to_latex(self, jupyter_display=False):
        cont = self.base.to_latex() + "^{" + self.exponent.to_latex() + "}"
        if cont and jupyter_display:
            display(Math(cont))
            display(Markdown("<details><pre>$" + cont + "$</pre></details>"))
            return None
        else:
            return cont


class Ln():
    def __init__(self, argument):
        self.argument = argument


    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__ and self.argument == other.argument


    def __str__(self):
        return "Ln(argument=" + str(self.argument) + ")"


    def variables(self):
        return self.argument.variables()


    def simplify(self):
        return self


    def complexity(self):
        return 1 + self.argument.complexity()


    def eval(self, variables):
        return math.log(self.argument.eval(variables))


    def to_latex(self, jupyter_display=False):
        cont = r"\text{ln}(" + self.argument.to_latex() + ")"
        if cont and jupyter_display:
            display(Math(cont))
            display(Markdown("<details><pre>$" + cont + "$</pre></details>"))
            return None
        else:
            return cont


class Log():
    def __init__(self, base, argument):
        self.base = base
        self.argument = argument


    def __eq__(self, other):
        return self.__class__.__name__ == other.__class__.__name__ and self.base == other.base\
                                                                   and self.argument == other.argument


    def __str__(self):
        return "Log(base=" + str(self.base) + ",argument=" + str(self.argument) + ")"


    def variables(self):
        return self.argument.variables()


    def simplify(self):
        return self


    def complexity(self):
        return 1 + max(self.base.complexity(), self.argument.complexity())


    def eval(self, variables):
        return math.log(self.argument.eval(variables), self.base.eval(variables))


    def to_latex(self, jupyter_display=False):
        cont = r"\text{log}_{" + self.base.to_latex() + "}(" + self.argument.to_latex() + ")"
        if cont and jupyter_display:
            display(Math(cont))
            display(Markdown("<details><pre>$" + cont + "$</pre></details>"))
            return None
        else:
            return cont
