#!/usr/bin/env python

"""Automatisch gegenereerde uitwerkingen van de sommen."""

__author__      = "Brian van der Bijl"
__copyright__   = "Copyright 2020, Hogeschool Utrecht"

import numpy as np
from scipy.linalg import lu
from sympy import Matrix
from IPython.display import display, Markdown, Math

from ac_random import RNG, random_tensor, random_sys_of_eq,\
                      DEGENERATE, NONDEGENERATE, DONT_CARE,\
                      halve, hide_one
from ac_latex import latex_bmatrix


def negatieven_en_sommen():
    # pylint: disable=C0103
    RNG().set(0)

    u = random_tensor(r"\vec u", ret=True, details=False)
    v = random_tensor(r"\vec v", ret=True, details=False)
    w = random_tensor(r"\vec w", 3, ret=True, details=False)
    x = random_tensor(r"\vec x", 3, ret=True, details=False)
    y = random_tensor(r"\vec y", 5, ret=True, details=False)
    z = random_tensor(r"\vec z", 5, ret=True, details=False)
    display(Markdown("<hr>"))
    latex_bmatrix(-u, r"-\vec u", details=False)
    latex_bmatrix(-v, r"-\vec v", details=False)
    latex_bmatrix(-w + x, r"-\vec w + \vec x", details=False)
    latex_bmatrix(-y-z, r"- \vec y - \vec z", details=False)


def lineaire_combinaties():
    # pylint: disable=C0103
    RNG().set(1)

    u = random_tensor(r"\vec u", ret=True, details=False)
    v = random_tensor(r"\vec v", ret=True, details=False)
    w = random_tensor(r"\vec w", 2, ret=True, details=False)
    x = random_tensor(r"\vec x", 2, ret=True, details=False)
    y = random_tensor(r"\vec y", 4, ret=True, details=False)
    z = random_tensor(r"\vec z", 4, ret=True, details=False)
    display(Markdown("<hr>"))
    latex_bmatrix(3*u, r"3\vec{u}", details=False)
    latex_bmatrix(-5*v, r"-5\vec{v}", details=False)
    latex_bmatrix(v/2, r"\frac{1}{2} \vec{v}", details=False)
    latex_bmatrix(3*w + 4*x, r"3\vec{w} + 4\vec{x}", details=False)
    latex_bmatrix(8*y-z/2, r"8\vec{y} - \frac{1}{2}\vec{z}", details=False)


def inwendige_producten():
    # pylint: disable=C0103
    RNG().set(3)

    u = random_tensor(r"\vec u", 2, ret=True, details=False)
    v = random_tensor(r"\vec v", 3, ret=True, details=False)
    w = random_tensor(r"\vec w", 2, ret=True, details=False)
    x = random_tensor(r"\vec x", 4, ret=True, details=False)
    y = random_tensor(r"\vec y", 4, ret=True, details=False)
    display(Markdown("<hr>"))
    dot = lambda u, v: (u.T@v)[0][0]

    table = f"""$\\begin{{array}}{{|r|c|c|c|c|c|}} \\hline
                & \\vec u & \\vec v & \\vec w & \\vec x & \\vec y \\\\ \\hline
                \\vec u & {dot(u,u)}
                        & \\bot
                        & {dot(u,w)}
                        & \\bot
                        & \\bot
                        \\\\ \\hline
                \\vec v & \\bot
                        & {dot(v,v)}
                        & \\bot
                        & \\bot
                        & \\bot
                        \\\\ \\hline
                \\vec w & {dot(w,u)}
                        & \\bot
                        & {dot(w,w)}
                        & \\bot
                        & \\bot
                        \\\\ \\hline
                \\vec x & \\bot
                        & \\bot
                        & \\bot
                        & {dot(x,x)}
                        & {dot(x,y)}
                        \\\\ \\hline
                \\vec y & \\bot
                        & \\bot
                        & \\bot
                        & {dot(y,x)}
                        & {dot(y,y)}
                        \\\\ \\hline
                \\end{{array}}$"""
    display(Markdown(table))
    display(Markdown("<hr>"))
    display(Markdown(r"**Wat valt je op qua symmetrie aan de tabel? Is $\langle \vec u | \vec v\rangle$ hetzelfde als $\langle \vec v | \vec u \rangle$?**"))
    display(Markdown(r"*Ja, dus de volgorde maakt niet uit / het inwendig product is hier commutatief.*"))
    display(Markdown(r"**Heeft $\langle \vec u | \vec u \rangle$ een speciale betekenis? Of de wortel daarvan?**"))
    display(Markdown(r"*De wortel van het inwendig product van een vector met zichzelf is de lengte van de vector (stelling van Pythagoras)*"))


def matrix_vector():
    # pylint: disable=C0103
    RNG().set(4)

    u = random_tensor(r"\vec u", 3, ret=True, details=False)
    v = random_tensor(r"\vec v", 2, ret=True, details=False)
    M = random_tensor(r"\mathbf{M}", (3,2), ret=True, details=False)
    N = random_tensor(r"\mathbf{N}", (2,3), ret=True, details=False)
    O = random_tensor(r"\mathbf{O}", (2,2), ret=True, details=False)

    RNG().set(2).consume_entropy(0x06, -0x14, 0x14)

    pa = random_tensor(r"\vec {p_a}", 2, ret=True, details=False)
    pb = random_tensor(r"\vec {p_b}", 2, ret=True, details=False)
    qa = random_tensor(r"\vec {q_a}", 4, ret=True, details=False)
    qb = random_tensor(r"\vec {q_b}", 4, ret=True, details=False)

    display(Markdown("<hr>"))

    latex_bmatrix(M.dot(v), r"\mathbf{M}\vec{v}", details=False)
    display(Markdown(r"$\mathbf{M}\vec{u} = \bot$"))
    display(Markdown(r"$\mathbf{N}\vec{v} = \bot$"))
    latex_bmatrix(N.dot(u), r"\mathbf{N}\vec{u}", details=False)
    latex_bmatrix(O.dot(N.dot(u)), r"\mathbf{O} (\mathbf{N} \vec u)", details=False)

    display(Markdown("<hr>"))

    P = np.hstack((pa,pb))
    Q = np.hstack((qa,qb))

    latex_bmatrix(P, r"\mathbf{P}", details=False)
    latex_bmatrix(Q, r"\mathbf{Q}", details=False)

    display(Markdown("<hr>"))

    latex_bmatrix(P.dot(np.array((3,4))).reshape(2,1), r"\mathbf{P} \begin{bmatrix}3 \\ 4\end{bmatrix}", details=False)
    latex_bmatrix(Q.dot(np.array((8,-0.5))).reshape(4,1), r"\mathbf{Q} \begin{bmatrix}8 \\ -\frac{1}{2}\end{bmatrix}", details=False)


def matrix_producten():
    # pylint: disable=C0103
    RNG().set(4)

    u = random_tensor(r"\vec u", 3, ret=True, details=False)
    RNG().consume_entropy(0x02, -0x14, 0x14)
    M = random_tensor(r"\mathbf{M}", (3,2), ret=True, details=False)
    N = random_tensor(r"\mathbf{N}", (2,3), ret=True, details=False)
    O = random_tensor(r"\mathbf{O}", (2,2), ret=True, details=False)

    display(Markdown("<hr>"))

    latex_bmatrix(O.dot(N.dot(u)), r"\mathbf{O} (\mathbf{N} \vec u)", details=False)
    latex_bmatrix(O.dot(N).dot(u), r"(\mathbf{O} \mathbf{N}) \vec u", details=False)
    latex_bmatrix(O.dot(N), r"\mathbf{O} \mathbf{N}", details=False)

    display(Markdown("<hr>"))

    display(Markdown(r"$\mathbf{O}\mathbf{M} = \bot$"))
    latex_bmatrix(O.dot(O), r"\mathbf{O} \mathbf{O}", details=False)
    display(Markdown(r"$\mathbf{N}\mathbf{N} = \bot$"))
    latex_bmatrix(N.dot(M), r"\mathbf{N} \mathbf{M}", details=False)
    display(Markdown(r"$\mathbf{N}\mathbf{O} = \bot$"))
    latex_bmatrix(M.dot(N), r"\mathbf{M} \mathbf{N}", details=False)
    display(Markdown(r"$\mathbf{M}\mathbf{M} = \bot$"))
    latex_bmatrix(M.dot(O), r"\mathbf{M} \mathbf{O}", details=False)


def gauss_jordan():
    # pylint: disable=C0103
    RNG().set(5)
    (A,b) = random_sys_of_eq(details=False, ret=True)
    b = (np.linalg.det(A)*b).astype(int)

    display(Markdown("<hr>"))

    display(Markdown(f"${A[0][0]} x_0 + {A[0][1]} x_1 + {A[0][2]} x_2 = {b[0]}$"))
    display(Markdown(f"${A[1][0]} x_0 + {A[1][1]} x_1 + {A[1][2]} x_2 = {b[1]}$"))
    display(Markdown(f"${A[2][0]} x_0 + {A[2][1]} x_1 + {A[2][2]} x_2 = {b[2]}$"))

    x = np.linalg.solve(A, b)

    display(Markdown("<hr>"))

    display(Markdown("**Hier moet de je de Gauss-Jordan eliminatie uitvoeren; hier zijn meerdere wegen mogelijk, dus check vooral of je logische stappen uitvoert en of het eindresultaat klopt met wat hieronder staat gegeven.**"))

    display(Markdown(f"$x_0 = {x[0].round(0).astype(int)}$"))
    display(Markdown(f"$x_1 = {x[1].round(0).astype(int)}$"))
    display(Markdown(f"$x_2 = {x[2].round(0).astype(int)}$"))


def determinanten():
    # pylint: disable=C0103
    RNG().set(6)

    M = random_tensor(r"\textbf{M}", (2,2), singular=NONDEGENERATE, ret=True, details=False)
    N = random_tensor(r"\textbf{N}", (2,2), singular=DEGENERATE, ret=True, details=False)
    O = random_tensor(r"\textbf{O}", (2,2), singular=NONDEGENERATE, ret=True, details=False)
    P = random_tensor(r"\textbf{P}", (3,3), singular=NONDEGENERATE, interval=(0,5), ret=True, details=False)
    Q = random_tensor(r"\textbf{Q}", (3,3), singular=DEGENERATE, interval=(0,5), ret=True, details=False)

    display(Markdown("<hr>"))

    display(Markdown(f"$\\text{{det}}(\\mathbf{{M}}) = {np.linalg.det(M).round(0).astype(int)}$"))
    display(Markdown(f"$\\text{{det}}(\\mathbf{{N}}) = {np.linalg.det(N).round(0).astype(int)}$"))
    display(Markdown(f"$\\text{{det}}(\\mathbf{{O}}) = {np.linalg.det(O).round(0).astype(int)}$"))
    display(Markdown(f"$\\text{{det}}(\\mathbf{{P}}) = {np.linalg.det(P).round(0).astype(int)}$"))
    display(Markdown(f"$\\text{{det}}(\\mathbf{{Q}}) = {np.linalg.det(Q).round(0).astype(int)}$"))


def inverses():
    # pylint: disable=C0103
    RNG().set(6)

    def fr_matrix(M, divisor=1, label=None):
        def fraction(n):
            n = int(n)
            gcd = np.gcd(n,divisor)
            s = '' if divisor*n > 0 else '-'
            n = abs((n/gcd).round(0).astype(int))
            d = abs((divisor/gcd).round(0).astype(int))

            if d == 1:
                return s+str(n)
            else:
                return f"{s}\\frac{{{n}}}{{{d}}}"

        if len(M.shape) > 2:
            raise ValueError('bmatrix can at most display two dimensions')
        lines = str(M).replace("[", "").replace("]", "").splitlines()
        if label:
            result = [label + " = "]
        else:
            result = [""]
        result += [r"\begin{bmatrix}"]
        result += ["  " + " & ".join(map(fraction, l.split())) + r"\\" for l in lines]
        result +=  [r"\end{bmatrix}"]
        display(Math("\n".join(result)))

    adj = lambda M: np.array(((M[1][1], -M[0][1]),(-M[1][0], M[0][0])))

    M = random_tensor(r"\textbf{M}", (2,2), singular=NONDEGENERATE, ret=True, details=False)
    det = np.linalg.det(M).round(0).astype(int)
    latex_bmatrix(adj(M), r"$\text{adj}(\mathbf{M})", details=False)
    display(Markdown(f"$\\text{{det}}(\\mathbf{{M}}) = {det}$"))
    fr_matrix(adj(M), det, r"$\mathbf{M}^{-1}")

    display(Markdown("<hr>"))

    N = random_tensor(r"\textbf{N}", (2,2), singular=DEGENERATE, ret=True, details=False)
    latex_bmatrix(adj(N), r"$\text{adj}(\mathbf{N})", details=False)
    display(Markdown(f"$\\text{{det}}(\\mathbf{{N}}) = {np.linalg.det(N).round(0).astype(int)}$"))
    display(Markdown(r"$\mathbf{N}^{-1} = \bot$"))

    display(Markdown("<hr>"))

    O = random_tensor(r"\textbf{O}", (2,2), singular=NONDEGENERATE, ret=True, details=False)
    det = np.linalg.det(O).round(0).astype(int)
    latex_bmatrix(adj(O), r"$\text{adj}(\mathbf{O})", details=False)
    display(Markdown(f"$\\text{{det}}(\\mathbf{{O}}) = {det}$"))
    fr_matrix(adj(O), det, r"$\mathbf{O}^{-1}")


def rank():
    # pylint: disable=C0103
    RNG().set(7)

    def stats(M, label):
        def basis(vs):
            res = r"\left\{"
            res += ", ".join([r"\begin{bmatrix}" + r"\\".join([str(c) for c in v]) + r"\end{bmatrix}" for v in vs])
            res += r"\right\}"
            return res

        U = lu(M)[2]
        cols = []
        for i in range(U.shape[0]):
            r = np.flatnonzero(U[i, :])
            if r.size > 0:
                cols.append(r[0])
        cols = dict.fromkeys(cols)
        display(Markdown(f"$\\text{{I}}(\\textbf{{{label}}}) = {basis([M[:,c] for c in cols])}$"))
        display(Markdown(f"$\\text{{Rank}}(\\textbf{{{label}}}) = {np.linalg.matrix_rank(M)}$"))
        display(Markdown(f"$\\text{{Ker}}(\\textbf{{{label}}}) = {basis(Matrix(M).nullspace())}$"))
        display(Markdown(f"$\\text{{Nullity}}(\\textbf{{{label}}}) = {M.shape[1] - np.linalg.matrix_rank(M)}$"))

    M = random_tensor(r"\textbf{M}", (2,2), singular=NONDEGENERATE, interval=(0,5), ret=True, details=False)
    stats(M, 'M')
    display(Markdown("<hr>"))

    N = random_tensor(r"\textbf{N}", (2,2), singular=DEGENERATE, interval=(0,5), ret=True, details=False)
    stats(N, 'N')
    display(Markdown("<hr>"))

    O = random_tensor(r"\textbf{O}", (3,3), singular=NONDEGENERATE, interval=(0,5), ret=True, details=False)
    stats(O, 'O')
    display(Markdown("<hr>"))

    P = random_tensor(r"\textbf{P}", (3,3), singular=DEGENERATE, interval=(0,5), ret=True, details=False)
    stats(P, 'P')
    display(Markdown("<hr>"))

    Q = random_tensor(r"\textbf{Q}", (2,3), singular=DONT_CARE, interval=(0,5), ret=True, details=False)
    stats(Q, 'Q')
    display(Markdown("<hr>"))

    R = random_tensor(r"\textbf{R}", (2,3), singular=DONT_CARE, interval=(0,5), ret=True, details=False)
    stats(R, 'R')

    RNG().set(1)


def derivatives():
    # pylint: disable=C0103
    RNG().set(8)
    a, b, c = np.random.randint(2, 7, 3)
    text = f"Gegeven $f(x) = ({a-1}- {hide_one(b)}x)^{hide_one(c)}$, bepaal $f^\\prime(x)$"
    display(Markdown("**(a)** " + text + r"$\\\\[3mm]$"))
    if c-1 == 1:
        xterm = "x"
    else:
        xterm = f"x^{c-1}"
    display(Markdown(f"""\\begin{{align}}f^\\prime(x)
      &= {-b}\\cdot{c}({a-1}- {hide_one(b)}{xterm}) \\\\[1mm]
      &= {hide_one(b*b*c)}{xterm} {-b*c*(a-1)}
      \\end{{align}}"""))
    display(Markdown("*Je kan er voor kiezen de polynoom eerst helemaal uit te schrijven en hier de afgeleidde van te nemen; deze kan daarna wel of niet versimpeld worden.*"))

    display(Markdown("<hr>"))

    a, b, c, d = np.random.randint(2, 7, 4)
    text = f"Gegeven $g(x) = {hide_one(a-1)}x^{hide_one(b)}\\ \\text{{tan}}({hide_one(c)}x^{hide_one(d)})$, geef $g^\\prime(x)$"
    display(Markdown("**(b)** " + text + r"$\\\\[3mm]$"))
    inner = f"{hide_one(c)}x^{hide_one(d)}"
    tan = r"\text{tan}(" + inner + ")"
    sec = r"\text{sec}^2(" + inner + ")"
    g = (a-1) * np.gcd(b, c*d)
    display(Markdown(f"""\\begin{{align}}g^\\prime(x)
      &= {(a-1)*b}x^{hide_one(b-1)}\\ {tan} + {a-1}x^{b}{c*d}x^{hide_one(d-1)}\\ {sec} \\\\[1mm]
      &= {(a-1)*b}x^{hide_one(b-1)}\\ {tan} + {(a-1)*c*d}x^{{{hide_one(b+d-1)}}}\\ {sec} \\\\[1mm]
      &= {hide_one(g)}x^{{{hide_one(b-1)}}}\\ \\big({hide_one((a-1)*b/g)}{tan} + {hide_one((a-1)*c*d/g)}x^{{{d}}}\\ {sec}\\big)
      \\end{{align}}"""))

    display(Markdown("<hr>"))

    a, b, c, d = np.random.randint(2, 7, 4)
    text = f"Gegeven $h(x) = \\text{{log}}_{a}({b-1}x-{c}x^{d})$, geef $h^\\prime(x)$"
    display(Markdown("**(c)** " + text + r"$\\\\[3mm]$"))
    g = np.gcd(c,b-1)
    if g != 1:
        simplification = f"\\\\[2mm] &= \\frac{{{int((b-1)/g)}-{hide_one(c*d/g)}x^{{{d-1}}}}}{{({hide_one((b-1)/g)}x-{hide_one(c/g)}x^{d})\\ \\text{{ln}}({a})}}"
    else:
        simplification = ""
    display(Markdown(f"""\\begin{{align}}h^\\prime(x)
     &= \\frac{{{b-1}-{hide_one(c*d)}x^{{{d-1}}}}}{{({b-1}x-{c}x^{d})\\ \\text{{ln}}({a})}} {simplification} \\end{{align}}"""))

def double_derivatives():
    # pylint: disable=C0103
    RNG().set(12)
    a, b = np.random.randint(2, 7, 2)
    text = f"Gegeven $k(x) = \\frac{{{a}}}{{x^{b}}}$, geef $k^{{\\prime\\prime}}(x)$"
    display(Markdown("**(a)** " + text + r"$\\\\[3mm]$"))
    display(Markdown(f"""\\begin{{align}}
    k(x)                    &= {a}x^{{{-b}}} \\\\[1mm]
    k^\\prime(x)            &= {a*-b}x^{{{-b-1}}} \\\\[1mm]
    k^{{\\prime\\prime}}(x) &= {a*b*(b+1)}x^{{{-b-2}}} \\\\[2mm]
                            &= \\frac{{{a*b*(b+1)}}}{{x^{{{b+2}}}}}
     \\end{{align}}"""))

    display(Markdown("<hr>"))

    a, b = np.random.randint(2, 7, 2)
    text = f"Gegeven $\\frac{{dy}}{{dx}} = x^{a} - {hide_one(b-1)}y$, geef $\\frac{{d^2y}}{{dx^2}}$"
    display(Markdown("**(b)** " + text + r"$\\\\[3mm]$"))
    display(Markdown(f"""\\begin{{align}}\\frac{{d^2y}}{{dx^2}}
      &= \\frac{{dy}}{{dx}}x^{a} - {hide_one(b-1)}\\frac{{dy}}{{dx}} y \\\\[1mm]
      &= {a}x^{hide_one(a-1, True)} - {hide_one(b-1)} (x^{a} - {hide_one(b-1)}y) \\\\[1mm]
      &= {a}x^{hide_one(a-1, True)} - {hide_one(b-1)}x^{a} + {hide_one((b-1)**2)}y
      \\end{{align}}"""))

def implicit_diff():
    # pylint: disable=C0103
    RNG().set(13)
    a, b, c, d = np.random.randint(2, 7, 4)
    text = f"Gegeven ${hide_one(a-1)}x^3y - {hide_one(b-1)}x^2 + {hide_one(c-1)}y^4 = {2*d}$, geef $\\frac{{dy}}{{dx}}$"
    display(Markdown("**(a)** " + text + r"$\\\\[3mm]$"))
    display(Markdown(f"""\\begin{{align}}
    \\frac{{dy}}{{dx}}({hide_one(a-1)}x^3y - {hide_one(b-1)}x^2 + {hide_one(c-1)}y^4) &= \\frac{{dy}}{{dx}} {2*d} \\\\[2mm]
    {hide_one(a-1)}\\frac{{dy}}{{dx}}x^3y - {hide_one(b-1)} \\frac{{dy}}{{dx}} x^2 + {hide_one(c-1)}\\frac{{dy}}{{dx}}y^4 &= 0 \\\\[2mm]
    {hide_one(a-1)}(3x^2y + x^3 \\frac{{dy}}{{dx}}) - {hide_one(b-1)}(2x) + {hide_one(c-1)}(4y^3) \\frac{{dy}}{{dx}} &= 0 \\\\[2mm]
    {hide_one(3*(a-1))}x^2y + {hide_one(a-1)}x^3 \\frac{{dy}}{{dx}} - {hide_one(2*(b-1))}x + {hide_one(4*(c-1))}y^3 \\frac{{dy}}{{dx}} &= 0 \\\\[2mm]
    {hide_one(a-1)}x^3 \\frac{{dy}}{{dx}} + {hide_one(4*(c-1))}y^3 \\frac{{dy}}{{dx}} &= {hide_one(2*(b-1))}x - {hide_one(3*(a-1))}x^2y \\\\[2mm]
    ({hide_one(a-1)}x^3 + {hide_one(4*(c-1))}y^3) \\frac{{dy}}{{dx}} &= {hide_one(2*(b-1))}x - {hide_one(3*(a-1))}x^2y \\\\[2mm]
    \\frac{{dy}}{{dx}} &= \\frac{{{hide_one(2*(b-1))}x - {hide_one(3*(a-1))}x^2y}}{{{hide_one(a-1)}x^3 + {hide_one(4*(c-1))}y^3}} \\\\[2mm]
    \\end{{align}}"""))


def integrals():
    # pylint: disable=C0103
    RNG().set(9)

    a, b = np.random.randint(2, 7, 2)
    display(Markdown("**(a)**"))
    display(Markdown(f"""\\begin{{align}}
      \\int \\sqrt[{a}]x^{b}\\ dx
      &= \\int  x^{{\\frac{{{b}}}{{{a}}}}} \\ dx \\\\[2mm]
      &= \\frac{{{a}}}{{{a+b}}} x^{{\\frac{{{a+b}}}{{{a}}}}} + C\\\\[2mm]
      &= \\frac{{{a}}}{{{a+b}}} \\sqrt[{a}]x^{{{a+b}}} + C
    \\end{{align}}"""))

    display(Markdown("<hr>"))

    a, b, c, d, e = np.random.randint(3, 9, 5)
    a = (a % 3) + 2
    b = (b % 3) + 1
    a = a if a != b else a+1
    a, b = max(a, b), min(a, b)
    e = e if e != d else d+e
    d, e = min(d, e), max(d, e)
    display(Markdown("**(b)**"))
    display(Markdown(f"""\\begin{{align}}
      \\int_{{{d}}}^{{{e}}} {a*b}x^{{{a-1}}} - {b*c}x^{{{b-1}}}\\ dx
      &= {b}x^{{{a}}} - {c}x^{{{b}}} \\ \\biggr\\rvert_{{{d}}}^{{{e}}}  \\\\[2mm]
      &= \\big({b}\\cdot{{{e}}}^{{{a}}} - {c}\\cdot{{{e}}}^{{{b}}}\\big) - \\big({b}\\cdot{{{d}}}^{{{a}}} - {c}\\cdot{{{d}}}^{{{b}}}\\big) \\\\[2mm]
      &= \\big({b}\\cdot{{{e**a}}} - {c}\\cdot{{{e**b}}}\\big) - \\big({b}\\cdot{{{d**a}}} - {c}\\cdot{{{d**b}}}\\big) \\\\[2mm]
      &= \\big({b*e**a} - {c*e**b}\\big) - \\big({b*d**a} - {c*d**b}\\big) \\\\[2mm]
      &= {b*e**a-c*e**b} - {b*d**a-c*d**b} \\\\[2mm]
      &= {(b*e**a-c*e**b)-(b*d**a-c*d**b)} \\\\[2mm]
    \\end{{align}}"""))


def integrals_billy():
    # pylint: disable=C0103
    RNG().set(10)

    a, b, c = np.random.randint(2, 7, 3)
    display(Markdown("**(b)**"))
    display(Markdown(f"""\\begin{{align}}
      \\int_{min(a,b)}^{max(a,b)+2} {c}e^x\\ dx
      &= {c} \\int_{min(a,b)}^{max(a,b)+2} e^x\\ dx \\\\[1mm]
      &= {c}\\cdot e^x\\ \\biggr\\rvert_{min(a,b)}^{max(a,b)+2} \\\\[1mm]
      &= {c} (e^{max(a,b)+2} - e^{min(a,b)})\\\\[1mm]
      &= {c} e^{max(a,b)+2} - {c} e^{min(a,b)}
    \\end{{align}}"""))

    display(Markdown("<hr>"))

    a, b, c = np.random.randint(2, 5, 3)
    display(Markdown("**(a)**"))
    display(Markdown(f"""\\begin{{align}}
      \\int_{{{halve(min(a,b))}\\pi}}^{{{halve(max(a,b)+2)}\\pi}} -{c} \\text{{sin}}(x)\\ dx
      &= -{c} \\int_{{{halve(min(a,b))}\\pi}}^{{{halve(max(a,b)+2)}\\pi}} \\text{{sin}}(x)\\ dx \\\\[1mm]
      &= -{c}\\cdot -\\text{{cos}}(x)\\ \\biggr\\rvert_{{{halve(min(a,b))}\\pi}}^{{{halve(max(a,b)+2)}\\pi}} \\\\[1mm]
      &= {c} \\text{{cos}}(x)\\ \\biggr\\rvert_{{{halve(min(a,b))}\\pi}}^{{{halve(max(a,b)+2)}\\pi}} \\\\[1mm]
      &= {c} \\left(\\text{{cos}}\\left({halve(max(a,b)+2)}\\pi\\right) - \\text{{cos}}\\left({halve(min(a,b))}\\pi\\right)\\right) \\\\[1mm]
      &= {c} ( {int(np.round(np.cos((0.5*np.pi*(max(a,b)+2)))))} - {int(np.round(np.cos((0.5*np.pi*min(a,b)))))} ) \\\\[1mm]
      &= {c} ( {int(np.round(np.cos((0.5*np.pi*(max(a,b)+2))))) - int(np.round(np.cos((0.5*np.pi*min(a,b)))))} ) \\\\[1mm]
      &= {c* ( int(np.round(np.cos((0.5*np.pi*(max(a,b)+2))))) - int(np.round(np.cos((0.5*np.pi*min(a,b))))))}
    \\end{{align}}"""))

    display(Markdown("<hr>"))

    a, b, c, d = np.random.randint(3, 9, 4)
    display(Markdown("**(b)**"))
    display(Markdown(f"""\\begin{{align}}
      \\int ({a*b}x^{b-1})({a}x^{b}+{c})^{d}\\ dx
      &= \\int u^{d}\\ du \\qquad \\text{{via $u$-substitutie met}}\\quad u ={a}x^{b}+{c};\\ du = {a*b}x^{b-1} \\\\[1mm]
      &= \\frac{{1}}{{{d+1}}} u^{d+1} + C\\\\[1mm]
      &= \\frac{{1}}{{{d+1}}} ({a}x^{b}+{c})^{d+1} + C
    \\end{{align}}"""))

    display(Markdown("<hr>"))

    a, b, c = np.random.randint(2, 7, 3)
    display(Markdown("**(c)**"))
    display(Markdown(f"""\\begin{{align}}
      \\int ({a}x^{b})\\text{{log}}_{c}(x)\\ dx
      &= \\int \\frac{{({a}x^{b})\\text{{ln}}(x)}}{{\\text{{ln}}({c})}}\\ dx \\\\[1mm]
      &= \\frac{{{a}}}{{\\text{{ln}}({c})}}\\ \\int \\text{{ln}}(x)\\ x^{b} \\ dx \\\\[1mm]
      &= \\frac{{{a}}}{{\\text{{ln}}({c})}}\\ \\left(\\int u\\ dv\\right)  \\
         \\text{{via partiële integratie met}}\\quad u = \\text{{ln}}(x);\\  du = \\frac{{1}}{{x}}\\ dx;\\
                                                    dv = x^{{{b}}}\\ dx;\\ v = \\frac{{x^{b+1}}}{{{b+1}}} \\\\[1mm]
      &= \\frac{{{a}}}{{\\text{{ln}}({c})}}\\ \\left(uv - \\int v\\ du\\right) \\\\[1mm]
      &= \\frac{{{a}}}{{\\text{{ln}}({c})}}\\ \\left(\\text{{ln}}(x)\\frac{{x^{b+1}}}{{{b+1}}}
                                                    - \\int \\frac{{x^{b+1}}}{{{b+1}}}\\ \\frac{{1}}{{x}}\\ dx\\right) \\\\[1mm]
      &= \\frac{{{a}}}{{\\text{{ln}}({c})}}\\ \\left(\\frac{{x^{b+1}\\text{{ln}}(x)}}{{{b+1}}}
                                                    - \\frac{{1}}{{{b+1}}}\\int x^{b}\\ dx \\right) \\\\[1mm]
      &= \\frac{{{a}}}{{\\text{{ln}}({c})}}\\ \\left(\\frac{{x^{b+1}\\text{{ln}}(x)}}{{{b+1}}}
                                                    - \\frac{{1}}{{{b+1}}}\\cdot \\frac{{x^{b+1}}}{{{b+1}}} + C \\right) \\\\[1mm]
      &= \\frac{{{a}}}{{\\text{{ln}}({c})}}\\ \\left(\\frac{{{b+1}x^{b+1}\\text{{ln}}(x)}}{{{(b+1)**2}}}
                                                    - \\frac{{x^{b+1}-x^{b+1}}}{{{(b+1)**2}}} + C \\right) \\\\[1mm]
      &= \\frac{{{a}}}{{\\text{{ln}}({c})}}\\ \\left(\\frac{{x^{b+1}\\ ({b+1}\\ \\text{{ln}}(x)-1)}}{{{(b+1)**2}}} + C \\right) \\\\[1mm]
      &= \\frac{{{a}x^{b+1}\\ ({b+1}\\ \\text{{ln}}(x)-1)}}{{{(b+1)**2}\\ \\text{{ln}}({c})}} + C \\\\[1mm]
    \\end{{align}}"""))

    display(Markdown("<hr>"))


def dif_eq():
    # pylint: disable=C0103
    RNG().set(11)

    a, b, c, d, e, f, g, h = np.random.randint(2, 9, 8)
    b = int(b/2)
    c = c*d
    d = 3*e*f
    e = 2*(g-4)
    f = h*2 -  1
    deriv = f"f^\\prime(x) = {a*b}x^{b-1}+{c}e^x"
    val = f"f({e}) = {a*(e**b)-d}+{c}e^{{{e}}}"
    form = f"f(x) = {a}x^{b} + {c}e^x-{d}"
    ant = f"{a*(f**b)-d}+{c}e^{{{f}}}"
    display(Markdown(f"Vind $f({f})$ gegeven de volgende afgeleidde en waarde:\n\n$${deriv},\\quad {val}$$"))

    display(Markdown("<hr>"))

    display(Markdown(f"""\\begin{{align}} f(x)
      &= \\int f ^\\prime(x)\\ dx \\\\[1mm]
      &= \\int {a*b}x^{b-1}+{c}e^x \\ dx  \\\\[1mm]
      &= {a}x^{b}+{c}e^x + C
    \\end{{align}}"""))

    display(Markdown("<hr>"))

    display(Markdown(f"""\\begin{{align}}
      f({e})                     &= {a}({e})^{b}+{c}e^{{{e}}} + C \\\\[1mm]
      {a*(e**b)-d}+{c}e^{{{e}}}  &= {a}({e})^{b}+{c}e^{{{e}}} + C \\\\[1mm]
      {a*(e**b)-d}               &= {a}({e**b})+ C \\\\[1mm]
      {a*(e**b)-d}               &= {a*e**b}+ C \\\\[1mm]
      C                          &= {-d} \\\\[1mm]
    \\end{{align}}"""))

    display(Markdown("<hr>"))

    display(Markdown(f"""\\begin{{align}}
      f(x)   &= {a}x^{b}+{c}e^x - {d} \\\\[1mm]
      f({f}) &= {a}({f})^{b}+{c}e^{{{f}}} - {d} \\\\[1mm]
             &= {a}({f**b})+{c}e^{{{f}}} - {d} \\\\[1mm]
             &= {a*f**b}+{c}e^{{{f}}} - {d} \\\\[1mm]
             &= {a*f**b-d}+{c}e^{{{f}}} \\\\[1mm]
    \\end{{align}}"""))

    display(Markdown("<hr>"))

    display(Markdown(f"**Antwoord:** ${ant}$, met ${form}$."))
