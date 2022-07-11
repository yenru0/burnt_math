import random

import numpy as np
import sympy as sp
from sympy.abc import x, y, z
import jinja2

a = sp.symbols("a:100")


def render_from_template(directory, template_name, **kwargs):
    loader = jinja2.FileSystemLoader(directory)
    env = jinja2.Environment(loader=loader)
    template = env.get_template(template_name)
    return template.render(**kwargs)

class ProblemTemplate:
    def __init__(self, expr, parameter_count, parameter_range, tags: list = [], question_message = r"{0}의 값은?",):
        self.expr = expr
        self.tags = tags
        self.parameter_count = parameter_count
        self.parameter_range = parameter_range

class TemplateManager:
    def __init__(self,):
        self.templates = dict()
    
    def register_template(self, name, template: ProblemTemplate):
        self.templates[name] = template

    def generate_problem(self, name):
        tmp: ProblemTemplate = self.templates[name]
        params = [(a[i], random.choice(tmp.parameter_range[i])) for i in range(tmp.parameter_count)]
        n_exp = tmp.expr.subs(params)
        return n_exp

    def randomly_generate_problems(self, k: int):
        keys = random.choices(list(self.templates.keys()), k=k)
        return [self.generate_problem(key) for key in keys]

TMX = TemplateManager()

TMX.register_template(
    "삼각적분-사인",
    ProblemTemplate(
        sp.Integral(sp.sin(a[0]*x), (x, a[1]/a[0], a[2]/a[0])),
        3, [[1, 2, 3],
        [0, sp.pi/6, sp.pi/4, sp.pi/3, sp.pi/2, 2*sp.pi/3, 3*sp.pi/4, 5*sp.pi/6, sp.pi],
        [0, sp.pi/6, sp.pi/4, sp.pi/3, sp.pi/2, 2*sp.pi/3, 3*sp.pi/4, 5*sp.pi/6, sp.pi],
        ],
    )
)

TMX.register_template(
    "삼각적분-코사인",
    ProblemTemplate(
        sp.Integral(sp.cos(a[0]*x), (x, a[1]/a[0], a[2]/a[0])),
        3, [[1, 2, 3],
        [0, sp.pi/6, sp.pi/4, sp.pi/3, sp.pi/2, 2*sp.pi/3, 3*sp.pi/4, 5*sp.pi/6, sp.pi],
        [0, sp.pi/6, sp.pi/4, sp.pi/3, sp.pi/2, 2*sp.pi/3, 3*sp.pi/4, 5*sp.pi/6, sp.pi],
        ],
    )
)

TMX.register_template(
    "삼각적분-탄젠트",
    ProblemTemplate(
        sp.Integral(sp.tan(a[0]*x), (x, a[1]/a[0], a[2]/a[0])),
        3, [[1, 2, 3],
        [-sp.pi/3, -sp.pi/4, -sp.pi/6, 0 , sp.pi/6, sp.pi/4, sp.pi/3],
        [-sp.pi/3, -sp.pi/4, -sp.pi/6, 0 , sp.pi/6, sp.pi/4, sp.pi/3],
        ],
    )
)

TMX.register_template(
    "삼각적분-코탄젠트(pos)",
    ProblemTemplate(
        sp.Integral(sp.cot(a[0]*x), (x, a[1]/a[0], a[2]/a[0])),
        3, [[1, 2, 3],
        [sp.pi/6, sp.pi/4, sp.pi/3],
        [sp.pi/6, sp.pi/4, sp.pi/3],
        ],
    )
)

TMX.register_template(
    "삼각적분-시컨트스퀘어",
    ProblemTemplate(
        sp.Integral(sp.sec(a[0]*x)**2, (x, a[1]/a[0], a[2]/a[0])),
        3, [[1, 2, 3],
        [-sp.pi/3, -sp.pi/4, -sp.pi/6, 0 , sp.pi/6, sp.pi/4, sp.pi/3],
        [-sp.pi/3, -sp.pi/4, -sp.pi/6, 0 , sp.pi/6, sp.pi/4, sp.pi/3],
        ],
    )
)

TMX.register_template(
    "삼각적분-코시컨트스퀘어",
    ProblemTemplate(
        sp.Integral(sp.csc(a[0]*x)**2, (x, a[1]/a[0], a[2]/a[0])),
        3, [[1, 2, 3],
        [sp.pi/6, sp.pi/4, sp.pi/3, sp.pi/2, 2*sp.pi/3, 3*sp.pi/4, 5*sp.pi/6],
        [sp.pi/6, sp.pi/4, sp.pi/3, sp.pi/2, 2*sp.pi/3, 3*sp.pi/4, 5*sp.pi/6],
        ],
    )
)
TMX.register_template(
    "삼각치환-"
)
TMX.register_template(
    "삼각치환-탄젠트",
    ProblemTemplate(
    sp.Integral(a[0]/(x**2+a[1]**2)**(a[2]), (x, 0, a[1])),
    3, [[1, 2, 3], [1, 2, 3,], [sp.Rational(1, 2), 1]],
    tags= ["integral", "trigonometric substitution"]),
)


class RandomProblemGenerator:
    def __init__(self, tm):
        self.template_manager: TemplateManager = tm
        self.problems = []
        self.solved = []
    def generate(self, n):
        self.problems = self.template_manager.randomly_generate_problems(n)
        self.solved = [problem.doit() for problem in self.problems]
    
    def print(self):
        for i, problem in enumerate(self.problems):
            print(i, problem, self.solved[i])
    
    def pprint(self):
        for i, problem in enumerate(self.problems):
            sp.pprint(i)
            sp.pprint(problem)
            sp.pprint(self.solved[i])

    def make_document(self,):
        html = "<ol>\n"

        latex_probs = [f"\\(\displaystyle{sp.latex(prob, mode='plain')}\\)" for prob in self.problems]
        latex_solved = [f"\\(\displaystyle{sp.latex(sol, mode='plain')}\\)" for sol in self.solved]
        p = render_from_template(".", "html_paper_template.html", problems = latex_probs)
        s = render_from_template(".", "html_solution_template.html", solved = latex_solved)

        with open("paper.html", mode='w', encoding='utf-8') as f:
            f.write(p)

        with open("solution.html", mode='w', encoding='utf-8') as f:
            f.write(s)

if __name__ == '__main__':
    sp.init_printing(use_latex='mathjax')
    rpg = RandomProblemGenerator(TMX)
    rpg.generate(50)
    rpg.make_document()

    
    
    
