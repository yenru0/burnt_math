import random

import numpy as np
import sympy as sp
import jinja2

sp.Rational

x, y, z = sp.symbols("x, y, z")
a_ = sp.symbols("a:100")


def render_from_template(directory, template_name, **kwargs):
    loader = jinja2.FileSystemLoader(directory)
    env = jinja2.Environment(loader=loader)
    template = env.get_template(template_name)
    return template.render(**kwargs)


# TEMPLATE CLASS

class ExactProblem:
    def __init__(self, exprs, solution, question):
        self.exprs = exprs
        self.solution = solution
        self.question = question


class ProblemTemplate:
    def __init__(self, exprs: list,
                parameter_count, parameter_range, #a[0:100]
                solver  = lambda x: x[0].doit(),
                question_maker = lambda x: f"\\(\displaystyle{sp.latex(x[0])}\\)",
                solution_maker = lambda s: f"\\(\displaystyle{sp.latex(s)}\\)",
                props: dict = {},
                tags: list = [],):
        if isinstance(exprs, list):
            self.exprs = exprs
        else:
            self.exprs = [exprs]
        self.solver = solver
        self.parameter_count = parameter_count
        self.parameter_range = parameter_range
        self.question_maker = question_maker
        self.solution_maker = solution_maker
        self.props = props # properties

        self.tags = tags

    def make_exact(self, rng=True):
        params = [(a_[i], random.choice(self.parameter_range[i])) for i in range(self.parameter_count)]
        subs_exprs = [expr.subs(params) for expr in self.exprs]
        return ExactProblem(subs_exprs, self.solution_maker(self.solver(subs_exprs)), self.question_maker(subs_exprs))

class TemplateManager:
    def __init__(self,):
        self.templates = dict()
    
    def register_template(self, name, template: ProblemTemplate):
        self.templates[name] = template

    def generate_problem(self, name):
        tmp: ProblemTemplate = self.templates[name]
        return tmp.make_exact()

    def randomly_generate_problems(self, k: int):
        keys = random.choices(list(self.templates.keys()), k=k)
        return [self.generate_problem(key) for key in keys]

TMX = TemplateManager()

TMX.register_template(
    "단항분수확장적분",
    ProblemTemplate(
        sp.Integral(1/x**a_[0], (x, 1, a_[1])),
        2, [[-sp.Rational(1, 2), sp.Rational(1, 2), -sp.Rational(1, 3), 1, 2, 3,], [2, 3, 4],]
    )
)

TMX.register_template(
    "로그적분-이차다항식치환적분",
    ProblemTemplate(
        sp.Integral((2*x+a_[0])/(x**2+a_[0]*x+a_[1]), (x, 1, 2)),
        2, [[0, 1, 2, 3], [0, 1, 2, 3]]
    )
)



TMX.register_template(
    "삼각적분-사인",
    ProblemTemplate(
        sp.Integral(sp.sin(a_[0]*x), (x, a_[1]/a_[0], a_[2]/a_[0])),
        3, [[1, 2, 3],
        [0, sp.pi/6, sp.pi/4, sp.pi/3, sp.pi/2, 2*sp.pi/3, 3*sp.pi/4, 5*sp.pi/6, sp.pi],
        [0, sp.pi/6, sp.pi/4, sp.pi/3, sp.pi/2, 2*sp.pi/3, 3*sp.pi/4, 5*sp.pi/6, sp.pi],
        ],
    )
)

TMX.register_template(
    "삼각적분-코사인",
    ProblemTemplate(
        sp.Integral(sp.cos(a_[0]*x), (x, a_[1]/a_[0], a_[2]/a_[0])),
        3, [[1, 2, 3],
        [0, sp.pi/6, sp.pi/4, sp.pi/3, sp.pi/2, 2*sp.pi/3, 3*sp.pi/4, 5*sp.pi/6, sp.pi],
        [0, sp.pi/6, sp.pi/4, sp.pi/3, sp.pi/2, 2*sp.pi/3, 3*sp.pi/4, 5*sp.pi/6, sp.pi],
        ],
    )
)

TMX.register_template(
    "삼각적분-탄젠트",
    ProblemTemplate(
        sp.Integral(sp.tan(a_[0]*x), (x, a_[1]/a_[0], a_[2]/a_[0])),
        3, [[1, 2, 3],
        [-sp.pi/3, -sp.pi/4, -sp.pi/6, 0 , sp.pi/6, sp.pi/4, sp.pi/3],
        [-sp.pi/3, -sp.pi/4, -sp.pi/6, 0 , sp.pi/6, sp.pi/4, sp.pi/3],
        ],
    )
)

TMX.register_template(
    "삼각적분-코탄젠트(pos)",
    ProblemTemplate(
        sp.Integral(sp.cot(a_[0]*x), (x, a_[1]/a_[0], a_[2]/a_[0])),
        3, [[1, 2, 3],
        [sp.pi/6, sp.pi/4, sp.pi/3],
        [sp.pi/6, sp.pi/4, sp.pi/3],
        ],
    )
)

TMX.register_template(
    "삼각적분-시컨트스퀘어",
    ProblemTemplate(
        sp.Integral(sp.sec(a_[0]*x)**2, (x, a_[1]/a_[0], a_[2]/a_[0])),
        3, [[1, 2, 3],
        [-sp.pi/3, -sp.pi/4, -sp.pi/6, 0 , sp.pi/6, sp.pi/4, sp.pi/3],
        [-sp.pi/3, -sp.pi/4, -sp.pi/6, 0 , sp.pi/6, sp.pi/4, sp.pi/3],
        ],
    )
)

TMX.register_template(
    "삼각적분-코시컨트스퀘어",
    ProblemTemplate(
        sp.Integral(sp.csc(a_[0]*x)**2, (x, a_[1]/a_[0], a_[2]/a_[0])),
        3, [[1, 2, 3],
        [sp.pi/6, sp.pi/4, sp.pi/3, sp.pi/2, 2*sp.pi/3, 3*sp.pi/4, 5*sp.pi/6],
        [sp.pi/6, sp.pi/4, sp.pi/3, sp.pi/2, 2*sp.pi/3, 3*sp.pi/4, 5*sp.pi/6],
        ],
    )
)

TMX.register_template(
    "삼각치환-일반삼각",
    ProblemTemplate(
        sp.Integral((a_[0]*(a_[1]**2-x**2))**(sp.Rational(1, 2)), (x, 0, a_[1]* a_[2])),
        3, [[1, 2, 3], [1, 2, 3], [sp.Rational(1, 2), sp.sqrt(2)/2, sp.sqrt(3)/2]]
    )
)

TMX.register_template(
    "삼각치환-탄젠트",
    ProblemTemplate(
    sp.Integral(a_[0]/(x**2+a_[1]**2)**(a_[2]), (x, 0, a_[1])),
    3, [[1, 2, 3], [1, 2, 3,], [sp.Rational(1, 2), 1]],)
)


class RandomProblemGenerator:
    def __init__(self, tm):
        self.template_manager: TemplateManager = tm
        self.problems = []

    def generate(self, n):
        self.problems = self.template_manager.randomly_generate_problems(n) # exactProblems 
    
    def print(self):
        for i, problem in enumerate(self.problems):
            print(i, problem.question, problem.solution)
    
    def pprint(self):
        for i, problem in enumerate(self.problems):
            sp.pprint(i)
            sp.pprint(problem.question)
            sp.pprint(problem.solution)

    def make_document(self,):

        latex_probs = [prob.question for prob in self.problems]
        latex_solved = [prob.solution for prob in self.problems]
        p = render_from_template(".", "html_paper_template.html", problems = latex_probs)
        s = render_from_template(".", "html_solution_template.html", solved = latex_solved)

        with open("paper.html", mode='w', encoding='utf-8') as f:
            f.write(p)

        with open("solution.html", mode='w', encoding='utf-8') as f:
            f.write(s)

if __name__ == '__main__':
    #sp.init_printing(use_latex='mathjax')
    rpg = RandomProblemGenerator(TMX)
    rpg.generate(50)
    rpg.make_document()

    
    
    
