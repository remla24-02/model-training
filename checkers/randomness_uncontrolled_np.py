from pylint.checkers import BaseChecker
import astroid


class NpRandomSeedChecker(BaseChecker):
    __implements__ = BaseChecker

    name = 'np-random-uncontrolled'
    priority = -1
    msgs = {
        'CR002': (
            'np.random.seed(...) not found in the code',
            'np-random-uncontrolled',
            'Used when np.random.seed(...) is not found in the code.',
        ),
    }
    options = ()

    def __init__(self, linter=None):
        super().__init__(linter)
        self.found_np_random_seed = False
        self.dummy_node = None

    def visit_call(self, node):
        if self.dummy_node is None:
            self.dummy_node = node

        if (isinstance(node.func, astroid.Attribute) and
            node.func.attrname == 'seed' and
            isinstance(node.func.expr, astroid.Attribute) and
            node.func.expr.attrname == 'random' and
            isinstance(node.func.expr.expr, astroid.Name) and
            node.func.expr.expr.name == 'np'):
            self.found_np_random_seed = True

    def close(self):
        if not self.found_np_random_seed:
            self.add_message('np-random-uncontrolled', node=self.dummy_node)


def register(linter):
    linter.register_checker(NpRandomSeedChecker(linter))
