from pylint.checkers import BaseChecker
import astroid


class RandomSeedChecker(BaseChecker):
    __implements__ = BaseChecker

    name = 'random-uncontrolled'
    priority = -1
    msgs = {
        'CR001': (
            'random.seed(...) not found in the code',
            'random-uncontrolled',
            'Used when random.seed(...) is not found in the code.',
        ),
    }
    options = ()

    def __init__(self, linter=None):
        super().__init__(linter)
        self.found_random_seed = False
        self.dummy_node = None

    def visit_call(self, node):
        if self.dummy_node is None:
            self.dummy_node = node

        if (isinstance(node.func, astroid.Attribute) and
            node.func.attrname == 'seed' and
            isinstance(node.func.expr, astroid.Name) and
            node.func.expr.name == 'random'):
            self.found_random_seed = True

    def close(self):
        if not self.found_random_seed:
            self.add_message('random-uncontrolled', node=self.dummy_node)


def register(linter):
    linter.register_checker(RandomSeedChecker(linter))
