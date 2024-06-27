from pylint.checkers import BaseChecker
import astroid


class TfRandomSetSeedChecker(BaseChecker):
    __implements__ = BaseChecker

    name = 'tf-random-uncontrolled'
    priority = -1
    msgs = {
        'CR003': (
            'tf.random.set_seed(...) not found in the code',
            'tf-random-uncontrolled',
            'Used when tf.random.set_seed(...) is not found in the code.',
        ),
    }
    options = ()

    def __init__(self, linter=None):
        super().__init__(linter)
        self.found_tf_random_set_seed = False
        self.dummy_node = None

    def visit_call(self, node):
        if self.dummy_node is None:
            self.dummy_node = node

        if (isinstance(node.func, astroid.Attribute) and
            node.func.attrname == 'set_seed' and
            isinstance(node.func.expr, astroid.Attribute) and
            node.func.expr.attrname == 'random' and
            isinstance(node.func.expr.expr, astroid.Name) and
            node.func.expr.expr.name == 'tf'):
            self.found_tf_random_set_seed = True

    def close(self):
        if not self.found_tf_random_set_seed:
            self.add_message('tf-random-uncontrolled', node=self.dummy_node)


def register(linter):
    linter.register_checker(TfRandomSetSeedChecker(linter))
