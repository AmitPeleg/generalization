import pathlib
import platform

# Get the name of the PC
HOST_NAME = platform.node()


class CustomPath(pathlib.Path):
    """A custom path class that can be formatted to display as a hyperlink in terminal."""

    # This is a hack to inherit pathlib.Path and initialize the _flavour property.
    # https://stackoverflow.com/questions/61689391/error-with-simple-subclassing-of-pathlib-path-no-flavour-attribute
    # noinspection PyProtectedMember
    # noinspection PyUnresolvedReferences
    _flavour = type(pathlib.Path())._flavour

    def __format__(self, format_spec):
        if format_spec == '':
            return HOST_NAME + ": " + str(self)
        elif format_spec == 'link':
            return HOST_NAME + ": " + _create_hyperlink(self.absolute())
        elif format_spec == 'exists':
            if self.exists():
                return _create_hyperlink(self.absolute())
            else:
                return _create_hyperlink(self.absolute()) + ' does not exist. \nParent directory: ' + _create_hyperlink(
                    self.parent.absolute())
        else:
            raise NotImplementedError(f"Format spec {format_spec} is not implemented")

    def __iadd__(self, other: str):
        return CustomPath(str(self) + other)

    def __add__(self, other: str):
        return CustomPath(str(self) + other)


def _create_hyperlink(text: [str, pathlib.Path]):
    if isinstance(text, pathlib.Path):
        text = str(text)
    return f'file:///' + text.replace('\\', '/')


ROOT_DIR = CustomPath(__file__).parent.absolute()

print(f"Running on host {HOST_NAME} from root dir {ROOT_DIR:link}")


DATA_DIR = ROOT_DIR / 'data'
DATA_DIR.mkdir(exist_ok=True)

FIGURES_DIR = ROOT_DIR / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

CONFIG_ROOT = ROOT_DIR / 'configs'
CONFIG_ROOT.mkdir(exist_ok=True)

OUTPUT_ROOT = ROOT_DIR / 'output'
OUTPUT_ROOT.mkdir(exist_ok=True)

suffix_with_acc = 'model_stats.db'
suffix_summary = 'model_stats_summary.db'

table_dict = {
    'mnist': 'mnist',
    'cifar': 'cifar10',
    'cifar10': 'cifar10',
}
