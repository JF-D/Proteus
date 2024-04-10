from collections import OrderedDict
from proteus.ir.tensor import Tensor, Parameter
from proteus.ir.graph_builder import enter_name_scope, exit_name_scope


def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


MODULES = ()


def register_module(name, class_type):
    global MODULES
    MODULES = tuple(list(MODULES) + [class_type])


class Module:
    def __init__(self):
        super().__init__()
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._modules = OrderedDict()
        self.training = False
        self.full_name = ''

    def train(self, mode=True):
        self.training = mode
        for child in self.children():
            child.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def add_module(self, name, module):
        if not isinstance(module, Module) and module is not None:
            raise TypeError("{} is not a Module subclass".format(type(module)))
        if not isinstance(name, str):
            raise TypeError("module name should be a string. Got {}".format(
                type(name)))
        if hasattr(self, name) and name not in self._modules:
            raise KeyError("attribute '{}' already exists".format(name))
        if '.' in name:
            raise KeyError("module name can't contain \".\"")
        if name == '':
            raise KeyError("module name can't be empty string \"\"")
        self._modules[name] = module
        if module is not None:
            module._set_full_name(self.full_name, name)

    def parameters(self):
        """Returns an iterator over module parameters.
        """

        for name, param in self.named_parameters():
            yield param

    def named_parameters(self, param_memo=None, prefix=''):
        """Returns an iterator over module parameters, yielding both the
        name of the parameter as well as the parameter itself.
        """

        if param_memo is None:
            param_memo = set()
        for name, param in self._parameters.items():
            if param is not None and param not in param_memo:
                param_memo.add(param)
                yield prefix + ('.' if prefix else '') + name, param
        for mname, module in self.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, param in module.named_parameters(param_memo,
                                                       submodule_prefix):
                yield name, param

    def modules(self):
        for name, module in self.named_modules():
            yield module

    def named_modules(self, memo=None, prefix=''):
        """Returns an iterator over all child modules in this module and
        all sub-modules.
        """

        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self

        for name, module in self._modules.items():
            if module is None:
                continue
            submodule_prefix = prefix + ('.' if prefix else '') + name
            for m in module.named_modules(memo, submodule_prefix):
                yield m

    def children(self):
        for name, module in self.named_children():
            yield module

    def named_children(self):
        children_memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in children_memo:
                children_memo.add(module)
                yield name, module

    def forward(self):
        raise NotImplementedError

    def register_parameter(self, name, param):
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "Cannot assign parameter before Module.__init__() call")
        if hasattr(self, name) and name not in self._parameters:
            raise KeyError("attribute '{}' already exists".format(name))

        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError(
                "Parameter '{}' type error"
                "(parrots.nn.Parameter or None required)".format(name))
        else:
            self._parameters[name] = param

    def __call__(self, *args, **kwargs):
        global MODULES
        if not isinstance(self, MODULES):
            module_name = self.full_name.split('.')[-1]
            enter_name_scope(module_name, self._get_name(), args)
        result = self.forward(*args, **kwargs)
        if not isinstance(self, MODULES):
            exit_name_scope(result)
        return result

    def __getattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __setattr__(self, name, value):
        def remove_from(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]

        params = self.__dict__.get('_parameters')
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call")
            remove_from(self.__dict__, self._buffers, self._modules)
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError("cannot assign '{}' as parameter '{}' "
                                "(nn.Parameter or None expected)".format(
                                    type(value), name))
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get('_modules')
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call")
                remove_from(self.__dict__, self._parameters, self._buffers)
                modules[name] = value
                value._set_full_name(self.full_name, name)
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError("cannot assign '{}' as child module '{}' "
                                    "(nn.Module or None expected)".format(
                                        type(value), name))
                modules[name] = value
            else:
                buffers = self.__dict__.get('_buffers')
                if buffers is not None and name in buffers:
                    if value is not None and not isinstance(value, Tensor):
                        raise TypeError("cannot assign '{}' as buffer '{}' "
                                        "(Tensor or None expected)".format(
                                            type(value), name))
                    buffers[name] = value
                else:
                    object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
        elif name in self._modules:
            del self._modules[name]
        else:
            object.__delattr__(self, name)

    def _set_full_name(self, prefix, name):
        if prefix:
            self.full_name = prefix + '.' + name
        else:
            self.full_name = name
        for key, value in self._modules.items():
            if value is not None:
                value._set_full_name(self.full_name, key)

    def _get_name(self):
        return self.__class__.__name__

    def extra_repr(self):
        return ''

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str
