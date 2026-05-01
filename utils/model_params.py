
import copy


def add_parameters(model1, model2):
    result = copy.deepcopy(model1)
    for r, param1, param2 in zip(
        result.parameters(), model1.parameters(), model2.parameters()
    ):
        r.data = param1.data.clone() + param2.data.clone()
    return result


def subtract_parameters(model1, model2):
    result = copy.deepcopy(model1)
    for r, param1, param2 in zip(
        result.parameters(), model1.parameters(), model2.parameters()
    ):
        r.data = param1.data.clone() - param2.data.clone()
    return result


def divide_constant(model, divisor):
    result = copy.deepcopy(model)
    for r, param in zip(result.parameters(), model.parameters()):
        r.data = param.data.clone() / divisor
    return result


def divide_parameters(model1, model2):
    result = copy.deepcopy(model1)
    for r, param1, param2 in zip(
        result.parameters(), model1.parameters(), model2.parameters()
    ):
        r.data = param1.data.clone() / param2.data.clone()
    return result


def multiply_parameters(model1, model2):
    result = copy.deepcopy(model1)
    for r, param1, param2 in zip(
        result.parameters(), model1.parameters(), model2.parameters()
    ):
        r.data = param1.data.clone() * param2.data.clone()
    return result


def multiply_constant(model, constant):
    result = copy.deepcopy(model)
    for r, param in zip(result.parameters(), model.parameters()):
        r.data = param.data.clone() * constant
    return result


def copy_parameters(f, to):
    for f_param, to_param in zip(f.parameters(), to.parameters()):
        to_param.data = f_param.data.clone()
