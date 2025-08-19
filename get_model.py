from pbac import PBAC


def get_model(env, args):
    model_name = "pbac"
    return PBAC(env, args)
