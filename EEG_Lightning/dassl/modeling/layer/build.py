from dassl.utils import Registry, check_availability

LAYER_REGISTRY = Registry('LAYER')


def build_layer(name, verbose=True, **kwargs):
    avai_layers = LAYER_REGISTRY.registered_names()
    check_availability(name, avai_layers)
    if verbose:
        print('Layer: {}'.format(name))
    return LAYER_REGISTRY.get(name)(**kwargs)
