from .register import register_kernel


try:
    from .kernel_jigsaw import register as register_jigsaw
    register_kernel(register_jigsaw())
except Exception as e:
    print(e)