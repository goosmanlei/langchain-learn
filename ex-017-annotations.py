def show_annotations(func):
    print(f'func.__name__: {func.__name__}')
    print(f'func.__annotations__: {func.__annotations__}')
    print('')

@show_annotations
def multiply(x: int, y: int) -> int:
    return x * y
