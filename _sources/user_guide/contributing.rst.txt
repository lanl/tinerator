.. _contributing:

Contributing
============

Pull Request Process
--------------------

Pull requests for missing features, bug fixes, and documentation updates are
more than welcome. To contribute, please submit a pull request at

    https://github.com/lanl/tinerator

To ensure the best chance of acceptance for your pull request, please ensure that:

1. All tests pass, and you have added test case(s) for your developed functionality
2. Your code is well documented, formatted with the [Black](https://github.com/psf/black) code formatter, and, if applicable, that the documentation has been updated to reflect any new functionality
3. That the purpose and scope of your contributions are well explained in the PR

Need an Issue?
--------------

Visit the Issues page of TINerator and search for tag 'Good First Issue'.
These are great issues for someone unfamiliar with the codebase to work on.

### Function Docstrings

Documentation is generated with [Sphinx](https://www.sphinx-doc.org/en/master/), and docstrings must follow the [Google Style for Python Docstrings](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html).

Further, all functions should be annotated with [type hints](https://docs.python.org/3/library/typing.html).

As an example of proper formatting,


.. code:: python

   def example_function(param1: int, param2: str = None, *args, **kwargs) -> bool:
       """
       This is an example of proper docstring formatting.

       Args:
           param1 (int): The first parameter.
           param2 (:obj:`str`, optional): The second parameter. Defaults to None.
               Second line of description should be indented.
           *args: Variable length argument list.
           **kwargs: Arbitrary keyword arguments.

       Returns:
           bool: True if successful, False otherwise.
   
       Raises:
           ValueError: if `param2` is not of type `str`.
   
       Examples:
           >>> result = example_function(10, param2 = '10')
           >>> print(result)
           True
       """
       if not isinstance(param2, str):
           raise ValueError(f'Invalid type for param2: {type(param2)}')
       return param1 == int(param2)

