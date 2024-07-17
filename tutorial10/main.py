from engine import Value

def tests():
    # Run some tests on the value class

    x = Value(1)
    y = x + 2 - 3
    y.backward()
    assert x.grad == 1, "Error in test 1"

    x = Value(1)
    y = 4*x/2
    y.backward()
    assert x.grad == 2, "Error in test 2"

    x = Value(2)
    y = x**2
    y.backward()
    assert x.grad == 2*2, "Error in test 3"

    x = Value(0)
    y = x.cos()
    y.backward()
    assert x.grad == 0, "Error in test 4"

    x = Value(0)
    y = x.sin()
    y.backward()
    assert x.grad == 1, "Error in test 5"

    x = Value(0)
    y = x*x.cos() + x**2 + 3
    y.backward()
    assert x.grad == 1, "Error in test 5"
