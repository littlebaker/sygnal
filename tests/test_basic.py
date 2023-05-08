
from sygnal import basic
import jax.numpy as jnp

def test_moment_x():
    n = jnp.array([[1, 2, 3], [4, 5, 6]])
    y = jnp.array([[2, 3, 4], [5, 95, 7]])
    assert jnp.allclose(basic.moment_x(y, n), jnp.array([1, 45.5, 1]))
    
    
def test_moment_x2():
    return
    n = jnp.array([[1, 2, 3], [4, 5, 6]])
    y = jnp.array([[2, 3, 4], [5, 95, 7]])
    assert jnp.allclose(basic.moment_x2(y, n), jnp.array([1, 45.5, 1]))