

from sygnal import gaussian_noise as gn
import jax.numpy as jnp
import jax
from sygnal import correlate

def test_correlate_x2y2_g():
    mode = 'circulate'
    key = jax.random.PRNGKey(0)
    x =  jax.random.bernoulli(key, 0.5, (1000,)) + 1
    y =  1 - x + 2
    
    n1 = jax.random.normal(key, (10000, 1000)) 
    n2 = jax.random.normal(key, (10000, 1000))
    
    z1 = x + n1
    z2 = y + n2
    
    n3 = jax.random.normal(key, (10000, 1000))
    n4 = jax.random.normal(key, (10000, 1000))
    print(gn.correlate_x2y2_g(z1, z2, n3, n4, mode=mode).shape)
    print((x**2).shape)
    a = jnp.mean(gn.correlate_x2y2_g(z1, z2, n3, n4, mode=mode), axis=0)
    b = correlate.correlate(x**2, y**2, mode=mode)
    print(a)
    print(b)
    print(max(abs(a-b)), jnp.argmax(abs(a-b)), a[int(jnp.argmax(abs(a-b)))], b[int(jnp.argmax(abs(a-b)))])
    assert jnp.allclose(a, b, rtol=0.5, atol=0.1)
    
    
def test_correlate_xabs2yabs2_g():
    return
    key = jax.random.PRNGKey(0)
    x =  jax.random.uniform(key, (1000,))+2
    y =  jax.random.uniform(key, (1000,)) + 3
    
    n1 = jax.random.normal(key, (1000, 1000)) 
    n2 = jax.random.normal(key, (1000, 1000))
    
    z1 = x + n1
    z2 = y + n2
    
    n3 = jax.random.normal(key, (1000, 1000))
    n4 = jax.random.normal(key, (1000, 1000))
    print(jnp.mean(gn.correlate_xabs2yabs2_g(z1, z2, n3, n4), axis=0))
    print(correlate.correlate(x**2, y**2))
    assert jnp.allclose(jnp.mean(gn.correlate_xabs2yabs2_g(z1, z2, n3, n4), axis=0), correlate.correlate(x**2, y**2), rtol=0.05)