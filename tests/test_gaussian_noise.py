

from sygnal import gaussian_noise as gn
import jax.numpy as jnp
import jax
from sygnal import correlate

def test_correlate_x2y2_g():
    # jax.config.update("jax_enable_x64", True)
    num_samples = 10000
    amp = 1
    mode = 'circulate'
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    x =  jax.random.bernoulli(subkey, 0.5, (1000,)) + 1
    y =  1 - x + 2
    
    key, subkey = jax.random.split(key)
    n1 = jax.random.normal(subkey, (num_samples, 1000)) * amp
    key, subkey = jax.random.split(key)
    n2 = jax.random.normal(subkey, (num_samples, 1000)) * amp
    
    z1 = x + n1
    z2 = y + n2
    
    key, subkey = jax.random.split(key)
    n3 = jax.random.normal(subkey, (num_samples, 1000)) * amp
    key, subkey = jax.random.split(key)
    n4 = jax.random.normal(subkey, (num_samples, 1000)) * amp
    print(gn.correlate_x2y2_gn_nc(z1, z2, n3, n4, mode=mode).shape)
    print((x**2).shape)
    a = jnp.real(jnp.mean(gn.correlate_x2y2_gn_nc(z1, z2, n3, n4, mode=mode), axis=0))
    b = jnp.real(correlate.correlate(x**2, y**2, mode=mode))
    print(a)
    print(b)
    print(max(abs(a-b)), jnp.argmax(abs(a-b)), a[int(jnp.argmax(abs(a-b)))], b[int(jnp.argmax(abs(a-b)))])
    
    assert jnp.allclose(a, b, rtol=0.5, atol=0.1)
    
    
def test_correlate_xabs2yabs2_g():
    num_samples = 10000
    amp = 1
    mode = 'circulate'
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    x =  jax.random.bernoulli(subkey, 0.5, (1000,)) + 1 + 3j
    y =  1 - x + 2
    
    key, subkey = jax.random.split(key)
    n1 = jax.random.normal(subkey, (num_samples, 1000)) * amp
    key, subkey = jax.random.split(key)
    n2 = jax.random.normal(subkey, (num_samples, 1000)) * amp
    
    z1 = x + n1
    z2 = y + n2
    
    key, subkey = jax.random.split(key)
    n3 = jax.random.normal(subkey, (num_samples, 1000)) * amp
    key, subkey = jax.random.split(key)
    n4 = jax.random.normal(subkey, (num_samples, 1000)) * amp
    print(gn.correlate_xabs2yabs2_gn_nc(z1, z2, n3, n4, mode=mode).shape)
    print((x**2).shape)
    a = jnp.real(jnp.mean(gn.correlate_xabs2yabs2_gn_nc(z1, z2, n3, n4, mode=mode), axis=0))
    b = jnp.real(correlate.correlate(abs(x)**2, abs(y)**2, mode=mode))
    print(a)
    print(b)
    print(max(abs(a-b)), jnp.argmax(abs(a-b)), a[int(jnp.argmax(abs(a-b)))], b[int(jnp.argmax(abs(a-b)))])
    
    assert jnp.allclose(a, b, rtol=0.5, atol=0.1)
    
    
if __name__ == "__main__":
    test_correlate_x2y2_g()
    test_correlate_xabs2yabs2_g()