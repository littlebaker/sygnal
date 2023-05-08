
import jax 
import jax.numpy as jnp
import numpy as np



def correlate(x: jnp.ndarray, y: jnp.ndarray, mode='circulate', axis=-1):
    r"""Calculate the correlation between two signals along specific axis.
    ```math
        
    ```

    Args:
        x (jnp.ndarray): signal 1
        y (jnp.ndarray): signal 2
        mode (str, optional): ['circulate', 'full', 'valid', 'same']. Defaults to 'circulate'.
            in 'circulate' mode, the circulate correlation will be calculated using fft.
        axis (int, optional): calculation axis. Defaults to -1.

    Returns:
        jnp.ndarray: The correlation between two signals
    """
    assert x.ndim <= 2 and x.ndim == y.ndim, "Only support 1D or 2D arrays"
    ndim = x.ndim
    if mode == 'circulate':
        assert x.shape == y.shape, "Signals must share same shape"
        return jnp.fft.ifft(
            jnp.conj(jnp.fft.fft(x, axis=axis)) * jnp.fft.fft(y, axis=axis),
            axis=axis
        )
        
    if ndim == 1:
        return jnp.correlate(x, y, mode=mode)
    else:
        func = lambda x, y: jnp.correlate(x, y, mode=mode)
        # +1 to make that axis's meaning in other modes is same as ciculate mode.
        return jax.vmap(func, ((axis + 1)%ndim , (axis + 1)%ndim))(x, y)

