import jax 
import jax.numpy as jnp
import numpy as np

from .correlate import correlate


def moment_x2_g(y: jnp.ndarray, n: jnp.ndarray, axis=0):
    r"""Calculate the second moment of signal X. **Gaussian noise hypothesis**
    ```math
        E(X^2)=E(\hat Z^2)-E(\hat{N_1}^2)
    ```

    Args:
        y (jnp.ndarray): signal with noise
        n (jnp.ndarray): noise signal
        axis (int, optional): average axis. Defaults to 0.

    Returns:
        jnp.ndarray: The first moment of signal X
    """
    assert y.shape == n.shape, "Signals must share same shape"
    
    yh = y - jnp.mean(n, axis=axis, keepdims=True)
    nh = n - jnp.mean(n, axis=axis, keepdims=True)
    
    
    return jnp.mean(yh**2, axis=axis) - jnp.mean(nh**2, axis=axis)


def moment_xy_g(z1: jnp.ndarray, z2: jnp.ndarray, n1: jnp.ndarray, n2: jnp.ndarray,axis=0):
    """Calculate the mix second order moment of XY between two signals. **Gaussian noise hypothesis**
    ```math
        E(XY)=E(\hat Z_1 \hat Z_2)-E(\hat N_1 \hat N_2)
    ```

    Args:
        z1 (jnp.ndarray): signal 1 with noise
        z2 (jnp.ndarray): signal 2 with noise
        n1 (jnp.ndarray): noise signal 1
        n2 (jnp.ndarray): noise signal 2
        axis (int, optional): average axis. Defaults to 0.
        noise_correction (bool, optional): [description]. Defaults to True.

    Returns:
        jnp.ndarray: The mix second order moment of XY
    """
    assert z1.shape == z2.shape == n1.shape == n2.shape, "Signals must share same shape"
    
    z1h = z1 - jnp.mean(n1, axis=axis, keepdims=True)
    z2h = z2 - jnp.mean(n2, axis=axis, keepdims=True)
    n1h = n1 - jnp.mean(n1, axis=axis, keepdims=True)
    n2h = n2 - jnp.mean(n2, axis=axis, keepdims=True)
    
    return jnp.mean(z1h * z2h, axis=axis) - jnp.mean(n1h * n2h, axis=axis)


def moment_x2y_g(z1: jnp.ndarray, z2: jnp.ndarray, n1: jnp.ndarray, n2: jnp.ndarray,axis=0):
    """Calculate the mix second order moment of X^2Y between two signals. **Gaussian noise hypothesis**
    ```math
        E(X^2Y)=E(\hat Z_1^2 \hat Z_2)-2E(X)E(\hat N_1\hat N_2)-E(Y)E(\hat N_1^2)
    ```

    Args:
        z1 (jnp.ndarray): signal 1 with noise
        z2 (jnp.ndarray): signal 2 with noise
        n1 (jnp.ndarray): noise signal 1
        n2 (jnp.ndarray): noise signal 2
        axis (int, optional): average axis. Defaults to 0.

    Returns:
        jnp.ndarray: The mix third order moment of X^2Y
    """
    assert z1.shape == z2.shape == n1.shape == n2.shape, "Signals must share same shape"
    
    z1h = z1 - jnp.mean(n1, axis=axis, keepdims=True)
    z2h = z2 - jnp.mean(n2, axis=axis, keepdims=True)
    n1h = n1 - jnp.mean(n1, axis=axis, keepdims=True)
    n2h = n2 - jnp.mean(n2, axis=axis, keepdims=True)
    
    Ex = jnp.mean(z1h, axis=axis) - jnp.mean(n1h, axis=axis)
    Ey = jnp.mean(z2h, axis=axis) - jnp.mean(n2h, axis=axis)
    
    return jnp.mean(z1h**2 * z2h, axis=axis) - 2 * Ex * jnp.mean(n1h * n2h, axis=axis) - Ey* jnp.mean(n1h**2, axis=axis)


def moment_x2y2_g(z1: jnp.ndarray, z2: jnp.ndarray, n1: jnp.ndarray, n2: jnp.ndarray, axis=0):
    """Calculate the mix second order moment of X^2Y^2 between two signals. **Gaussian noise hypothesis**
    ```math
        E(X^2Y^2)=E(\hat Z_1^2 \hat Z_2^2) - 
            E(\hat N_1^2 \hat N_2^2)-E(X^2)E(\hat N_2^2)-E(Y^2)E(\hat N_1^2)-4E(XY)E(\hat N_1\hat N_2)
    ```

    Args:
        z1 (jnp.ndarray): signal 1 with noise
        z2 (jnp.ndarray): signal 2 with noise
        n1 (jnp.ndarray): noise signal 1
        n2 (jnp.ndarray): noise signal 2
        axis (int, optional): average axis. Defaults to 0.

    Returns:
        jnp.ndarray: The mix fourth order moment of X^2Y^2
    """
    assert z1.shape == z2.shape == n1.shape == n2.shape, "Signals must share same shape"
    
    z1h = z1 - jnp.mean(n1, axis=axis, keepdims=True)
    z2h = z2 - jnp.mean(n2, axis=axis, keepdims=True)
    n1h = n1 - jnp.mean(n1, axis=axis, keepdims=True)
    n2h = n2 - jnp.mean(n2, axis=axis, keepdims=True)
    
    Ex2 = moment_x2_g(z1, n1, axis=axis)
    Ey2 = moment_x2_g(z2, n2, axis=axis)
    Exy = moment_xy_g(z1, z2, n1, n2, axis=axis)
    
    
    return jnp.mean(z1h**2 * z2h**2, axis=axis) - jnp.mean(n1h**2 * n2h**2, axis=axis) - Ex2 * jnp.mean(n2h**2, axis=axis) - Ey2 * jnp.mean(n1h**2, axis=axis) - 4 * Exy * jnp.mean(n1h * n2h, axis=axis)


def correlate_xy_g(z1: jnp.ndarray, z2: jnp.ndarray, n1: jnp.ndarray, n2: jnp.ndarray, mode='circulate', ):
    """calculate the :math:`X\star Y` subjects to 
    ```math
        Z1 = X + N1
        Z2 = Y + N2
        E[X\star Y]  = E[\hat Z_1\star \hat Z_2] - E[\hat N_1\star \hat N_2]
    ```

    Args:
        z1 (jnp.ndarray): signal 1 with noise
        z2 (jnp.ndarray): signal 2 with noise
        n1 (jnp.ndarray): noise signal 1
        n2 (jnp.ndarray): noise signal 2
        mode (str, optional): ['circulate', 'full', 'valid', 'same']. Defaults to 'circulate'.
        axis (int, optional): calculation axis. Defaults to 0.
    """
    
    z1h = z1 - jnp.mean(n1, axis=0, keepdims=True)
    z2h = z2 - jnp.mean(n2, axis=0, keepdims=True)
    n1h = n1 - jnp.mean(n1, axis=0, keepdims=True)
    n2h = n2 - jnp.mean(n2, axis=0, keepdims=True)
    
    return correlate(z1h, z2h, mode=mode) - correlate(n1h, n2h, mode=mode)


def correlate_x2y2_gn_nc(z1: jnp.ndarray, z2: jnp.ndarray, n1: jnp.ndarray, n2: jnp.ndarray, mode='circulate'):
    """calculate the :math:`X^2 \star Y^2`, noise gaussian distributed, not correlated which subjects to 
    ```math
        Z1 = X + N1
        Z2 = Y + N2
        E[X^2 \star Y^2]=E[\hat Z_1^2\star \hat Z_2^2]-E[\hat N_1^2\star \hat N_2^2]-E[X^2]\star E[\hat N_2^2]-E[Y^2]\star E[\hat N_1^2]
        
        ```

    Args:
        z1 (jnp.ndarray): signal 1 with noise
        z2 (jnp.ndarray): signal 2 with noise
        n1 (jnp.ndarray): noise signal 1
        n2 (jnp.ndarray): noise signal 2
        mode (str, optional): ['circulate', 'full', 'valid', 'same']. Defaults to 'circulate'.
        axis (int, optional): calculation axis. Defaults to 0.
    """
    
    z1h = z1 - jnp.mean(n1, axis=0, keepdims=True)
    z2h = z2 - jnp.mean(n2, axis=0, keepdims=True)
    n1h = n1 - jnp.mean(n1, axis=0, keepdims=True)
    n2h = n2 - jnp.mean(n2, axis=0, keepdims=True)
    
    x2 = z1h **2 - n1h**2
    y2 = z2h **2 - n2h**2

    correlate_xy = correlate(z1h, z2h, mode=mode) - correlate(n1h, n2h, mode=mode)
    correlate_n1n2 = correlate(n1h, n2h, mode=mode)
    print(jnp.mean(correlate_xy * correlate_n1n2, axis=0))
    
    return correlate(z1h**2, z2h**2, mode=mode) - correlate(n1h**2, n2h**2, mode=mode)\
        - correlate(x2, n2h**2, mode=mode)\
        - correlate(y2, n1h**2, mode=mode)
        

def correlate_xabs2yabs2_gn_nc(z1: jnp.ndarray, z2: jnp.ndarray, n1: jnp.ndarray, n2: jnp.ndarray, mode='circulate'):
    """calculate the :math:`X^*X \star Y^*Y`, noise gaussian distributed, not correlated which subjects to 
    ```math
        Z1 = X + N1
        Z2 = Y + N2
        E[X^*X \star Y^*Y]=E[\hat Z_1^*\hat Z_1\star \hat Z_2^*\hat Z_2]
            -E[\hat N_1^*\hat N_1\star \hat N_2^*\hat N_2]-E[X^*X]\star E[\hat N_2^*\hat N_2]
            -E[Y^*Y]\star E[\hat N_1^*\hat N_1]        
    ```

    Args:
        z1 (jnp.ndarray): signal 1 with noise
        z2 (jnp.ndarray): signal 2 with noise
        n1 (jnp.ndarray): noise signal 1
        n2 (jnp.ndarray): noise signal 2
        mode (str, optional): ['circulate', 'full', 'valid', 'same']. Defaults to 'circulate'.
        axis (int, optional): calculation axis. Defaults to 0.
        
    """
    
    z1h = z1 - jnp.mean(n1, axis=0, keepdims=True)
    z2h = z2 - jnp.mean(n2, axis=0, keepdims=True)
    n1h = n1 - jnp.mean(n1, axis=0, keepdims=True)
    n2h = n2 - jnp.mean(n2, axis=0, keepdims=True)

    xsx = z1h * jnp.conj(z1h) - n1h * jnp.conj(n1h)
    ysy = z2h * jnp.conj(z2h) - n2h * jnp.conj(n2h)
    
    return correlate(jnp.conj(z1h) * z1h, jnp.conj(z2h) * z2h, mode=mode) - correlate(jnp.conj(n1h) * n1h, jnp.conj(n2h) * n2h, mode=mode)\
        - correlate(xsx, jnp.conj(n2h)*n2h, mode=mode)\
        - correlate(ysy, jnp.conj(n1h)*n1h, mode=mode)
        