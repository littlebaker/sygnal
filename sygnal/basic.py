
import jax 
import jax.numpy as jnp
import numpy as np


def moment_x(y: jnp.ndarray, n: jnp.ndarray, axis=0):
    r"""Calculate the first moment of the difference between two signals
    ```math
        Y = X + N
        E(X)=E(Y)-E(N)
    ```

    Args:
        y (jnp.ndarray): signal with noise
        n (jnp.ndarray): noise signal
        axis (int, optional): average axis. Defaults to 0.

    Returns:
        jnp.ndarray: The first moment of signal X
    """
    assert y.shape == n.shape, "Signals must share same shape"
    return jnp.mean(y - n, axis=axis)


def moment_x2(y: jnp.ndarray, n: jnp.ndarray, axis=0):
    r"""Calculate the second order moment of the difference(X) between two signals
    ```math
        Y = X + N
        E(X^2)=E(Y^2)-E(N^2)-2E(N)(E(Y)-E(N))
    ```

    Args:
        y (jnp.ndarray): signal with noise
        n (jnp.ndarray): noise signal
        axis (int, optional): average axis. Defaults to 0.

    Returns:
        jnp.ndarray: The second moment of signal X
    """
    assert y.shape == n.shape, "Signals must share same shape"
    Ex = jnp.mean(y - n, axis=axis)
    
    return jnp.mean(y**2, axis=axis) - jnp.mean(n**2, axis=axis) - 2 * jnp.mean(n, axis=axis) * Ex 


def moment_xy(z1: jnp.ndarray, z2: jnp.ndarray, n1: jnp.ndarray, n2: jnp.ndarray,axis=0, noise_correction=True):
    """Calculate the mix second order moment of XY between two signals
    ```math
        Z1 = X + N1
        Z2 = Y + N2
        E(XY)=E(Z1Z2)-E(N1N2)-E(N1)E(Y)-E(N2)E(X)
    ```
    
    if noise_correction is False, then E(N1N2) = E(N1)E(N2)

    Args:
        z1 (jnp.ndarray): signal 1 with noise
        z2 (jnp.ndarray): signal 2 with noise
        n1 (jnp.ndarray): noise signal 1
        n2 (jnp.ndarray): noise signal 2
        axis (int, optional): average axis. Defaults to 0.
    """
    
    Ex = jnp.mean(z1 - n1, axis=axis)
    Ey = jnp.mean(z2 - n2, axis=axis)
    
    En1n2 = None
    if noise_correction:
        En1n2 = jnp.mean(n1 * n2, axis=axis)
    else:
        En1n2 = jnp.mean(n1, axis=axis) * jnp.mean(n2, axis=axis)
    
    return jnp.mean(z1 * z2, axis=axis) - En1n2 - jnp.mean(n1, axis=axis) * Ey - jnp.mean(n2, axis=axis) * Ex


def moment_x2y(z1: jnp.ndarray, z2: jnp.ndarray, n1: jnp.ndarray, n2: jnp.ndarray,axis=0, noise_correction=True):
    """Calculate the mix second order moment of X^2Y between two signals
    ```math
        Z1 = X + N1
        Z2 = Y + N2
        E(X^2Y)=E(Z_1^2Z_2)-2E(X)E(N_1N_2)-2E(XY)E(N_1)-E(X^2)E(N_2)-E(Y)E(N_1^2)-E(N_1^2N_2)
    ```
    
    if noise_correction is False, then E(N1N2) = E(N1)E(N2)

    Args:
        z1 (jnp.ndarray): signal 1 with noise
        z2 (jnp.ndarray): signal 2 with noise
        n1 (jnp.ndarray): noise signal 1
        n2 (jnp.ndarray): noise signal 2
        axis (int, optional): average axis. Defaults to 0.
    """
    
    Ex = jnp.mean(z1 - n1, axis=axis)
    Ey = jnp.mean(z2 - n2, axis=axis)
    Exy = moment_xy(z1, z2, n1, n2, axis=axis, noise_correction=noise_correction)
    Ex2 = moment_x2(z1, n1, axis=axis)
    Ey2 = moment_x2(z2, n2, axis=axis)
    
    En1n2 = None
    if noise_correction:
        En1n2 = jnp.mean(n1 * n2, axis=axis)
    else:
        En1n2 = jnp.mean(n1, axis=axis) * jnp.mean(n2, axis=axis)
    
    return jnp.mean(z1**2 * z2, axis=axis) - 2 * Ex * En1n2 -\
        2 * Exy * jnp.mean(n1, axis=axis) - Ex2 * jnp.mean(n2, axis=axis) - \
            Ey * jnp.mean(n1**2, axis=axis) - jnp.mean(n1**2 * n2, axis=axis)


def moment_x2y2(z1: jnp.ndarray, z2: jnp.ndarray, n1: jnp.ndarray, n2: jnp.ndarray,axis=0, noise_correction=True):
    """Calculate the mix second order moment of X^2Y^2 between two signals
    ```math
        Z1 = X + N1
        Z2 = Y + N2
        E(X^2Y^2)=E(Z_1^2Z_2^2) - E(N_1^2N_2^2)-2E(X)E(N_1N_2^2)-2E(XY^2)E(N_1)-2E(Y)E(N_1^2N_2)
            -2E(X^2Y)E(N_2)-E(X^2)E(N_2^2)-E(Y^2)E(N_1^2)-4E(XY)E(N_1N_2)
    ```
    
    if noise_correction is False, then E(N1N2) = E(N1)E(N2)

    Args:
        z1 (jnp.ndarray): signal 1 with noise
        z2 (jnp.ndarray): signal 2 with noise
        n1 (jnp.ndarray): noise signal 1
        n2 (jnp.ndarray): noise signal 2
        axis (int, optional): average axis. Defaults to 0.
    """
    
    Ex = jnp.mean(z1 - n1, axis=axis)
    Ey = jnp.mean(z2 - n2, axis=axis)
    Exy = moment_xy(z1, z2, n1, n2, axis=axis, noise_correction=noise_correction)
    Ex2 = moment_x2(z1, n1, axis=axis)
    Ey2 = moment_x2(z2, n2, axis=axis)
    Ex2y = moment_x2y(z1, z2, n1, n2, axis=axis, noise_correction=noise_correction)
    Exy2 = moment_x2y(z2, z1, n2, n1, axis=axis, noise_correction=noise_correction)
    
    En1n2 = None
    if noise_correction:
        En1n2 = jnp.mean(n1 * n2, axis=axis)
    else:
        En1n2 = jnp.mean(n1, axis=axis) * jnp.mean(n2, axis=axis)
    
    return jnp.mean(z1**2 * z2**2, axis=axis) - jnp.mean(n1**2 * n2**2, axis=axis) - \
        2 * Ex * jnp.mean(n1 * n2**2, axis=axis) - 2 * Exy2 * jnp.mean(n1, axis=axis) - \
            2 * Ey * jnp.mean(n1**2 * n2, axis=axis) - 2 * Ex2y * jnp.mean(n2, axis=axis) - \
                Ex2 * jnp.mean(n2**2, axis=axis) - Ey2 * jnp.mean(n1**2, axis=axis) - \
                    4 * Exy * En1n2
