from Cyclone.axis_functions import r_axis, z_axis, r_axis_prime, z_axis_prime, axis_del_phi
import numpy as np

def test_r_axis():
    angles = np.linspace(0, 2*np.pi, 500)
    nfp = 3
    r_axis_cc = [1, 0.1, -0.2]
    desired = np.sum([rax * np.cos(nfp * i * angles) for i, rax in enumerate(r_axis_cc)], axis = 0)
    actual = [r_axis(r_axis_cc, nfp, angle) for angle in angles]
    np.testing.assert_allclose(actual, desired, atol=1e-14)
    nfp = 2
    r_axis_cc = [1, 0.1, -0.2]
    r_axis_cs = [0.2, -0.4, 1.2]
    desired = np.sum([rax * np.cos(nfp * i * angles) for i, rax in enumerate(r_axis_cc)], axis = 0)
    desired += np.sum([rax * np.sin(nfp * i * angles) for i, rax in enumerate(r_axis_cs)], axis = 0)
    actual = [r_axis(r_axis_cc, nfp, angle, raxiscs=r_axis_cs) for angle in angles]
    np.testing.assert_allclose(actual, desired, atol=1e-14)
    nfp = 17
    r_axis_cc = [13, -0.1, -0.2111, -8, 3]
    r_axis_cs = [0.2, -0.4, 1.2, -0.2, 0.6, 0, 8.1]
    desired = np.sum([rax * np.cos(nfp * i * angles) for i, rax in enumerate(r_axis_cc)], axis = 0)
    desired += np.sum([rax * np.sin(nfp * i * angles) for i, rax in enumerate(r_axis_cs)], axis = 0)
    actual = [r_axis(r_axis_cc, nfp, angle, raxiscs=r_axis_cs) for angle in angles]
    np.testing.assert_allclose(actual, desired, atol=1e-14)

def test_z_axis():
    angles = np.linspace(0, 2*np.pi, 500)
    nfp = 3
    z_axis_cs = [1, 0.1, -0.2]
    desired = np.sum([zax * np.sin(nfp * i * angles) for i, zax in enumerate(z_axis_cs)], axis = 0)
    actual = [z_axis(z_axis_cs, nfp, angle) for angle in angles]
    np.testing.assert_allclose(actual, desired, atol=1e-14)
    nfp = 2
    z_axis_cs = [1, 0.1, -0.2]
    z_axis_cc = [0.2, -0.4, 1.2]
    desired = np.sum([zax * np.sin(nfp * i * angles) for i, zax in enumerate(z_axis_cs)], axis = 0)
    desired += np.sum([zax * np.cos(nfp * i * angles) for i, zax in enumerate(z_axis_cc)], axis = 0)
    actual = [z_axis(z_axis_cs, nfp, angle, zaxiscc=z_axis_cc) for angle in angles]
    np.testing.assert_allclose(actual, desired, atol=1e-14)
    nfp = 17
    z_axis_cs = [13, -0.1, -0.2111, -8, 3]
    z_axis_cc = [0.2, -0.4, 1.2, -0.2, 0.6, 0, 8.1]
    desired = np.sum([zax * np.sin(nfp * i * angles) for i, zax in enumerate(z_axis_cs)], axis = 0)
    desired += np.sum([zax * np.cos(nfp * i * angles) for i, zax in enumerate(z_axis_cc)], axis = 0)
    actual = [z_axis(z_axis_cs, nfp, angle, zaxiscc=z_axis_cc) for angle in angles]
    np.testing.assert_allclose(actual, desired, atol=1e-14)

def test_r_axis_prime():
    angles = np.linspace(0, 2*np.pi, 500)
    nfp = 3
    r_axis_cc = [1, 0.1, -0.2]
    desired = np.sum([-rax * nfp * i * np.sin(nfp * i * angles) for i, rax in enumerate(r_axis_cc)], axis = 0)
    actual = [r_axis_prime(r_axis_cc, nfp, angle) for angle in angles]
    np.testing.assert_allclose(actual, desired, atol=1e-14)
    nfp = 2
    r_axis_cc = [1, 0.1, -0.2]
    r_axis_cs = [0.2, -0.4, 1.2]
    desired = np.sum([-rax * nfp * i * np.sin(nfp * i * angles) for i, rax in enumerate(r_axis_cc)], axis = 0)
    desired += np.sum([rax * nfp * i * np.cos(nfp * i * angles) for i, rax in enumerate(r_axis_cs)], axis = 0)
    actual = [r_axis_prime(r_axis_cc, nfp, angle, raxiscs=r_axis_cs) for angle in angles]
    np.testing.assert_allclose(actual, desired, atol=1e-14)
    nfp = 17
    r_axis_cc = [13, -0.1, -0.2111, -8, 3]
    r_axis_cs = [0.2, -0.4, 1.2, -0.2, 0.6, 0, 8.1]
    desired = np.sum([-rax * nfp * i * np.sin(nfp * i * angles) for i, rax in enumerate(r_axis_cc)], axis = 0)
    desired += np.sum([rax * nfp * i * np.cos(nfp * i * angles) for i, rax in enumerate(r_axis_cs)], axis = 0)
    actual = [r_axis_prime(r_axis_cc, nfp, angle, raxiscs=r_axis_cs) for angle in angles]
    np.testing.assert_allclose(actual, desired, atol=1e-14)

def test_z_axis_prime():
    angles = np.linspace(0, 2*np.pi, 500)
    nfp = 3
    z_axis_cs = [1, 0.1, -0.2]
    desired = np.sum([zax * nfp * i * np.cos(nfp * i * angles) for i, zax in enumerate(z_axis_cs)], axis = 0)
    actual = [z_axis_prime(z_axis_cs, nfp, angle) for angle in angles]
    np.testing.assert_allclose(actual, desired, atol=1e-14)
    nfp = 2
    z_axis_cs = [1, 0.1, -0.2]
    z_axis_cc = [0.2, -0.4, 1.2]
    desired = np.sum([zax * nfp * i * np.cos(nfp * i * angles) for i, zax in enumerate(z_axis_cs)], axis = 0)
    desired += np.sum([-zax * nfp * i * np.sin(nfp * i * angles) for i, zax in enumerate(z_axis_cc)], axis = 0)
    actual = [z_axis_prime(z_axis_cs, nfp, angle, zaxiscc=z_axis_cc) for angle in angles]
    np.testing.assert_allclose(actual, desired, atol=1e-14)
    nfp = 17
    z_axis_cs = [13, -0.1, -0.2111, -8, 3]
    z_axis_cc = [0.2, -0.4, 1.2, -0.2, 0.6, 0, 8.1]
    desired = np.sum([zax * nfp * i * np.cos(nfp * i * angles) for i, zax in enumerate(z_axis_cs)], axis = 0)
    desired += np.sum([-zax * nfp * i * np.sin(nfp * i * angles) for i, zax in enumerate(z_axis_cc)], axis = 0)
    actual = [z_axis_prime(z_axis_cs, nfp, angle, zaxiscc=z_axis_cc) for angle in angles]
    np.testing.assert_allclose(actual, desired, atol=1e-14)

def test_axis_del_phi():
    angles = np.linspace(0, 2*np.pi, 500)
    nfp = 3
    r_axis_cc = [1, 0.1, -0.2]
    z_axis_cs = [1, 0.1, -0.2]
    r_ax = np.sum([rax * np.cos(nfp * i * angles) for i, rax in enumerate(r_axis_cc)], axis = 0)
    r_ax_p = np.sum([-rax * nfp * i * np.sin(nfp * i * angles) for i, rax in enumerate(r_axis_cc)], axis = 0)
    z_ax_p = np.sum([zax * nfp * i * np.cos(nfp * i * angles) for i, zax in enumerate(z_axis_cs)], axis = 0)
    X = r_ax_p * np.cos(angles) - r_ax * np.sin(angles)
    Y = r_ax_p * np.sin(angles) + r_ax * np.cos(angles)
    Z = z_ax_p
    XYZ = np.concatenate([X[:, None], Y[:, None], Z[:, None]], axis=1)
    actual = [axis_del_phi(r_axis_cc, z_axis_cs, nfp, angle) for angle in angles]
    np.testing.assert_allclose(actual, XYZ, atol=1e-14)
    nfp = 2
    r_axis_cc = [1, 0.1, -0.2]
    r_axis_cs = [0.2, -0.4, 1.2]
    z_axis_cs = [1, 0.1, -0.2]
    z_axis_cc = [0.2, -0.4, 1.2]
    r_ax = np.sum([rax * np.cos(nfp * i * angles) for i, rax in enumerate(r_axis_cc)], axis = 0)
    r_ax += np.sum([rax * np.sin(nfp * i * angles) for i, rax in enumerate(r_axis_cs)], axis = 0)
    r_ax_p = np.sum([-rax * nfp * i * np.sin(nfp * i * angles) for i, rax in enumerate(r_axis_cc)], axis = 0)
    r_ax_p += np.sum([rax * nfp * i * np.cos(nfp * i * angles) for i, rax in enumerate(r_axis_cs)], axis = 0)
    z_ax_p = np.sum([zax * nfp * i * np.cos(nfp * i * angles) for i, zax in enumerate(z_axis_cs)], axis = 0)
    z_ax_p += np.sum([-zax * nfp * i * np.sin(nfp * i * angles) for i, zax in enumerate(z_axis_cc)], axis = 0)
    X = r_ax_p * np.cos(angles) - r_ax * np.sin(angles)
    Y = r_ax_p * np.sin(angles) + r_ax * np.cos(angles)
    Z = z_ax_p
    XYZ = np.concatenate([X[:, None], Y[:, None], Z[:, None]], axis=1)
    actual = [axis_del_phi(r_axis_cc, z_axis_cs, nfp, angle, raxiscs=r_axis_cs, zaxiscc=z_axis_cc) for angle in angles]
    np.testing.assert_allclose(actual, XYZ, atol=1e-14)
    nfp = 17
    r_axis_cc = [13, -0.1, -0.2111, -8, 3]
    r_axis_cs = [0.2, -0.4, 1.2, -0.2, 0.6, 0, 8.1]
    z_axis_cs = [13, -0.1, -0.2111, -8, 3]
    z_axis_cc = [0.2, -0.4, 1.2, -0.2, 0.6, 0, 8.1]
    r_ax = np.sum([rax * np.cos(nfp * i * angles) for i, rax in enumerate(r_axis_cc)], axis = 0)
    r_ax += np.sum([rax * np.sin(nfp * i * angles) for i, rax in enumerate(r_axis_cs)], axis = 0)
    r_ax_p = np.sum([-rax * nfp * i * np.sin(nfp * i * angles) for i, rax in enumerate(r_axis_cc)], axis = 0)
    r_ax_p += np.sum([rax * nfp * i * np.cos(nfp * i * angles) for i, rax in enumerate(r_axis_cs)], axis = 0)
    z_ax_p = np.sum([zax * nfp * i * np.cos(nfp * i * angles) for i, zax in enumerate(z_axis_cs)], axis = 0)
    z_ax_p += np.sum([-zax * nfp * i * np.sin(nfp * i * angles) for i, zax in enumerate(z_axis_cc)], axis = 0)
    X = r_ax_p * np.cos(angles) - r_ax * np.sin(angles)
    Y = r_ax_p * np.sin(angles) + r_ax * np.cos(angles)
    Z = z_ax_p
    XYZ = np.concatenate([X[:, None], Y[:, None], Z[:, None]], axis=1)
    actual = [axis_del_phi(r_axis_cc, z_axis_cs, nfp, angle, raxiscs=r_axis_cs, zaxiscc=z_axis_cc) for angle in angles]
    np.testing.assert_allclose(actual, XYZ, atol=1e-14)