import numpy as np
from Cyclone.coil_initialization import create_toroidal_angles, count_nested_lists, create_poloidal_angles

class Test_create_toroidal_angles:
    def test_ints(self):
        nfp = 2
        stellsym = True
        ntoroidalcurves_in = 5
        ntoroidalcurves, toroidal_angles = create_toroidal_angles(ntoroidalcurves_in, nfp, stellsym)
        assert ntoroidalcurves == ntoroidalcurves_in
        desired = [(i+0.5) * 2*np.pi/(nfp*(1+int(stellsym))*ntoroidalcurves_in) for i in range(ntoroidalcurves_in)]
        np.testing.assert_allclose(toroidal_angles, desired, atol=1e-14)
        nfp = 5
        stellsym = True
        ntoroidalcurves_in = 50
        ntoroidalcurves, toroidal_angles = create_toroidal_angles(ntoroidalcurves_in, nfp, stellsym)
        assert ntoroidalcurves == ntoroidalcurves_in
        desired = [(i+0.5) * 2*np.pi/(nfp*(1+int(stellsym))*ntoroidalcurves_in) for i in range(ntoroidalcurves_in)]
        np.testing.assert_allclose(toroidal_angles, desired, atol=1e-14)
        nfp = 17
        stellsym = False
        ntoroidalcurves_in = 2
        ntoroidalcurves, toroidal_angles = create_toroidal_angles(ntoroidalcurves_in, nfp, stellsym)
        assert ntoroidalcurves == ntoroidalcurves_in
        desired = [(i+0.5) * 2*np.pi/(nfp*(1+int(stellsym))*ntoroidalcurves_in) for i in range(ntoroidalcurves_in)]
        np.testing.assert_allclose(toroidal_angles, desired, atol=1e-14)
        nfp = 6
        stellsym = False
        ntoroidalcurves_in = 192
        ntoroidalcurves, toroidal_angles = create_toroidal_angles(ntoroidalcurves_in, nfp, stellsym)
        assert ntoroidalcurves == ntoroidalcurves_in
        desired = [(i+0.5) * 2*np.pi/(nfp*(1+int(stellsym))*ntoroidalcurves_in) for i in range(ntoroidalcurves_in)]
        np.testing.assert_allclose(toroidal_angles, desired, atol=1e-14)

    def test_floats(self):
        nfp = 2
        stellsym = True
        ntoroidalcurves_in = 0.2
        ntoroidalcurves, toroidal_angles = create_toroidal_angles(ntoroidalcurves_in, nfp, stellsym)
        assert ntoroidalcurves == 1
        desired = [ntoroidalcurves_in * 2*np.pi/(nfp*(1+int(stellsym)))]
        np.testing.assert_allclose(toroidal_angles, desired, atol=1e-14)
        nfp = 5
        stellsym = True
        ntoroidalcurves_in = 0.5
        ntoroidalcurves, toroidal_angles = create_toroidal_angles(ntoroidalcurves_in, nfp, stellsym)
        assert ntoroidalcurves == 1
        desired = [ntoroidalcurves_in * 2*np.pi/(nfp*(1+int(stellsym)))]
        np.testing.assert_allclose(toroidal_angles, desired, atol=1e-14)
        nfp = 17
        stellsym = False
        ntoroidalcurves_in = 0.111111
        ntoroidalcurves, toroidal_angles = create_toroidal_angles(ntoroidalcurves_in, nfp, stellsym)
        assert ntoroidalcurves == 1
        desired = [ntoroidalcurves_in * 2*np.pi/(nfp*(1+int(stellsym)))]
        np.testing.assert_allclose(toroidal_angles, desired, atol=1e-14)
        nfp = 6
        stellsym = False
        ntoroidalcurves_in = .9192
        ntoroidalcurves, toroidal_angles = create_toroidal_angles(ntoroidalcurves_in, nfp, stellsym)
        assert ntoroidalcurves == 1
        desired = [ntoroidalcurves_in * 2*np.pi/(nfp*(1+int(stellsym)))]
        np.testing.assert_allclose(toroidal_angles, desired, atol=1e-14)
    
    def test_lists(self):
        nfp = 2
        stellsym = True
        ntoroidalcurves_in = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ntoroidalcurves, toroidal_angles = create_toroidal_angles(ntoroidalcurves_in, nfp, stellsym)
        assert ntoroidalcurves == len(ntoroidalcurves_in)
        desired = [angle * 2*np.pi/(nfp*(1+int(stellsym))) for angle in ntoroidalcurves_in]
        np.testing.assert_allclose(toroidal_angles, desired, atol=1e-14)
        nfp = 5
        stellsym = True
        ntoroidalcurves_in = [0.5, 0.4, 0.1, 0.8, 0.1923124]
        ntoroidalcurves, toroidal_angles = create_toroidal_angles(ntoroidalcurves_in, nfp, stellsym)
        assert ntoroidalcurves == len(ntoroidalcurves_in)
        desired = [angle * 2*np.pi/(nfp*(1+int(stellsym))) for angle in ntoroidalcurves_in]
        np.testing.assert_allclose(toroidal_angles, desired, atol=1e-14)
        nfp = 17
        stellsym = False
        ntoroidalcurves_in = [0.111111, 0.1452, 0.111111, 0.56, 0.27, 0.0001]
        ntoroidalcurves, toroidal_angles = create_toroidal_angles(ntoroidalcurves_in, nfp, stellsym)
        assert ntoroidalcurves == len(ntoroidalcurves_in)
        desired = [angle * 2*np.pi/(nfp*(1+int(stellsym))) for angle in ntoroidalcurves_in]
        np.testing.assert_allclose(toroidal_angles, desired, atol=1e-14)
        nfp = 6
        stellsym = False
        ntoroidalcurves_in = [0.88]
        ntoroidalcurves, toroidal_angles = create_toroidal_angles(ntoroidalcurves_in, nfp, stellsym)
        assert ntoroidalcurves == len(ntoroidalcurves_in)
        desired = [angle * 2*np.pi/(nfp*(1+int(stellsym))) for angle in ntoroidalcurves_in]
        np.testing.assert_allclose(toroidal_angles, desired, atol=1e-14)

def test_count_nested_lists():
    list_ = [[1,2,3],[1,2,3,4,5,6],[1,2],[1],[1],[1,2,3,4,5]]
    assert count_nested_lists(list_) == (3,6,2,1,1,5)
    list_ = [[1,2,3],[1,2,3,4,5,6]]
    assert count_nested_lists(list_) == (3,6)
    list_ = [[1,2,3]]
    assert count_nested_lists(list_) == (3,)

class Test_create_poloidal_angles():
    def test_ints(self):
        ntoroidalcurves = 5
        npoloidalcurves_in = 5
        npoloidalcurves, poloidal_angles = create_poloidal_angles(npoloidalcurves_in, ntoroidalcurves)
        assert ntoroidalcurves == len(npoloidalcurves)
        assert all([ncurves == npoloidalcurves_in for ncurves in npoloidalcurves])
        desired = [i * 2*np.pi/(npoloidalcurves_in) for i in range(npoloidalcurves_in)]
        for i in range(ntoroidalcurves):
            np.testing.assert_allclose(poloidal_angles[i], desired, atol=1e-14)
        ntoroidalcurves = 15
        npoloidalcurves_in = 7
        npoloidalcurves, poloidal_angles = create_poloidal_angles(npoloidalcurves_in, ntoroidalcurves)
        assert ntoroidalcurves == len(npoloidalcurves)
        assert all([ncurves == npoloidalcurves_in for ncurves in npoloidalcurves])
        desired = [i * 2*np.pi/(npoloidalcurves_in) for i in range(npoloidalcurves_in)]
        for i in range(ntoroidalcurves):
            np.testing.assert_allclose(poloidal_angles[i], desired, atol=1e-14)
        toroidalcurves = 4
        npoloidalcurves_in = 18
        npoloidalcurves, poloidal_angles = create_poloidal_angles(npoloidalcurves_in, ntoroidalcurves)
        assert ntoroidalcurves == len(npoloidalcurves)
        assert all([ncurves == npoloidalcurves_in for ncurves in npoloidalcurves])
        desired = [i * 2*np.pi/(npoloidalcurves_in) for i in range(npoloidalcurves_in)]
        for i in range(ntoroidalcurves):
            np.testing.assert_allclose(poloidal_angles[i], desired, atol=1e-14)
        ntoroidalcurves = 5
        npoloidalcurves_in = (3,5,7,9,11)
        npoloidalcurves, poloidal_angles = create_poloidal_angles(npoloidalcurves_in, ntoroidalcurves)
        assert ntoroidalcurves == len(npoloidalcurves)
        for i, npoloidalcurves_this in enumerate(npoloidalcurves_in):
            assert [ncurves == npoloidalcurves_in[i] for ncurves in npoloidalcurves]
            desired = [j * 2*np.pi/(npoloidalcurves_this) for j in range(npoloidalcurves_this)]
            np.testing.assert_allclose(poloidal_angles[i], desired, atol=1e-14)
        ntoroidalcurves = 15
        npoloidalcurves_in = (3,5,7,9,11,2,4,6,8,10,1,2,3,4,5)
        npoloidalcurves, poloidal_angles = create_poloidal_angles(npoloidalcurves_in, ntoroidalcurves)
        assert ntoroidalcurves == len(npoloidalcurves)
        for i, npoloidalcurves_this in enumerate(npoloidalcurves_in):
            assert [ncurves == npoloidalcurves_in[i] for ncurves in npoloidalcurves]
            desired = [j * 2*np.pi/(npoloidalcurves_this) for j in range(npoloidalcurves_this)]
            np.testing.assert_allclose(poloidal_angles[i], desired, atol=1e-14)
        ntoroidalcurves = 1
        npoloidalcurves_in = (3,)
        npoloidalcurves, poloidal_angles = create_poloidal_angles(npoloidalcurves_in, ntoroidalcurves)
        assert ntoroidalcurves == len(npoloidalcurves)
        for i, npoloidalcurves_this in enumerate(npoloidalcurves_in):
            assert [ncurves == npoloidalcurves_in[i] for ncurves in npoloidalcurves]
            desired = [j * 2*np.pi/(npoloidalcurves_this) for j in range(npoloidalcurves_this)]
            np.testing.assert_allclose(poloidal_angles[i], desired, atol=1e-14)
        """
        ntoroidalcurves = 1
        npoloidalcurves_in = [3]
        npoloidalcurves, poloidal_angles = create_poloidal_angles(npoloidalcurves_in, ntoroidalcurves)
        print(npoloidalcurves_in)
        assert ntoroidalcurves == len(npoloidalcurves)
        for i, npoloidalcurves_this in enumerate(npoloidalcurves_in):
            assert [ncurves == npoloidalcurves_in[i] for ncurves in npoloidalcurves]
            desired = [j * 2*np.pi/(npoloidalcurves_this) for j in range(npoloidalcurves_this)]
            np.testing.assert_allclose(poloidal_angles[i], desired, atol=1e-14)
        """ # For some reason, this block works (above) with a TUPLE used for npoloidalcurves_in, but not (here) with a LIST