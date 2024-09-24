import numpy as np

import simsoptpp as sopp
from simsopt._core.optimizable import Optimizable
from simsopt._core.derivative import derivative_dec

'''
class Max_BdotN(Optimizable):
    def __init__(self, surface, field, target=None, definition="quadratic flux"):
        self.surface = surface
        if target is not None:
            self.target = np.ascontiguousarray(target)
        else:
            self.target = np.zeros(self.surface.normal().shape[:2])
        self.field = field
        xyz = self.surface.gamma()
        self.field.set_points(xyz.reshape((-1, 3)))
        if definition not in ["quadratic flux", "normalized", "local"]:
            raise ValueError("Unrecognized option for 'definition'.")
        self.definition = definition
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[field])
    
    def J(self):
        n = self.surface.normal()
        absn = np.linalg.norm(n, axis=2)
        unitn = n * (1. / absn)[:, :, None]
        Bcoil = self.field.B().reshape(n.shape)
        Bcoil_n = np.sum(Bcoil * unitn, axis=2)
        if self.target is not None:
            B_n = (Bcoil_n - self.target)
        else:
            B_n = Bcoil_n
        
        arg = np.argmax(np.abs(B_n))
        unravel_arg = np.unravel_index(arg, B_n.shape)
        largest_B_n = B_n[unravel_arg]
        
        if self.definition == "quadratic flux":
            J = 0.5 * largest_B_n ** 2 * absn[unravel_arg]

        elif self.definition == "local":
            J = 0.5 * (largest_B_n ** 2) / (np.linalg.norm(Bcoil[unravel_arg]) ** 2) * absn[unravel_arg]

        elif self.definition == "normalized":
            mod_Bcoil = np.linalg.norm(Bcoil, axis=2)
            num = largest_B_n ** 2 * absn[unravel_arg]
            denom = np.mean(mod_Bcoil**2 * absn)
            J =  0.5 * num / denom
        
        return J
    
    @derivative_dec
    def dJ(self):
        n = self.surface.normal()
        absn = np.linalg.norm(n, axis=2)
        unitn = n * (1. / absn)[:, :, None]
        Bcoil = self.field.B().reshape(n.shape)
        Bcoil_n = np.sum(Bcoil * unitn, axis=2)
        if self.target is not None:
            B_n = (Bcoil_n - self.target)
        else:
            B_n = Bcoil_n
        
        arg = np.argmax(np.abs(B_n))
        unravel_arg = np.unravel_index(arg, B_n.shape)
        largest_B_n = B_n[unravel_arg]
        
        B_n_eff = np.zeros_like(B_n)
        B_n_eff[unravel_arg] = largest_B_n
        
        if self.definition == "quadratic flux":
            dJdB = (B_n_eff[..., None] * unitn * absn[..., None]) / absn.size
            dJdB = dJdB.reshape((-1, 3))

        elif self.definition == "local":
            mod_Bcoil = np.linalg.norm(Bcoil, axis=2)
            dJdB = ((
                (B_n_eff/mod_Bcoil)[..., None] * (
                    unitn / mod_Bcoil[..., None] - (B_n_eff / mod_Bcoil**3)[..., None] * Bcoil
                )) * absn[..., None]) / absn.size

        elif self.definition == "normalized":
            mod_Bcoil = np.linalg.norm(Bcoil, axis=2)
            num = np.mean(B_n_eff**2 * absn)
            denom = np.mean(mod_Bcoil**2 * absn)

            dnum = 2 * (B_n_eff[..., None] * unitn * absn[..., None]) / absn.size
            ddenom = 2 * (Bcoil * absn[..., None]) / absn.size
            dJdB = 0.5 * (dnum / denom - num * ddenom / denom**2)

        else:
            raise ValueError("Should never get here")

        dJdB = dJdB.reshape((-1, 3))
        return self.field.B_vjp(dJdB)
'''

class Max_BdotN(Optimizable):
    def __init__(self, surface, field, target=None, definition="quadratic flux", num_BdotN=30):
        self.surface = surface
        if target is not None:
            self.target = np.ascontiguousarray(target)
        else:
            self.target = np.zeros(self.surface.normal().shape[:2])
        self.field = field
        xyz = self.surface.gamma()
        self.field.set_points(xyz.reshape((-1, 3)))
        if definition not in ["quadratic flux", "normalized", "local"]:
            raise ValueError("Unrecognized option for 'definition'.")
        self.definition = definition
        self.num_BdotN = num_BdotN
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[field])
    
    def J(self):
        n = self.surface.normal()
        absn = np.linalg.norm(n, axis=2)
        unitn = n * (1. / absn)[:, :, None]
        Bcoil = self.field.B().reshape(n.shape)
        Bcoil_n = np.sum(Bcoil * unitn, axis=2)
        if self.target is not None:
            B_n = (Bcoil_n - self.target)
        else:
            B_n = Bcoil_n
        
        lower_bound = np.sort(np.abs(B_n.flatten()))[-self.num_BdotN]
        arg_list = np.argwhere(np.abs(B_n) >= lower_bound)
        
        J=0
        
        if self.definition == "quadratic flux":
            for arg in arg_list:
                B_n_here = B_n[tuple(arg)]
                J += 0.5 * B_n_here ** 2 * absn[tuple(arg)]
            J = J / len(arg_list)

        elif self.definition == "local":
            for arg in arg_list:
                B_n_here = B_n[tuple(arg)]
                J += 0.5 * (B_n_here ** 2) / (np.linalg.norm(Bcoil[tuple(arg)]) ** 2) * absn[tuple(arg)]
            J = J / len(arg_list)

        elif self.definition == "normalized":
            num = 0
            denom = 0
            for arg in arg_list:
                B_n_here = B_n[tuple(arg)]
                mod_Bcoil = np.linalg.norm(Bcoil, axis=2)
                num += B_n_here ** 2 * absn[tuple(arg)]
                denom += np.mean(mod_Bcoil**2 * absn)
            J =  0.5 * num / denom
        
        return J
    
    @derivative_dec
    def dJ(self):
        n = self.surface.normal()
        absn = np.linalg.norm(n, axis=2)
        unitn = n * (1. / absn)[:, :, None]
        Bcoil = self.field.B().reshape(n.shape)
        Bcoil_n = np.sum(Bcoil * unitn, axis=2)
        if self.target is not None:
            B_n = (Bcoil_n - self.target)
        else:
            B_n = Bcoil_n
        
        lower_bound = np.sort(np.abs(B_n.flatten()))[-self.num_BdotN]
        arg_list = np.argwhere(np.abs(B_n) >= lower_bound)
        
        B_n_eff = np.zeros_like(B_n)
        for arg in arg_list:
            B_n_eff[tuple(arg)] = B_n[tuple(arg)]
        
        if self.definition == "quadratic flux":
            dJdB = (B_n_eff[..., None] * unitn * absn[..., None]) / absn.size
            dJdB = dJdB.reshape((-1, 3))

        elif self.definition == "local":
            mod_Bcoil = np.linalg.norm(Bcoil, axis=2)
            dJdB = ((
                (B_n_eff/mod_Bcoil)[..., None] * (
                    unitn / mod_Bcoil[..., None] - (B_n_eff / mod_Bcoil**3)[..., None] * Bcoil
                )) * absn[..., None]) / absn.size

        elif self.definition == "normalized":
            mod_Bcoil = np.linalg.norm(Bcoil, axis=2)
            num = np.mean(B_n_eff**2 * absn)
            denom = np.mean(mod_Bcoil**2 * absn)

            dnum = 2 * (B_n_eff[..., None] * unitn * absn[..., None]) / absn.size
            ddenom = 2 * (Bcoil * absn[..., None]) / absn.size
            dJdB = 0.5 * (dnum / denom - num * ddenom / denom**2)

        else:
            raise ValueError("Should never get here")

        dJdB = dJdB.reshape((-1, 3))
        return self.field.B_vjp(dJdB)
