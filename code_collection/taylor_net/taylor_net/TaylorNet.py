import torch
import torch.nn as nn

torch.manual_seed(42)
class TaylorNet(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_taylor_monomials: int):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_taylor_monomials = num_taylor_monomials
        
        self.taylor_coefficients = nn.ParameterDict()
        self.taylor_offsets = nn.Parameter(torch.randn((input_dim,),
                                                       requires_grad=True,
                                                       dtype=torch.float))
        
        for i in range(num_taylor_monomials):
            shape = (self.input_dim, ) * i if i > 0 else ()
            self.taylor_coefficients[f'monomial_{i}'] = nn.Parameter(torch.randn(shape, 
                                                                                 requires_grad=True, 
                                                                                 dtype=torch.float))

    def forward(self, x: torch.Tensor):
        """ Return functional value of X, approximated by taylor series.
        Args:
            x (torch.Tensor): shape: (B, d)
        """
        
        x_reshaped = x.detach()

        relative_x = x_reshaped - self.taylor_offsets
        
        out = self.taylor_coefficients[f'monomial_{0}']
        for i in range(self.num_taylor_monomials-1):
            einsum_str = 'bd' +''.join([f', b{"efghijklmnopqrstuvw"[z]}' for z in range(i)])
            einsum_str += ', d'+ ''.join("efghijklmnopqrstuvw"[:i])
            einsum_str += ' -> b'

            args = [relative_x] * (i+1)
            
            monomial= torch.einsum(einsum_str, *args, 
                                   self.taylor_coefficients[f'monomial_{i+1}'])
            out = out + monomial

        return out
    