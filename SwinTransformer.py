import torch.nn as nn

class PatchPartition(nn.Module):
    def __init__(self,
                 patch_size: int = 4,
                 ):
        """
        this patch partition + Linear Embedding
        :param patch_size:
        """
        super().__init__()
        self.proj = nn.Conv2d(3, 96, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(96)

    def forward(self, x):
        x = self.proj(x)                  # [B, 96, 56, 56]
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
    
