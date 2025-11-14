from typing import List, Optional
import torch
import torch.nn as nn

import math


class SimpleCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 3,
        channels: Optional[List[int]] = None,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [32, 64, 128]

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.channels = channels
        self.dropout = dropout

        # Feature extractor
        blocks = []
        in_c = in_channels
        for out_c in channels:
            blocks.append(self._make_block(in_c, out_c))
            in_c = out_c
        self.features = nn.Sequential(*blocks)

        # Pool & mlp
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels[-1], max(32, channels[-1] // 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(max(32, channels[-1] // 2), num_classes),
        )

        # Initialize weights
        self.reset_parameters()

    def _make_block(self, in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)  # shape (B, channels[-1])
        x = self.classifier(x)
        return x

    def count_params(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def feature_map_size(self, input_size: int) -> int:
        return 1  # since AdaptiveAvgPool2d((1,1))

# Example usage
if __name__ == "__main__":
    import math
    model = SimpleCNN(num_classes=2, in_channels=3)

    print("Params:", model.count_params())
    dummy = torch.randn(16, 3, 224, 224)
    if torch.cuda.is_available():
        model = model.cuda()
        dummy = dummy.cuda()

    try:
        model = torch.compile(model)    # type: ignore
    except Exception:
        pass

    # model.eval()
    # with torch.no_grad():
    out = model(dummy)

    print("Output shape:", out.shape)  # expect (8, num_classes)

