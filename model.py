import torch

class AutoEncoder(torch.nn.Module):
    def __init__(self, input_channels=3, output_channels=None, init_features=16) -> None:
        super().__init__()

        features = init_features

        self.encoder1 = AutoEncoder.block(in_channels=input_channels, features=features)
        self.encoder2 = AutoEncoder.block(in_channels=features, features=features)
        self.encoder3 = AutoEncoder.block(in_channels=features, features=features*2, stride=2)
        self.encoder4 = AutoEncoder.block(in_channels=features*2, features=features*2)
        self.encoder5 = AutoEncoder.block(in_channels=features*2, features=features*4, stride=2)
        self.encoder6 = AutoEncoder.block(in_channels=features*4, features=features*4)
        
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=features*256, out_features=1000),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=1000, out_features= features*256),
            torch.nn.ReLU()
        )

        self.decoder1 = AutoEncoder.block_d(in_channels=features*4, features=features*4)
        self.decoder2 = AutoEncoder.block_d(in_channels=features*4, features=features*2, stride=2)
        self.decoder3 = AutoEncoder.block_d(in_channels=features*2, features=features*2)
        self.decoder4 = AutoEncoder.block_d(in_channels=features*2, features=features, stride=2)
        self.decoder5 = AutoEncoder.block_d(in_channels=features, features=features)
        self.output = torch.nn.ConvTranspose2d(in_channels=features, out_channels=input_channels, kernel_size=3, padding=1)

    def forward(self, sample):
        encoder1 = self.encoder1(sample)
        encoder2 = self.encoder2(encoder1)
        encoder3 = self.encoder3(encoder2)
        encoder4 = self.encoder4(encoder3)
        encoder5 = self.encoder5(encoder4)
        encoder6 = self.encoder6(encoder5)
        bottleneck = self.bottleneck(encoder6)
        decoder1 = self.decoder1(bottleneck)
        decoder2 = self.decoder2(decoder1)
        decoder3 = self.decoder3(decoder2)
        decoder4 = self.decoder4(decoder3)
        decoder5 = self.decoder5(decoder4)
        output = self.output(decoder5)
        
        return output


    @staticmethod
    def block(in_channels, features, stride=1):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3, stride=stride, padding=1, bias=False),
            torch.nn.ReLU(inplace=True),      
        )
    @staticmethod   
    def block_d(in_channels, features, stride=1):
        return torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=features, kernel_size=3, stride=stride, padding=1, bias=False),
            torch.nn.ReLU()
        )
    
