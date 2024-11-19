import timm
import torch

def main():
    # Load the MobileNet model from timm
    model = timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k', pretrained=True)
    model.eval()

    # Check if MPS is available and move model to MPS if it is
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)

    print(f"Model Loaded Successfully on {device}")

if __name__ == "__main__":
    main()



