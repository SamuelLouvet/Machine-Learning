import argparse
import os
import torch
import torch.nn as nn

from net import Net
from load_data import create_data


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate face detector CNN')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--train-dir', default='./train_images')
    parser.add_argument('--test-dir', default='./test_images')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--valid-size', type=float, default=0.2)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--image-size', type=int, default=32)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def evaluate(model, loader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / max(total, 1), correct / max(total, 1)


def main():
    args = parse_args()
    device = torch.device(args.device)

    train_loader, valid_loader, test_loader, classes = create_data(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        batch_size=args.batch_size,
        valid_size=args.valid_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = Net().to(device)
    model.load_state_dict(checkpoint['model_state'])

    v_loss, v_acc = evaluate(model, valid_loader, device)
    t_loss, t_acc = evaluate(model, test_loader, device)
    print(f'Valid - loss {v_loss:.4f} acc {v_acc:.4f}')
    print(f'Test  - loss {t_loss:.4f} acc {t_acc:.4f}')


if __name__ == '__main__':
    main()


