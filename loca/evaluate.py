from models.loca import build_model
from utils.data import FSC147Dataset
from utils.arg_parser import get_argparser

import argparse
import os

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist


@torch.no_grad()
def evaluate(args):
    device = torch.device('cpu')
    print("Запуск инференса на CPU...")

    model = build_model(args).to(device)

    model_file = os.path.join(args.model_path, f'{args.model_name}.pt')

    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    for split in ['val', 'test']:
        test = FSC147Dataset(
            args.data_path,
            args.image_size,
            split=split,
            num_objects=args.num_objects,
            tiling_p=args.tiling_p,
        )
        test_loader = DataLoader(
            test,
            batch_size=args.batch_size,
            drop_last=False,
            num_workers=args.num_workers
        )
        ae = torch.tensor(0.0).to(device)
        se = torch.tensor(0.0).to(device)
        model.eval()
        for step, (img, bboxes, density_map) in enumerate(test_loader):
            if step >= 100:
                print(f"Достигнут лимит в 100 батчей. Завершаем проверку {split}...")
                break
            if step % 5 == 0:
                print(f"Обработка {split}: батч {step}/{len(test_loader)}")
            img = img.to(device)
            bboxes = bboxes.to(device)
            density_map = density_map.to(device)

            out, _ = model(img, bboxes)

            pred_count = out.flatten(1).sum(dim=1).item()
            gt_count = density_map.flatten(1).sum(dim=1).item()

            # Вывод в консоль для каждого батча
            print(f"[{split.upper()}] Батч {step:4d} | Предсказано: {pred_count:6.2f} | Реально: {gt_count:6.2f} | Ошибка: {abs(pred_count - gt_count):6.2f}")

        
            ae += torch.abs(
                density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
            ).sum()
            se += ((
                density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
            ) ** 2).sum()

        print(
            f"\n{split.capitalize()} set",
            f"MAE: {ae.item() / len(test):.2f}",
            f"RMSE: {torch.sqrt(se / len(test)).item():.2f}\n",
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LOCA', parents=[get_argparser()])
    args = parser.parse_args()
    evaluate(args)
