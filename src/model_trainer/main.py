import argparse
import os

import torch.multiprocessing as mp

from model_trainer.utils.quick_start import quick_start


os.environ.setdefault('NUMEXPR_MAX_THREADS', '48')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='electricity', help='name of datasets')
    parser.add_argument(
        '--use-primitive',
        choices=['true', 'false'],
        default=None,
        help='override yaml use_primitive for FNSPID 双嵌入选型（true/false）',
    )
    parser.add_argument('--gpu', '-g', type=int, default=0, help='gpu device')
    args, _ = parser.parse_known_args()
    config_dict = {
        'gpu_id': args.gpu,
        # Use repository-local dataset path to avoid absolute path issues
        'data_path': os.path.join(os.getcwd(), 'dataset/'),
    }
    if args.use_primitive is not None:
        config_dict['use_primitive'] = args.use_primitive.lower() == 'true'

    mp.set_start_method('spawn')  # 必须在主程序中设置
    
    quick_start(dataset=args.dataset, config_dict=config_dict, save_model=True)
