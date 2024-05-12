from Evaluator import FIDEvaluator, ClipEvaluator, MaskAccuracyEvaluator
import argparse
import random

def fid(args):
    fids = []
    for _ in range(20):
        fidEvaluator = FIDEvaluator(args.gt_path, args.pred_path, batch_idx=random.randint(0, 50))
        fid = fidEvaluator.evaluate()
        fids.append(fid)
    print(f'FID: {sum(fids) / 20}, best: {min(fids)}')

def clip(args):
    scores = []
    best_res = 0
    for _ in range(20):
        clipEvaluator = ClipEvaluator(args.pred_path, args.text_path)
        clip_score, max_score = clipEvaluator.evaluate()
        scores.append(clip_score)
        best_res = max(max_score, best_res)
    print(f'Clip Score: {sum(scores) / 20}, best: {best_res}')

def mask(args):
    maskEvaluator = MaskAccuracyEvaluator(args.mask_path, args.prompt_mask_path)
    acc = maskEvaluator.evaluate()
    print(f'Mask Accuracy: {acc}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', default='./datasets/gt_images')
    parser.add_argument('--pred_path', default='./datasets/pred_images')
    parser.add_argument('--text_path', default='./datasets/text/captions_hq_beard_and_age_2022-08-19.json')
    parser.add_argument('--mask_path', default='./eval/face_parsing/test_results')
    parser.add_argument('--prompt_mask_path', default='./datasets/mask/CelebAMask-HQ-mask-color-palette_32_nearest_downsampled_from_hq_512_one_hot_2d_tensor')
    parser.add_argument('--fid', action="store_true", default=False)
    parser.add_argument('--clip', action="store_true", default=False)
    parser.add_argument('--mask', action="store_true", default=False)
    args = parser.parse_args()
    
    if args.fid:
        fid(args)
    elif args.clip:
        clip(args)
    elif args.mask:
        mask(args)
    else:
        raise ValueError("Unexpected Choices")