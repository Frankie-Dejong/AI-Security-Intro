# Eval Scripts for Collaborative Diffusion Model

We calculate the evaluation metrics below for the collaborative diffusion model. 
- FID score
- Clip score
- Mask Accuracy

We construct test set using 27000-29000 sample from the CelebA dataset, and generate one image for each sample. 

After that, we randomly sample 50 images for 20 times to calculate FID score and Clip score. We also calculate the mask accuracy for the 2000 images in the testset. The result shows like:

|| FID score | Clip score | Mask Accuracy |
|-|-----------|------------|---------------|
|Ours| 132.8410      | 0.2420        | 0.8409          |
|Official| 111.36 | 0.2451 | 0.8025 |

If you want to reproduce the results, follow the steps below:

1. Generate testset mentioned above using [Collaborative Diffusion Model](https://github.com/ziqihuangg/Collaborative-Diffusion).
2. Place this repository in the sub directory as the Collaborative Diffusion Model like `/Collaborative-Diffusion/eval_scripts/`
3. Download the face parsing network provided by [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ?tab=readme-ov-file), then place it in the './face_parsing/models/parse_net' directory.
4. run `run_test.sh` in `./eval_scripts/face_parsing/` to generate mask for the 2000 images in the testset.
5. run `python eval_scripts/evaluate.py` for metrics calculation.

## Note

Our results may be different from the official results due to the choice of testset, especially on the FID score calculation. However, the FID score on the whole 2000 images is `19.2408`, which is much closer to the `17.27` calculated on the 3000 images.(refered to this [issue](https://github.com/ziqihuangg/Collaborative-Diffusion/issues/30))
