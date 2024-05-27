CUDA_VISIBLE_DEVICES=1 python WCE.py --ds wilt --alpha 0.8 --seed $1
CUDA_VISIBLE_DEVICES=1 python WCE.py --ds monks-3 --alpha 0.8 --seed $1
CUDA_VISIBLE_DEVICES=1 python WCE.py --ds breast-cancer-wisc --alpha 0.8 --seed $1
CUDA_VISIBLE_DEVICES=1 python WCE.py --ds eyepacs --alpha 0.8 --seed $1
CUDA_VISIBLE_DEVICES=1 python WCE.py --ds ade_v2 --alpha 0.8 --seed $1