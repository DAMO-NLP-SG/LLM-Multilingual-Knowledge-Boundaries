# LLM-Multilingual-Knowledge-Boundaries
Code and datasets for paper "Analyzing LLMs' Knowledge Boundary Cognition Across Languages Through the Lens of Internal Representations" to appear at ACL 2025.

We are updating all code and resources.

Dataset links:
[FreshQA-multilingual](https://huggingface.co/datasets/SeaLLMs/FreshQA-multilingual); [FreshQA-multilingual-augmented](https://huggingface.co/datasets/SeaLLMs/FreshQA-multilingual-augmented); [True-False-multilingual](https://huggingface.co/datasets/SeaLLMs/TrueFalse-Statements-multilingual); [SeaRefuse](https://huggingface.co/datasets/SeaLLMs/SeaRefuse-test)

Code for linear probe, and using mean-shifting \& linear projection to align language subspaces.
```
python inference.py \
    --model_name Qwen/Qwen2.5-7B \
    --dataset_name SeaLLMs/FreshQA-multilingual \
    --output_path "./transferability_results/7B/Qwen_base_7B.json" \
    --methods "identical" "mean shifting" "linear projection" \
    --use_template True \
    --batch_size 50
```

```
@misc{xiao2025analyzingllmsknowledgeboundary,
      title={Analyzing LLMs' Knowledge Boundary Cognition Across Languages Through the Lens of Internal Representations}, 
      author={Chenghao Xiao and Hou Pong Chan and Hao Zhang and Mahani Aljunied and Lidong Bing and Noura Al Moubayed and Yu Rong},
      year={2025},
      eprint={2504.13816},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.13816}, 
}
```
