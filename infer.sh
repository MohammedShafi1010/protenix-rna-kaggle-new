CUDA_VISIBLE_DEVICES=1  protenix predict \
    --input ./examples/casp16_part.json \
    --out_dir  ./bytedance_protenix_out_bfp16_comp_416_no_msa   --seeds 101 \
    --use_msa false