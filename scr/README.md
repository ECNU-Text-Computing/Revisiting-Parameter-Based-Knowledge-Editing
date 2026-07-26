Example command:

```python
python edit_rag.py \
  --model_path /path/to/llama3-8b \
  --retriever_type contriever-ms \  
  --retriever_path /path/to/contriever-msmarco \
  --dataset_path /path/to/test_cf.json \
  --memory_path /path/to/wiki_counterfact-test-all-sentence.json \
  --top_k 5 \
  --eval_metric contain \
  --summary \
  --edit_scene single
```


```bash
python edit_rag.py \
  --model_path /home/wyren/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
  --retriever_type contriever-ms \  
  --retriever_path /home/wyren/.cache/huggingface/hub/models--facebook--contriever-msmarco/snapshots/abe8c1493371369031bcb1e02acb754cf4e162fa \
  --dataset_path /home/wyren/Knowledge-Editing-Benchmark/wyren/dataset/wiki_counterfact/wiki_counterfact-test-all-sentence.json \
  --memory_path /path/to/wiki_counterfact-test-all-sentence.json \
  --top_k 5 \
  --eval_metric contain \
  --summary \
  --memory_start_index 0 \
  --memory_end_index 1 \ # 依次设为1，10，100，1000
  --edit_scene sequential  # 连续编辑场景下
```
