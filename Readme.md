# BinEnhance<br>

################################################<br>
This is where BinEnhance's code and data are stored<br>
################################################<br>
[Eval]For now, only the code and data needed for the evaluation experiments in the paper have been released, and the complete code and data will be released gradually:<br>

1. Install the required environment, the python environment we use is python3.8<br>

```python
pip install -r Requirements.txt
```

2. Download the validation dataset from [OneDrive]([OneDrive (live.com)](https://onedrive.live.com/?authkey=!AEB85BVgg38gBkw&id=EA9FB056053D7CE5!106&cid=EA9FB056053D7CE5&parId=root&parQt=sharedby&o=OneUp)) :<br>

   This validation dataset contains embeddings of all functions in the validation dataset by the three methods used in the paper (Gemini, TREX, Asteria), as well as all embeddings enhanced by BinEnhance. There are three folders after unpacking the validation dataset, which are the three methods, and there are two files inside each folder, where `init_func_embeddings.json` is the embedding file obtained by the original method, and `BinEnhance_func_embeddings.json` is the embedding file after BinEnhance enhancement<br>

3. Use the storage path of the validation dataset downloaded in step 2 to run Eval.py<br>

```
python Eval.py --data-path="xxx/OriginMethods"
```

################################################<br>