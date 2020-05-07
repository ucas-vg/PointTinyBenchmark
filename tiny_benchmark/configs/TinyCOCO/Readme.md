1. you need download dataset annotation to ${YOUR_DATASET_DIR} first.
2. you need change the line in your config file, assume your annotation is _\${YOUR\_DATASET\_DIR}/tiny\_set/erase\_with\_uncertain\_dataset/annotations/corner/task/tiny\_set\_train\_sw640\_sh512\_all.json_

```yaml
    TARGET_ANNO_FILE: '${YOUR_DATASET_DIR}/tiny_set/erase_with_uncertain_dataset/annotations/corner/task/tiny_set_train_sw640_sh512_all.json'
```


3. if you use monotonicity scale match and COCO as dataset E, you may need [instances_simple_merge2014.json](), which merge all image annotations in train set and valid set. you can download it from our merged version, or merged by yourself.

instances_simple_merge2014.json download link:<br/>
[Baidu Pan](https://pan.baidu.com/s/1_bEutedc3dz9DSR4v7xmVA): 545e<br/>
[Google Driver]: <br/>

and you need change line in config file, assume your download it to _\${YOUR\_DATASET\_DIR}/coco/annotations/instances\_simple\_merge2014.json_

```yaml
    SOURCE_ANNO_FILE: '${YOUR_DATASET_DIR}/coco/annotations/instances_simple_merge2014.json'
```

