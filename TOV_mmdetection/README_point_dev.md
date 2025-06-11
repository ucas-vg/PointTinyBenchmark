# point Experiment
## TinyPerson
[TinyPerson](configs2/TinyPerson/TinyPerson.md)

## TinyCOCO
[TinyCOCO](configs2/TinyCOCO/TinyCOCO.md)

## visDronePerson
[visDronePerson](configs2/visDronePerson/visDronePerson.md)

# semi supervised

```bash
export ANN_INFO="noise_uniform_1/corner_sw640_sh512_old/pseuw16h16"
python huicv/coarse_utils/generate_semi_annotation.py \
    data/tiny_set/mini_annotations/coarse_gen/${ANN_INFO}/tiny_set_train_sw640_sh512_all_erase_coarse.json \
    data/tiny_set/mini_annotations/coarse_gen/${ANN_INFO}/tiny_set_train_sw640_sh512_all_erase_coarse_semi0.2.json \
    --fully_ratio 0.2
```

Adap RepPoint, TinyPerson

fully ratio | round0 | round1| round2
--- | --- | --- | ---
0   | 30.73 | 54.07 | 
0.2 | 40.99 | 52.06 | 47.03
0.4 | hzx | |
0.6 | 47.08 | 55.63 | 43.77
1.0 | | |