# [CEGARETE_NN]()

The Repository contains the implementation of the paper 
Tighter Abstract Queries in Neural Network Verification
by Elazar Coher, Yizhak Yisrael Elboher, Clark Barrett and Guy Katz.

## Quick start

### MNIST

```bash
python main.py \
  --method "abstraction_refinement" \
  --output-path "./experiments/temp_experiment" \
  --property "./mnist/model2/file_properties/180_minst_property_image_5.py" \
  --network-path "tf" "./mnist/model2/model_without_softmax" \
  --abstraction-strategy "complete_right_to_left" \
  --abstraction-args "" \
  --refinement-strategy "by_max_loss" \
  --refinement-args "1"  \
  --update-property "Marabou" \
  --refine-until-not-satisfying
```


