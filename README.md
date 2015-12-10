# torch-inception

This is a fork of [jcjohnson's neural art style](https://github.com/jcjohnson/neural-style) project, modified to generate images using the techniques described in the Google Research blog [Inceptionism](http://googleresearch.blogspot.com/2015/06/inceptionism-going-deeper-into-neural.html). Hence the name "torch-inception"! I considered calling it nightmare-sauce, but decided that's for another project.

Most of what I did was delete code that wasn't relevant in this use case. The main addition is to multiply the syle layer's activations by the "acid_dose" constant (which can be negative!!) in order to train the image to accentuate that layer's features. Also I changed the default tag of gpu to be -1 so newbs that don't read README's won't think it requires a gpu ;)

I omit lbfgs as an optimization option because for some reason it always stops early with this technique.

Steps:

0) Install dependencies:

* [torch7](https://github.com/torch/torch7)
* [loadcaffe](https://github.com/szagoruyko/loadcaffe)

Optional:
* CUDA 6.5+
* [cudnn.torch](https://github.com/soumith/cudnn.torch)


1) Pick a model to use (some are provided in the models folder)
2) Pick an input image
3) Pick a level of abstaction ("macro" relu5_1, relu4_1, ..., relu1_1 "micro") (conv5_4 etc is also available)
4) Run a command that looks like this
```
th neural_dream.lua -content_image ./examples/inputs/frida_kahlo.jpg -style_layers relu5_1 -save_iter 1 -num_iterations 30 -gpu -1 -acid_dose 1000
```

It can make weird images. 

<img src="https://raw.githubusercontent.com/snakecharmer1024/neural-style/master/examples/inputs/golden_gate.jpg" height="160px">
<img src="https://raw.githubusercontent.com/snakecharmer1024/neural-style/master/examples/outputs/golden_gate_relu_5.png" height="160px">
<img src="https://raw.githubusercontent.com/snakecharmer1024/neural-style/master/examples/inputs/hoovertowernight.jpg" height="160px">
<img src="https://raw.githubusercontent.com/snakecharmer1024/neural-style/master/examples/outputs/stanford.png" height="160px">
<img src="https://raw.githubusercontent.com/snakecharmer1024/neural-style/master/examples/inputs/escher_sphere.jpg" height="160px">
<img src="https://raw.githubusercontent.com/snakecharmer1024/neural-style/master/examples/outputs/escher_relu_5.png" height="160px">


Depending on the dose, it can make brad look very different.

<img src="https://raw.githubusercontent.com/snakecharmer1024/neural-style/master/examples/inputs/brad_pitt.jpg" height="160px">
<img src="https://raw.githubusercontent.com/snakecharmer1024/neural-style/master/examples/outputs/acid_brad_10_ug.png" height="160px">
<img src="https://raw.githubusercontent.com/snakecharmer1024/neural-style/master/examples/outputs/acid_brad_100_ug.png" height="160px">
<img src="https://raw.githubusercontent.com/snakecharmer1024/neural-style/master/examples/outputs/acid_brad_1000_ug.png" height="160px">


Which layer you choose can make a big difference. The above were taken from relu5_1. If we change to conv5_4, it gives brad a third eye.

<img src="https://raw.githubusercontent.com/snakecharmer1024/neural-style/master/examples/outputs/third_eye_bradd.png" height="160px">

There's significant difference between nearby conv layers. These are taken from conv5_1 and conv5_4, respectively.

<img src="https://raw.githubusercontent.com/snakecharmer1024/neural-style/master/examples/outputs/matisse_conv5_1.png" height="160px">
<img src="https://raw.githubusercontent.com/snakecharmer1024/neural-style/master/examples/outputs/matisse_conv5_4.png" height="160px">

Looking across the network gives a good feel for what the network is doing. These are taken from relu1_1, relu2_1, ..., relu5_1.

<img src="https://raw.githubusercontent.com/snakecharmer1024/neural-style/master/examples/outputs/tubingen_relu_1.png" height="160px">
<img src="https://raw.githubusercontent.com/snakecharmer1024/neural-style/master/examples/outputs/tubingen_relu_2.png" height="160px">
<img src="https://raw.githubusercontent.com/snakecharmer1024/neural-style/master/examples/outputs/tubingen_relu_3.png" height="160px">
<img src="https://raw.githubusercontent.com/snakecharmer1024/neural-style/master/examples/outputs/tubingen_relu_4.png" height="160px">
<img src="https://raw.githubusercontent.com/snakecharmer1024/neural-style/master/examples/outputs/tubingen_relu_5.png" height="160px">


You can even do a negative dose, which results in subtracting the abstraction layer from the image.

<img src="https://raw.githubusercontent.com/snakecharmer1024/neural-style/master/examples/outputs/tubingen_negative_relu_1.png" height="160px">
<img src="https://raw.githubusercontent.com/snakecharmer1024/neural-style/master/examples/outputs/tubingen_negative_relu_2.png" height="160px">
<img src="https://raw.githubusercontent.com/snakecharmer1024/neural-style/master/examples/outputs/tubingen_negative_relu_3.png" height="160px">
<img src="https://raw.githubusercontent.com/snakecharmer1024/neural-style/master/examples/outputs/tubingen_negative_relu_4.png" height="160px">
<img src="https://raw.githubusercontent.com/snakecharmer1024/neural-style/master/examples/outputs/tubingen_negative_relu_5.png" height="160px">


**Options**:
* `-image_size`: Maximum side length (in pixels) of of the generated image. Default is 512.
* `-gpu`: Zero-indexed ID of the GPU to use; for CPU mode set `-gpu` to -1.

**Optimization options**:
* `-content_weight`: How much to weight the content reconstruction term. Default is 5e0.
* `-num_iterations`: Default is 1000.
* `-learning_rate`: Learning rate to use with the ADAM optimizer. Default is 1e1.
* `-normalize_gradients`: If this flag is present, style and content gradients from each layer will be
  L1 normalized. Idea from [andersbll/neural_artistic_style](https://github.com/andersbll/neural_artistic_style).

**Output options**:
* `-output_image`: Name of the output image. Default is `out.png`.
* `-print_iter`: Print progress every `print_iter` iterations. Set to 0 to disable printing.
* `-save_iter`: Save the image every `save_iter` iterations. Set to 0 to disable saving intermediate results.

**Layer options**:
* `-style_layers`: Comman-separated list of layer names to use for style reconstruction.
  Default is `relu1_1,relu2_1,relu3_1,relu4_1,relu5_1`.

**Other options**:
* `-style_scale`: Scale at which to extract features from the style image. Default is 1.0.
* `-proto_file`: Path to the `deploy.txt` file for the VGG Caffe model.
* `-model_file`: Path to the `.caffemodel` file for the VGG Caffe model.
  Default is the original VGG-19 model; you can also try the normalized VGG-19 model used in the paper.
* `-pooling`: The type of pooling layers to use; one of `max` or `avg`. Default is `max`.
  The VGG-19 models uses max pooling layers, but the paper mentions that replacing these layers with average
  pooling layers can improve the results. I haven't been able to get good results using average pooling, but
  the option is here.
* `-backend`: `nn` or `cudnn`. Default is `nn`. `cudnn` requires
  [cudnn.torch](https://github.com/soumith/cudnn.torch) and may reduce memory usage.

## Frequently Asked Questions

**Problem:** Generated image has saturation artifacts:

<img src="https://cloud.githubusercontent.com/assets/1310570/9694690/fa8e8782-5328-11e5-9c91-11f7b215ad19.png">

**Solution:** Update the `image` packge to the latest version: `luarocks install image`

**Problem:** Running without a GPU gives an error message complaining about `cutorch` not found

**Solution:**
Pass the flag `-gpu -1` when running in CPU-only mode

**Problem:** The program runs out of memory and dies

**Solution:** Try reducing the image size: `-image_size 256` (or lower). Note that different image sizes will likely
require non-default values for `-style_weight` and `-content_weight` for optimal results.
If you are running on a GPU, you can also try running with `-backend cudnn` to reduce memory usage.

**Problem:** Get the following error message:

`models/VGG_ILSVRC_19_layers_deploy.prototxt.cpu.lua:7: attempt to call method 'ceil' (a nil value)`

**Solution:** Update `nn` package to the latest version: `luarocks install nn`

**Problem:** Get an error message complaining about `paths.extname`

**Solution:** Update `torch.paths` package to the latest version: `luarocks install paths`

## Memory Usage
By default, `neural-style` uses the `nn` backend for convolutions and L-BFGS for optimization.
These give good results, but can both use a lot of memory. You can reduce memory usage with the following:

* **Use cuDNN**: Add the flag `-backend cudnn` to use the cuDNN backend. This will only work in GPU mode.
* **Use ADAM**: Add the flag `-optimizer adam` to use ADAM instead of L-BFGS. This should significantly
  reduce memory usage, but may require tuning of other parameters for good results; in particular you should
  play with the learning rate, content weight, style weight, and also consider using gradient normalization.
  This should work in both CPU and GPU modes.
* **Reduce image size**: If the above tricks are not enough, you can reduce the size of the generated image;
  pass the flag `-image_size 256` to generate an image at half the default size.
  
With the default settings, `neural-style` uses about 3.5GB of GPU memory on my system;
switching to ADAM and cuDNN reduces the GPU memory footprint to about 1GB.

## Speed
On a GTX Titan X, running 1000 iterations of gradient descent with `-image_size=512` takes about 2 minutes.
In CPU mode on an Intel Core i7-4790k, running the same takes around 40 minutes.
Most of the examples shown here were run for 2000 iterations, but with a bit of parameter tuning most images will
give good results within 1000 iterations.

## Implementation details
Images are initialized with white noise and optimized using L-BFGS.

We perform style reconstructions using the `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, and `conv5_1` layers
and content reconstructions using the `conv4_2` layer. As in the paper, the five style reconstruction losses have
equal weights.
