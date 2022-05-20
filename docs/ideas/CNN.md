# Filters
Generally speaking they take image, process it and return some transformed image.

Filters are described using "masks", they can have any shape, but need to be smaller than the input.

The identity mask is given by
\begin{align}
  M =
  \begin{pmatrix}
    \nicefrac{1}{3}  &\nicefrac{1}{3}  & \nicefrac{1}{3}\\
    \nicefrac{1}{3} &\nicefrac{1}{3}  &\nicefrac{1}{3}\\
    \nicefrac{1}{3} &\nicefrac{1}{3} &\nicefrac{1}{3}
  \end{pmatrix}
\end{align}

To apply a filter we shift the filter mask over the image, processing each pixel individually.

On edges:
- ignore values
- extend image with specific values

## Example masks

### Average blur

```
M = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
```
### gaussian blur

```
M = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
```


# Convolutional Neural Networks
- Combine DNNs and Kernel Convolution.

## DNNs

A set of input parameters, fully connected with several hidden layers of neurons in different sizes. In each layer every neuron is connected to all neurons in the previous and following layers.

## CNNs
Theoretically replace each neuron with a filter mask.
