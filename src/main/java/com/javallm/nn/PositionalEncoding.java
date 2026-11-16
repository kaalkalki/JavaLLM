package com.javallm.nn;

import com.javallm.tensor.Tensor;

/**
 * Applies positional information to a tensor, with support for offset positions.
 */
public interface PositionalEncoding extends Layer {

    Tensor forward(Tensor input, int startPosition);
}
