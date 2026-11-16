package com.javallm.nn;

/**
 * Normalizes the final feature dimension of an input tensor.
 */
public interface Normalization extends Layer {

    int featureSize();
}
