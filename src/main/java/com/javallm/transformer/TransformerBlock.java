package com.javallm.transformer;

import com.javallm.nn.Layer;
import com.javallm.tensor.Tensor;

/**
 * Decoder transformer block contract for ordinary and cache-aware evaluation.
 */
public interface TransformerBlock extends Layer {

    Tensor forwardWithCache(Tensor input, int startPosition, KeyValueCache cache, int layerIndex);

    int hiddenSize();

    int cacheHeadCount();

    int cacheHeadDimension();
}
