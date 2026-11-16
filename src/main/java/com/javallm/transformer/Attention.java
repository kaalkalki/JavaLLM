package com.javallm.transformer;

import com.javallm.nn.Layer;
import com.javallm.tensor.Tensor;

/**
 * Attention sublayer contract for a decoder transformer block.
 *
 * <p>The cache-aware entry point uses the engine's shared key/value cache. The
 * interface does not prescribe a positional encoding or a relationship between
 * query-head and cache-head counts.</p>
 */
public interface Attention extends Layer {

    Tensor forwardWithCache(Tensor input, int startPosition, KeyValueCache cache, int layerIndex);

    int cacheHeadCount();

    int cacheHeadDimension();

    int hiddenSize();
}
