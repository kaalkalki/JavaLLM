package com.javallm.model;

import com.javallm.tensor.Tensor;
import com.javallm.transformer.KeyValueCache;

/**
 * Decoder model contract used by the loading and inference layers.
 *
 * <p>Implementations return vocabulary logits for token sequences and expose
 * cache-aware forward calls for autoregressive generation.</p>
 */
public interface LanguageModel {

    int vocabularySize();

    int hiddenSize();

    int layerCount();

    Tensor forward(int... tokenIds);

    KeyValueCache newKeyValueCache(int maximumSequenceLength);

    Tensor forwardWithCache(int[] tokenIds, KeyValueCache cache);
}
