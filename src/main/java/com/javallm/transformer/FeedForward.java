package com.javallm.transformer;

import com.javallm.nn.Layer;

/**
 * Hidden-size-preserving feed-forward sublayer used by a transformer block.
 *
 * <p>Dense implementations and future routed expert implementations can share
 * this contract while keeping their calculations independent.</p>
 */
public interface FeedForward extends Layer {

    int hiddenSize();
}
